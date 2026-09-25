import copy
import math

import numpy as np
import torch
from torch import einsum, nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many


def exists(val):
    return val is not None


from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DynamicHead(nn.Module):

    def __init__(self,
                 num_classes,
                 d_model,
                 pooler_resolution,
                 strides,
                 in_channels,
                 dim_feedforward=2048,
                 nhead=8,
                 dropout=0.0,
                 activation="relu",
                 num_heads=6,
                 return_intermediate=True,
                 use_focal=False,
                 use_fed_loss=False,
                 prior_prob=0.01
                 ):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(pooler_resolution, strides, in_channels)
        self.box_pooler = box_pooler

        # Build heads.
        rcnn_head = RCNNHead(d_model, num_classes, pooler_resolution, dim_feedforward, nhead, dropout, activation,
                             use_focal=use_focal, use_fed_loss=use_fed_loss)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads
        self.return_intermediate = return_intermediate

        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(pooler_resolution, strides, in_channels):

        pooler_scales = [1 / s for s in strides]
        sampling_ratio = 2
        pooler_type = "ROIAlignV2"

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, t, lost_features=None, fix_ref_boxes=False):
        # assert t shape (batch_size)
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_association_scores = []

        bboxes = init_bboxes
        proposal_features = None

        for head_idx, rcnn_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features, association_score_logits = rcnn_head(features, bboxes,
                                                                                               proposal_features,
                                                                                               self.box_pooler, time,
                                                                                               lost_features,
                                                                                               fix_ref_boxes)
            if self.return_intermediate:
                inter_class_logits.append(torch.cat(class_logits, dim=0))
                inter_pred_bboxes.append(torch.cat(pred_bboxes, dim=0))
                inter_association_scores.append(torch.sigmoid(association_score_logits))
            bboxes = (pred_bbox.detach() for pred_bbox in pred_bboxes)

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), torch.stack(
                inter_association_scores)


        return torch.cat(class_logits, dim=0)[None], torch.cat(pred_bboxes, dim=0)[None], \
               torch.sigmoid(association_score_logits)[None]


class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, agent_num=49, window_size=7, **kwargs):
        super(AgentAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.agent_num = agent_num
        self.scale = self.head_dim ** -0.5
        # self.window_size = window_size

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)

        # 代理令牌作为可学习的参数
        self.agent_tokens = nn.Parameter(torch.randn(agent_num, num_heads, self.head_dim))

    def forward(self, q, k, v):
        b, n, c = v.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        # 生成 Q, K, V
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        # 调整 q 的形状以进行池化操作
        q_reshaped = q.reshape(b * n * num_heads, head_dim)

        # 通过池化生成代理令牌
        agent_tokens = F.adaptive_avg_pool1d(q_reshaped, 1).squeeze(-1)
        agent_tokens = agent_tokens.view(b, n, num_heads, -1)

        # 将代理令牌重塑为适合进一步处理的形状
        agent_tokens = agent_tokens.permute(0, 3, 1, 2)  # 形状变为 [b, agent_num, n, head_dim]

        # 调整 k 的形状以进行矩阵乘法
        k_reshaped = k.permute(0, 3, 2, 1)
        v = v.permute(0, 3, 2, 1)  # 形状变

        # 代理聚合
        agent_features = (agent_tokens * self.scale) @ k_reshaped.transpose(-2, -1)
        agent_features = self.softmax(agent_features)
        agent_features = self.attn_drop(agent_features)
        agent_output = agent_features @ v

        # 代理广播
        # 将代理令牌作为键，广播到所有查询
        agent_output = agent_output.permute(0, 3, 2, 1)
        q_broadcast = (q * self.scale) @ agent_output.transpose(-2, -1)
        q_broadcast = self.softmax(q_broadcast)
        q_broadcast = self.attn_drop(q_broadcast)

        # 计算最终输出
        x = q_broadcast @ agent_output

        # 重塑并应用输出投影和dropout
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.reshape(1, c, b, n)
        # v = v.permute(0, 2, 1, 3)
        # print(x.shape)
        # print(v.shape)
        v = self.dwc(v).reshape(b, n, c)

        x = x + v

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    # def resize_input(self, x):
    #     h, w = x.shape[-2:]
    #     new_h, new_w = self.window_size, self.window_size
    #
    #     # 将输入张量调整为固定的尺寸
    #     x = F.interpolate(x.permute(0, 2, 3, 1), size=(new_h, new_w), mode='bilinear').permute(0, 3, 1, 2)
    #     return x


class RCNNHead(nn.Module):

    def __init__(self, d_model, num_classes, pooler_resolution, dim_feedforward=2048, nhead=8, dropout=0.1,
                 activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0), use_focal=False,
                 use_fed_loss=False):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.self_agent_attention = AgentAttention(d_model, nhead)
        self.stf = SFT(d_model, pooler_resolution=pooler_resolution)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # cls.
        num_cls = 1
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # association score.
        num_score = 1
        score_module = list()
        for _ in range(num_score):
            score_module.append(nn.Linear(2 * d_model, d_model, False))
            score_module.append(nn.LayerNorm(d_model))
            score_module.append(nn.ReLU(inplace=True))
        self.score_module = nn.ModuleList(score_module)

        # reg.
        num_reg = 3
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, True))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.score_logits = nn.Linear(d_model, 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights
        nn.init.constant_(self.class_logits.bias, -math.log((1 - 1e-2) / 1e-2))
        nn.init.constant_(self.bboxes_delta.bias, -math.log((1 - 1e-2) / 1e-2))
        for sub_module in self.reg_module:
            if isinstance(sub_module, nn.Linear):
                nn.init.constant_(sub_module.bias, -math.log((1 - 1e-2) / 1e-2))

    def forward(self, features, bboxes, pro_features, pooler, time_emb, lost_features=None, fix_ref_boxes=False):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        if pro_features is not None:
            pro_features_x = pro_features
        else:
            pro_features_x = None

        bboxes_pre, bboxes_cur = bboxes

        N, nr_boxes = bboxes_pre.shape[:2]
        proposal_boxes_pre = list()
        proposal_boxes_curr = list()
        for b in range(N):
            proposal_boxes_pre.append(Boxes(bboxes_pre[b]))
            proposal_boxes_curr.append(Boxes(bboxes_cur[b]))

        roi_features_pre = pooler(features[0], proposal_boxes_pre)
        if lost_features is not None:
            roi_features_pre[roi_features_pre.shape[0] - lost_features.shape[0]:] = lost_features
        roi_features_curr = pooler(features[1], proposal_boxes_curr)

        if pro_features_x is None:
            pro_features_pre = roi_features_pre.view(N, nr_boxes, self.d_model, -1).mean(-1)
            pro_features_curr = roi_features_curr.view(N, nr_boxes, self.d_model, -1).mean(-1)
            pro_features_x = torch.cat([pro_features_pre, pro_features_curr], dim=0)
        roi_features_pre = roi_features_pre.view(N, nr_boxes, self.d_model, -1).permute(0, 1, 3, 2)
        roi_features_curr = roi_features_curr.view(N, nr_boxes, self.d_model, -1).permute(0, 1, 3, 2)
        roi_features_x = torch.cat([torch.cat([roi_features_pre, roi_features_curr], dim=-2).unsqueeze(2),
                                    torch.cat([roi_features_curr, roi_features_pre], dim=-2).unsqueeze(2)], dim=2)
        pro_features_x = pro_features_x.view(2 * N, nr_boxes, self.d_model)
        pro_features_x = pro_features_x + self.dropout1(
            self.self_attn(pro_features_x, pro_features_x, pro_features_x)[0])
        pro_features_x = self.norm1(pro_features_x)
        pro_features_x = torch.cat([x.unsqueeze(2) for x in pro_features_x.split(N, dim=0)], dim=-2)

        pro_features_x = pro_features_x + self.dropout2(self.stf(roi_features_x, pro_features_x))
        pro_features_x = self.norm2(pro_features_x)

        pro_features_x = torch.cat([x.squeeze(2) for x in pro_features_x.split(1, dim=-2)], dim=0).reshape(
            2 * N * nr_boxes, -1)

        # obj_feature.
        obj_features_tmp = self.linear2(self.dropout(self.activation(self.linear1(pro_features_x))))
        obj_features = pro_features_x + self.dropout3(obj_features_tmp)
        obj_features = self.norm3(obj_features)
        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = obj_features * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        score_feature = torch.cat(fc_feature.clone().split(N * nr_boxes, dim=0), dim=-1)

        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)

        for score_layer in self.score_module:
            score_feature = score_layer(score_feature)

        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)

        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)

        class_logits_pre, class_logits_curr = class_logits.split(N * nr_boxes, dim=0)
        bboxes_deltas_pre, bboxes_deltas_curr = bboxes_deltas.split(N * nr_boxes, dim=0)


        association_score = self.score_logits(score_feature)

        pred_bboxes_pre = self.apply_deltas(bboxes_deltas_pre, bboxes_pre.view(-1, 4))
        if fix_ref_boxes:
            assert not self.training, "fix reference bboxes only for inference mode"
            pred_bboxes_pre[:nr_boxes] = bboxes_pre[0, :nr_boxes]
        pred_bboxes_curr = self.apply_deltas(bboxes_deltas_curr, bboxes_cur.view(-1, 4))

        return (class_logits_pre.view(N, nr_boxes, -1), class_logits_curr.view(N, nr_boxes, -1)), (
        pred_bboxes_pre.view(N, nr_boxes, -1),
        pred_bboxes_curr.view(N, nr_boxes, -1)), obj_features, association_score.view(N, nr_boxes, -1)

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class SFT(nn.Module):

    def __init__(self, hidden_dim, pooler_resolution, dim_dynamic=2 * 64, num_dynamic=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.pooler_resolution = pooler_resolution
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        num_output = 2 * self.hidden_dim * self.pooler_resolution ** 2
        self.num_output = 2 * self.pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, roi_features, pro_features):
        '''
        pro_features: ( N,nr_boxes,2,self.d_model)
        roi_features: ( N,nr_boxes,2,49*2,self.d_model)
        '''
        N = pro_features.shape[0]
        # features=torch.cat([x.unsqueeze(2) for x in roi_features.split(self.num_output,dim=-2)],dim=2).reshape(-1,self.num_output,self.hidden_dim)
        features = roi_features.reshape(-1, self.num_output, self.hidden_dim)
        parameters = self.dynamic_layer(pro_features)

        param1 = parameters[:, :, :, :self.num_params].reshape(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, :, self.num_params:].reshape(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features.reshape(N, -1, 2, self.hidden_dim)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



