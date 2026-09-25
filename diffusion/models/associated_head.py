import torch.cuda
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
from .common import *
from icecream import ic
from yolox.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from diffusion.models.fast_reid_embedding import EmbeddingComputer
from scipy.optimize import linear_sum_assignment


def eval_acc(score, target, weight, th=0.5):
    """
    :param score: torch tensor, predicted score of shape [batch, H, W]
    :param target: torch tensor, ground truth value {0,1} of shape [batch, H, W]
    :param weight: torch tensor, weight for each batch for negative and positive examples of shape [batch, 2, 1, 1]
    :return: accuracy
    """
    acc = []
    predicted = torch.zeros_like(score).cuda().float()
    for b in range(score.size(0)):
        for h in range(score.size(1)):
            value, indice = score[b, h].max(0)
            if float(value) > th:
                predicted[b, h, int(indice)] = 1.0
        num_positive = float(target[b, :, :].sum())
        num_negative = float(target.size(1) * target.size(2) - num_positive)
        num_tp = float(((predicted[b, :, :] == target[b, :, :]).float() + (target[b, :, :] == 1.0).float()).eq(2).sum())
        num_tn = float(((predicted[b, :, :] == target[b, :, :]).float() + (target[b, :, :] == 0.0).float()).eq(2).sum())
        acc.append(1.0 * (num_tp * float(weight[b, 1, 0, 0]) + num_tn * float(weight[b, 0, 0, 0])) /
                   (num_positive * float(weight[b, 1, 0, 0]) + num_negative * float(weight[b, 0, 0, 0])))

    return predicted, np.mean(np.array(acc))


def weighted_binary_focal_entropy(output, target, weights=None, gamma=2):
    # output = torch.clamp(output, min=1e-8, max=1 - 1e-8)
    if weights is not None:
        assert weights.size(1) == 2

        # weight is of shape [batch,2, 1, 1]
        # weight[:,1] is for positive class, label = 1
        # weight[:,0] is for negative class, label = 0

        loss = (torch.pow(1.0 - output, gamma) * torch.mul(target, weights[:, 1]) * torch.log(output + 1e-8)) + \
               (torch.mul((1.0 - target), weights[:, 0]) * torch.log(1.0 - output + 1e-8) * torch.pow(output, gamma))
    else:
        loss = target * torch.log(output + 1e-8) + (1 - target) * torch.log(1 - output + 1e-8)

    return torch.neg(torch.mean(loss))


def hungarian_loss(similarity_matrix, target_matrix):
    """
    :param similarity_matrix: 模型输出的相似度矩阵 (batch_size, m, n)
    :param target_matrix: 真实关联矩阵 (batch_size, m, n)
    """
    batch_size, m, n = similarity_matrix.shape
    total_loss = 0.0

    for b in range(batch_size):
        # 使用匈牙利算法找到最优匹配
        row_indices, col_indices = linear_sum_assignment(
            -target_matrix[b].detach().cpu().numpy()  # 最大化相似度
        )

        # 提取匹配对的相似度
        matched_scores = similarity_matrix[b, row_indices, col_indices]
        # 计算匹配对的交叉熵损失
        matched_targets = target_matrix[b, row_indices, col_indices]
        diff_matrix = ((matched_scores > 0.5).int() != matched_targets)
        num_mismatch = diff_matrix.sum().item()
        ic(m)
        ic(num_mismatch)
        ic(matched_targets)
        ic(matched_scores)
        loss = nn.functional.binary_cross_entropy(matched_scores, matched_targets)
        total_loss += loss

    return total_loss / batch_size


class VarianceSchedule(Module):

    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
                    torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding
        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        # self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alpha_bars))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alpha_bars - 1))

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class HMINet(Module):

    def __init__(self, point_dim=300, context_dim=256, tf_layer=3, dropout=0.1, residual=False):
        super().__init__()
        self.timb = context_dim + 3
        self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=dropout, max_len=500)
        self.pos_emb2 = PositionalEncoding(d_model=context_dim, dropout=dropout, max_len=500)
        self.concat1 = MFL(context_dim, context_dim // 2, self.timb)
        self.concat1_2 = MFL(context_dim // 2, context_dim, self.timb)
        self.concat1_3 = MFL(context_dim, 2 * context_dim, self.timb)
        self.layer = nn.TransformerEncoderLayer(d_model=2 * context_dim, nhead=4, dim_feedforward=4 * context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.layer2 = nn.TransformerEncoderLayer(d_model=context_dim, nhead=4, dim_feedforward=2 * context_dim)
        self.transformer_encoder2 = nn.TransformerEncoder(self.layer2, num_layers=tf_layer)
        self.concat3 = MFL(2 * context_dim, context_dim, self.timb)
        self.concat4 = MFL(context_dim, context_dim // 2, self.timb)
        self.linear = MFL(context_dim // 2, context_dim, context_dim + 3)

    def forward(self, x, beta, context):
        x = x.unsqueeze(0)
        batch_size, dim1, dim2 = x.size(0), x.size(1), x.size(2)
        beta = beta.view(batch_size, x.size(1), 1)  # (B, 1)
        # context = context.view(batch_size, 1, context.size(2))  # (B, F)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, F+3)
        # x = x.permute(1, 0, 2)
        x = self.concat1_3(ctx_emb, self.concat1_2(ctx_emb, self.concat1(ctx_emb, x)))
        # final_emb = x.unsqueeze(0)
        final_emb = self.pos_emb(x)
        trans = self.transformer_encoder(final_emb).squeeze(1)
        trans = self.concat3(ctx_emb, trans)
        # final_emb = trans.unsqueeze(0)
        final_emb = self.pos_emb2(trans)
        trans = self.transformer_encoder2(final_emb).squeeze(1)
        trans = self.concat4(ctx_emb, trans)

        trans = self.linear(ctx_emb, trans)
        trans = torch.softmax(trans, dim=2)
        # trans = trans.permute(1, 0, 2)
        return trans


class D2MP_OB(Module):

    def __init__(self, net, var_sched: VarianceSchedule, config):
        super().__init__()
        self.config = config
        self.net = net
        self.var_sched = var_sched
        self.eps = self.config.eps
        self.weight = True

    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + torch.sqrt(time) * noise
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - torch.sqrt(time) * noise
        return x0

    def pred_C_from_xt(self, xt, noise, t):
        time = t.reshape(noise.shape[0], *((1,) * (len(noise.shape) - 1)))
        C = (xt - torch.sqrt(time) * noise) / (time - 1)
        return C

    def pred_xtms_from_xt(self, xt, noise, C, t, s):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt + C * (time - s) - C * time - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time - s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def loss_associated_focal(self, outputs, gt):
        num_gt = gt.size(0)
        num_bboxes = gt.size(1)
        # gt = gt[:, :num_bboxes].contiguous()
        # outputs = outputs.squeeze(0)
        # outputs = outputs[:, :num_bboxes].contiguous()
        acc = []
        test_p = []
        test_r = []
        output = outputs
        target = gt.reshape(-1, num_gt, num_bboxes).contiguous()
        output = output.reshape(-1, num_gt, num_bboxes).contiguous()
        num_positive = target.detach().clone().view(target.size(0), -1).sum(dim=1).unsqueeze(1)
        weight2negative = num_positive / (target.size(1) * target.size(2))
        # case all zeros, then weight2negative = 1.0
        weight2negative.masked_fill_((weight2negative == 0.0), 10)  # 10 is just a symbolic value representing 1.0
        # case all ones, then weight2negative = 0.0
        weight2negative.masked_fill_((weight2negative == 1.0), 0.0)
        # change all fake values 10 to their desired value 1.0
        weight2negative.masked_fill_((weight2negative == 10), 1.0)
        weight = torch.cat([weight2negative, 1.0 - weight2negative], dim=1)
        weight = weight.view(-1, 2, 1, 1).contiguous()
        loss = 50 * weighted_binary_focal_entropy(output, target, weights=weight)
        ic(loss)
        # loss_similar = focal_loss(similarity, label)
        predicted, curr_acc = eval_acc(output.float().detach(), target.float().detach(), weight.detach())
        acc.append(curr_acc)

        # calculate J value
        tp = torch.sum((predicted == target.float().detach())[target.data == 1.0]).double()
        fp = torch.sum((predicted != target.float().detach())[predicted.data == 1.0]).double()
        fn = torch.sum((predicted != target.float().detach())[predicted.data == 0.0]).double()

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        test_p.append(p.item())
        test_r.append(r.item())

        print('P {:.2f}% \t R {:.2f}% \t weighted Accuracy {:.2f} %'.format(100.0 * np.mean(np.array(test_p)),
                                                                            100.0 * np.mean(np.array(test_r)),
                                                                            100.0 * np.mean(np.array(acc))))
        # # losses['loss_similar'] = (loss_similar / num_boxes) * 10
        return loss

    def forward(self, x_0, context, indices, t=None):
        batch_size, point_dim = x_0.size()
        # x_0[numgt, num_proposals]
        if t == None:
            t = torch.rand(x_0.shape[0], device=x_0.device) * (1. - self.eps) + self.eps
        beta = t.log() / 4
        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)
        C = -1 * x_0
        x_noisy = self.q_sample(x_start=x_0, noise=e_rand, t=t, C=C)
        t = t.reshape(-1, 1)
        pred = self.net(x_noisy, beta=beta, context=context)
        C_pred = pred
        noise_pred = (x_noisy - (t - 1) * C_pred) / t.sqrt()
        if not self.weight:
            loss_C = self.loss_associated_focal(C_pred, x_0)
            loss_encoder = self.loss_associated_focal(context, x_0)
            # loss_x0 = F.smooth_l1_loss(x_rec.view(-1, point_dim), x_0.view(-1, point_dim), reduction='mean')
            # loss_noise = 50 * F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
            loss = (loss_C + loss_encoder).sum()
        else:
            # simple_weight1 = (t ** 2 - t + 1) / t
            # simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps)

            # simple_weight1 = (t + 1) / t
            # simple_weight2 = (2 - t) / (1 - t + self.eps)
            loss_C = self.loss_associated_focal(C_pred, x_0)
            loss_encoder = self.loss_associated_focal(context, x_0)
            # loss_x0 = F.smooth_l1_loss(x_rec.view(-1, point_dim), x_0.view(-1, point_dim), reduction='none')
            # loss_noise = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='none')
            # loss = simple_weight1 * loss_C + simple_weight2 * loss_noise
            loss = (loss_C + loss_encoder).sum()
            # loss = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')

        return loss

    def sample(self, context, sample, bestof, point_dim=4, flexibility=0.0, ret_traj=False):
        traj_list = []
        # context = context.to(self.var_sched.betas.device)
        # context.shape [batch,num_gt, self.num_proposal]
        for i in range(sample):
            batch_size = context.size(1)
            if bestof:
                x_T = torch.randn([batch_size, point_dim]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, point_dim]).to(context.device)

            self.var_sched.num_steps = 1
            traj = {self.var_sched.num_steps: x_T}

            cur_time = torch.ones((batch_size,), device=x_T.device)
            step = 1. / self.var_sched.num_steps
            for t in range(self.var_sched.num_steps, 0, -1):
                s = torch.full((batch_size,), step, device=x_T.device)
                if t == 1:
                    s = cur_time

                x_t = traj[t]
                beta = cur_time.log() / 4
                t_tmp = cur_time.reshape(-1, 1)
                pred = self.net(x_t, beta=beta, context=context)
                C_pred = pred.squeeze(0)
                noise_pred = (x_t - (t_tmp - 1) * C_pred) / t_tmp.sqrt()
                x0 = self.pred_x0_from_xt(x_t, noise_pred, C_pred, cur_time)
                x0.clamp_(-1., 1.)
                C_pred = -1 * x0
                x_next = self.pred_xtms_from_xt(x_t, noise_pred, C_pred, cur_time, s)
                cur_time = cur_time - s
                traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
                if not ret_traj:
                    del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])

        return torch.stack(traj_list)


class AssociatedHead(Module):
    def __init__(self, config, device="cuda:0"):
        super().__init__()
        self.dtype = None
        self.config = config
        self.device = device
        self.num_proposals = config.num_proposals
        self.reid_head = EmbeddingComputer(dataset=config.datasets, test_dataset=True, grid_off=True)
        self.asso_embedding = ATTWeightHead(2048, config.tf_layer, 0.1)
        self.pos_embed = nn.Linear(4, 2048)
        self.cross_attn_prev = FlexibleMultiheadAttention(2048, 8)
        self.cross_attn_curr = FlexibleMultiheadAttention(2048, 8)
        # self.cross_attn_prev = nn.MultiheadAttention(2048, 8)
        # self.cross_attn_curr = nn.MultiheadAttention(2048, 8)

        self.diffusion = D2MP_OB(
            # net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            net=HMINet(point_dim=4, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched=VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'
            ),
            config=self.config
        )

    def asso_generate(self, targets=None, imgs=None, reid_feature_pre=None, reid_feature_curr=None):
        pre_gt_boxes, cur_gt_boxes = targets
        pre_gt_boxes = torch.tensor(pre_gt_boxes).detach().to(self.device)
        cur_gt_boxes = torch.tensor(cur_gt_boxes).detach().to(self.device)

        gt_pre = pre_gt_boxes.size(0)
        gt_cur = cur_gt_boxes.size(0)

        pre_images, cur_images = imgs
        if reid_feature_pre.all():
            reid_feature_pre = torch.tensor(reid_feature_pre).detach().to(self.device)
            reid_feature_curr = reid_feature_curr.clone().detach().to(self.device)
        else:
            reid_feature_pre = self.reid_head.compute_embedding(pre_images, pre_gt_boxes[:, :4], tag="pre")
            reid_feature_curr = self.reid_head.compute_embedding(cur_images, cur_gt_boxes[:, :4], tag="curr")

        pos_prev = self._get_pos_encoding(pre_gt_boxes[:, :4].float())  # [N, D]
        pos_curr = self._get_pos_encoding(cur_gt_boxes[:, :4].float())

        attn_prev = self.cross_attn_prev(
            query=(reid_feature_pre + pos_prev).unsqueeze(0).float(),
            key=reid_feature_pre.unsqueeze(0).float(),
            value=reid_feature_pre.unsqueeze(0).float()
        )
        attn_curr = self.cross_attn_curr(
            query=(reid_feature_curr + pos_curr).unsqueeze(0).float(),
            key=reid_feature_curr.unsqueeze(0).float(),
            value=reid_feature_curr.unsqueeze(0).float()
        )
        # ic(attn_curr)
        reid_feature_pre = attn_prev[0]
        reid_feature_curr = attn_curr[0]

        pad_rows = self.num_proposals - reid_feature_curr.size(0)
        reid_feature_curr_pad = F.pad(reid_feature_curr, (0, 0, 0, pad_rows)).contiguous()
        ass_score_encoder = self.asso_embedding(reid_feature_pre.unsqueeze(0),
                                                reid_feature_curr_pad.unsqueeze(0))
        ass_matrix = self.diffusion.sample(ass_score_encoder, sample=1, bestof=False, point_dim=256).squeeze(0)
        ass_matrix = ass_matrix[:gt_pre, :gt_cur].cpu().numpy()
        return ass_matrix

    def get_reid_emb(self, imgs, bboxes, tag):
        return self.reid_head.compute_embedding(imgs, bboxes, tag=tag).cpu() if len(bboxes) else None

    def generate(self, conds, sample, bestof, flexibility=0.0, ret_traj=False, img_w=None, img_h=None):
        cond_encodeds = []
        for i in range(len(conds)):
            tmp_c = conds[i]
            tmp_c = np.array(tmp_c)
            tmp_c[:, 0::2] = tmp_c[:, 0::2] / img_w
            tmp_c[:, 1::2] = tmp_c[:, 1::2] / img_h
            tmp_conds = torch.tensor(tmp_c, dtype=torch.float)
            if len(tmp_conds) != 5:
                pad_conds = tmp_conds[-1].repeat((5, 1))
                tmp_conds = torch.cat((tmp_conds, pad_conds), dim=0)[:5]
            cond_encodeds.append(tmp_conds.unsqueeze(0))
        cond_encodeds = torch.cat(cond_encodeds)
        cond_encodeds = self.encoder(cond_encodeds)
        track_pred = self.diffusion.sample(cond_encodeds, sample, bestof, flexibility=flexibility, ret_traj=ret_traj)
        return track_pred.cpu().detach().numpy()

    def generate_association_matrix(self, x, y, tmp=100):
        # 计算相似性矩阵（欧氏距离）
        similarity_matrix = torch.mm(x, y.T) / (torch.norm(x, dim=1).unsqueeze(1) * torch.norm(y, dim=1).unsqueeze(0))
        distance = 1 - similarity_matrix
        row, col = linear_sum_assignment(distance.cpu().numpy())
        ic(similarity_matrix)
        # ic(col)

        # distances_matrix = torch.cdist(x, y)
        # sim = 1 - distances_matrix
        # row, col = linear_sum_assignment(distances_matrix.cpu().numpy())
        # ic(col)

        # ftrk = x
        # fdet = y
        # aff = torch.mm(ftrk, fdet.transpose(0, 1))
        # aff_td = F.softmax(tmp * aff, dim=1)
        # aff_dt = F.softmax(tmp * aff, dim=0).transpose(0, 1)
        # res_recons_ftrk = torch.mm(aff_td, fdet)
        # res_recons_fdet = torch.mm(aff_dt, ftrk)
        # sim = (torch.mm(ftrk, fdet.transpose(0, 1)) + torch.mm(res_recons_ftrk, res_recons_fdet.transpose(0, 1))) / 2
        # distances = 1 - sim
        # ic(sim)
        # row, col = linear_sum_assignment(distances.detach().cpu().numpy())
        # ic(col)
        return similarity_matrix

    def prepare_targets(self, targets):
        # targets.shape [2, 1000, 5]
        labels = targets[..., :6]
        # labels.shape [2, 1000, 5]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        # nlabel : tensor([147, 147], device='cuda:0')
        # 筛选出有效的框
        new_targets = []
        for batch_idx, num_gt in enumerate(nlabel):
            target = {}
            gt_bboxes_per_image = box_cxcywh_to_xyxy(labels[batch_idx, :num_gt, 1:5])
            target['id'] = labels[batch_idx, :num_gt, 5]
            target["boxes_xyxy"] = gt_bboxes_per_image
            new_targets.append(target)

        return new_targets

    def get_scores(self, shuffled_indices, num_gt):
        num_gt = num_gt
        num_targets = self.num_proposals
        association_matrix = np.zeros((num_gt, num_targets), dtype=np.float64)
        for i in range(num_gt):
            association_matrix[i, i] = 1

        association_matrix = association_matrix[:, shuffled_indices]
        association_matrix = torch.from_numpy(association_matrix)

        # # 创建新张量的值到索引的映射
        # value_to_index = {value.item(): idx for idx, value in enumerate(shuffled_indices)}
        # # 原索引范围
        # original_indices = list(range(len(shuffled_indices)))
        # # 获取每个原索引在新张量中的位置
        # mapped_indices = [value_to_index[i] for i in original_indices]
        # indices = torch.tensor(mapped_indices)
        # # ic(indices)
        # if num_gt != 0:
        #     association_matrix[torch.arange(num_gt), indices] = 1
        return association_matrix

    def _get_pos_encoding(self, boxes):
        """生成位置编码"""
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        wh = boxes[:, 2:] - boxes[:, :2]
        return self.pos_embed(torch.cat([centers, wh], dim=1))

    def forward(self, mate_info, targets=None, imgs=None):
        mate_shape, mate_device, mate_dtype = mate_info
        self.device = mate_device
        self.dtype = mate_dtype
        b, _, h, w = mate_shape
        # torch.set_printoptions(threshold=torch.inf)
        if self.training:
            targets = self.prepare_targets(targets)
            pre_gt_boxes = targets[0]['boxes_xyxy']
            cur_gt_boxes = targets[1]['boxes_xyxy']
            gt_pre = pre_gt_boxes.size(0)
            pre_images, cur_images = imgs
            if gt_pre != 0:
                reid_feature_pre = self.reid_head.compute_embedding(pre_images, pre_gt_boxes[:, :4], tag="pre")
                reid_feature_curr = self.reid_head.compute_embedding(cur_images, cur_gt_boxes[:, :4], tag="curr")
                pos_prev = self._get_pos_encoding(pre_gt_boxes[:, :4].float())  # [N, D]
                pos_curr = self._get_pos_encoding(cur_gt_boxes[:, :4].float())

                attn_prev = self.cross_attn_prev(
                    query=(reid_feature_pre + pos_prev).unsqueeze(0).float(),
                    key=reid_feature_pre.unsqueeze(0).float(),
                    value=reid_feature_pre.unsqueeze(0).float()
                )
                attn_curr = self.cross_attn_curr(
                    query=(reid_feature_curr + pos_curr).unsqueeze(0).float(),
                    key=reid_feature_curr.unsqueeze(0).float(),
                    value=reid_feature_curr.unsqueeze(0).float()
                )
                # ic(attn_curr)
                reid_feature_pre = attn_prev[0]
                reid_feature_curr = attn_curr[0]

                pad_rows = self.num_proposals - reid_feature_curr.size(0)
                reid_feature_curr_pad = F.pad(reid_feature_curr, (0, 0, 0, pad_rows)).contiguous()
                shuffled_indices = torch.randperm(reid_feature_curr_pad.shape[0])
                reid_feature_curr_pad = reid_feature_curr_pad[shuffled_indices]
                asso_gt = self.get_scores(shuffled_indices, gt_pre).to(self.device)

                # ass_encoder = self.generate_association_matrix(reid_feature_pre.to(self.dtype),
                #                                                      reid_feature_curr.to(self.dtype))

                ass_score_encoder = self.asso_embedding(reid_feature_pre.unsqueeze(0),
                                                        reid_feature_curr_pad.unsqueeze(0))
                loss = self.diffusion(asso_gt.to(self.dtype), ass_score_encoder, shuffled_indices)
                return loss
            else:
                reid_feature_pre = None
                reid_feature_curr = None
                return 0


class MLP(nn.Module):
    """Multi-layer perceptron with residual connections where applicable."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = []  # 标记各层是否需要残差连接

        if self.num_layers > 0:
            # 初始化各层维度
            h_dims = [hidden_dim] * (num_layers - 1)
            layer_dims = [input_dim] + h_dims
            next_dims = h_dims + [output_dim]

            # 创建层并记录残差标记
            self.layers = nn.ModuleList()
            for in_dim, out_dim in zip(layer_dims, next_dims):
                self.layers.append(nn.Linear(in_dim, out_dim))
                self.residual.append(in_dim == out_dim)  # 仅当输入输出维度相同时启用残差

            # 初始化Dropout层
            if self.dropout > 0 and num_layers > 1:
                self.dropouts = nn.ModuleList(
                    [nn.Dropout(dropout) for _ in range(num_layers - 1)])
        else:
            self.layers = nn.ModuleList()

    def forward(self, x):
        x = x.float()
        for i, layer in enumerate(self.layers):
            # 中间层处理（带激活函数和可能的残差）
            if i < self.num_layers - 1:
                identity = x  # 保存残差项
                x = layer(x)
                # 应用残差连接（仅在维度匹配时）
                if self.residual[i]:
                    x += identity

                x = F.relu(x)

                # 应用Dropout
                if hasattr(self, 'dropouts') and i < len(self.dropouts):
                    x = self.dropouts[i](x)

            # 最后一层处理（无激活函数）
            else:
                x = layer(x)

        return x


class ATTWeightHead(nn.Module):
    def __init__(self, feature_dim, num_layers, dropout):
        super().__init__()
        self.weight_out_dim = feature_dim
        self.q_proj = MLP(
            feature_dim, self.weight_out_dim, self.weight_out_dim,
            num_layers, dropout)
        self.k_proj = MLP(
            feature_dim, feature_dim, self.weight_out_dim,
            num_layers, dropout)

    def forward(self, query, key, temp_embs=None):
        '''
        Inputs:
          query: B x M x F
          key: B x N x F
          temp_embs: B x N x F
        '''
        # 确保输入是 float32
        query = query.float()
        key = key.float()

        k = self.k_proj(key)  # B x N x D
        q = self.q_proj(query)  # B x M x D
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # B x M x N
        return torch.softmax((attn_weights / math.sqrt(q.size(-1))), dim=2)


class FlexibleMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # 可分离的Q/K/V投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        query: (Lq, N, E)
        key: (Lk, N, E)
        value: (Lv, N, E)
        key_padding_mask: (N, Lk)
        返回: (Lq, N, E)
        """
        # 投影到Q/K/V空间
        q = self.q_proj(query)  # (Lq, N, E)
        k = self.k_proj(key)  # (Lk, N, E)
        v = self.v_proj(value)  # (Lv, N, E)

        # 多头拆分
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (N, H, Lq, D)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (N, H, Lk, D)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (N, H, Lv, D)
        # 注意力计算（允许Lq != Lk）
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (N, H, Lq, Lk)
        # 处理padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (N, 1, 1, Lk)
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)

        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)  # (N, H, Lq, D)

        # 合并多头
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()  # (Lq, N, H, D)
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), self.embed_dim)  # (Lq, N, E)

        return self.out_proj(attn_output)