import numpy as np
from collections import deque
import time
import torch
import torch.nn.functional as F
import torchvision
from copy import deepcopy
from yolox.tracker import matching
from detectron2.structures import Boxes
from yolox.utils.box_ops import box_xyxy_to_cxcywh
from yolox.utils.boxes import xyxy2cxcywh
from torchvision.ops import box_iou, nms
from yolox.utils.cluster_nms import cluster_nms

from .kalman_filter import KalmanFilter
from yolox.tracker import matching, match_es
from .basetrack import BaseTrack, TrackState
from scipy.optimize import linear_sum_assignment
from icecream import ic


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, emb=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.emb = emb

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self._tlwh = new_track.tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    def update_emb(self, emb, alpha=0.9):
        emb /= np.linalg.norm(emb)
        if self.emb is None:
            self.emb = emb
        else:
            self.emb = alpha * self.emb + (1 - alpha) * emb
            self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class DiffusionTracker(object):
    def __init__(self, model, tensor_type, conf_thresh=0.7, det_thresh=0.6, nms_thresh_3d=0.7, nms_thresh_2d=0.75,
                 interval=5, detections=None):

        self.frame_id = 0
        # BaseTrack._count=-1
        self.backbone = model.backbone
        self.feature_projs = model.projs
        self.diffusion_model = model.head
        self.asso_model = model.assohead
        self.feature_extractor = self.diffusion_model.head.box_pooler
        self.det_thresh = det_thresh
        self.association_thresh = conf_thresh
        # self.low_det_thresh = 0.1
        # self.low_association_thresh = 0.2
        self.nms_thresh_2d = nms_thresh_2d
        self.nms_thresh_3d = nms_thresh_3d
        self.same_thresh = 0.9
        self.pre_features = None
        self.data_type = tensor_type
        self.detections = detections

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.max_time_lost = 30
        self.kalman_filter = KalmanFilter()

        self.repeat_times = 1
        self.dynamic_time = True

        self.sampling_steps = 1
        self.num_boxes = 500

        self.track_t = 400
        self.mot17 = False

        self.pre_imgs = None

    def update(self, cur_image):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        cur_features, mate_info = self.extract_feature(cur_image=cur_image)
        mate_shape, mate_device, mate_dtype = mate_info
        self.diffusion_model.device = mate_device
        self.diffusion_model.dtype = mate_dtype
        b, _, h, w = mate_shape
        images_whwh = torch.tensor([w, h, w, h], dtype=mate_dtype, device=mate_device)[None, :].expand(4 * b, 4)
        if self.frame_id == 1:
            if self.pre_features is None:
                self.pre_features = cur_features
            inps = self.prepare_input(self.pre_features, cur_features)
            diffusion_outputs, conf_scores, association_time = self.diffusion_model.new_ddim_sample(inps, images_whwh,
                                                                                                    num_timesteps=self.sampling_steps,
                                                                                                    num_proposals=self.num_boxes,
                                                                                                    dynamic_time=self.dynamic_time,
                                                                                                    track_candidate=self.repeat_times)
            _, _, detections = self.diffusion_postprocess(diffusion_outputs, conf_scores,
                                                          conf_thre=self.association_thresh,
                                                          nms_thre=self.nms_thresh_3d)
            detections = self.diffusion_det_filt(detections, conf_thre=self.det_thresh, nms_thre=self.nms_thresh_2d)
            if detections.shape[0] > 0:
                detections_emb = self.asso_model.get_reid_emb(cur_image, detections[:, :4], "cur").cpu()
            else:
                detections_emb = torch.empty((0, 2048))

            for det, emb in zip(detections, detections_emb):
                track = STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                track.activate(self.kalman_filter, self.frame_id)
                track.update_emb(emb)
                self.tracked_stracks.append(track)
            output_stracks = [track for track in self.tracked_stracks if track.is_activated]
            self.pre_imgs = cur_image
            return output_stracks, association_time
        else:
            imgs = (self.pre_imgs, cur_image)
            ref_bboxes = [STrack.tlwh_to_tlbr(track._tlwh) for track in self.tracked_stracks]
            inps = self.prepare_input(self.pre_features, cur_features)
            if len(ref_bboxes) > 0:
                bboxes = box_xyxy_to_cxcywh(torch.tensor(np.array(ref_bboxes))).type(self.data_type).reshape(1, -1,
                                                                                                             4).repeat(
                    2, 1, 1)
            else:
                bboxes = None
            # ref_num_proposals=self.proposal_schedule(len(ref_bboxes))
            # ref_sampling_steps=self.sampling_steps_schedule(len(ref_bboxes))
            diffusion_outputs, conf_scores, association_time = self.diffusion_model.new_ddim_sample(inps, images_whwh,
                                                                                                    num_timesteps=self.sampling_steps,
                                                                                                    num_proposals=self.num_boxes,
                                                                                                    ref_targets=bboxes,
                                                                                                    dynamic_time=self.dynamic_time,
                                                                                                    track_candidate=self.repeat_times,
                                                                                                    diffusion_t=self.track_t)

            diffusion_ref_detections, diffusion_track_detections, detections = self.diffusion_postprocess(
                diffusion_outputs,
                conf_scores,
                conf_thre=self.association_thresh,
                nms_thre=self.nms_thresh_3d)

            detections = self.diffusion_det_filt(detections, conf_thre=0.1, nms_thre=self.nms_thresh_2d)

            scores = detections[:, 5]
            bboxes = detections[:, :4]
            remain_inds = scores > 0.6
            inds_low = scores > 0.1
            inds_high = scores < 0.6

            inds_second = np.logical_and(inds_low, inds_high)

            dets_second = bboxes[inds_second]
            dets = bboxes[remain_inds]
            if len(dets) > 0:
                dets_embs = self.asso_model.get_reid_emb(cur_image, dets, "cur").cpu()
            else:
                dets_embs = torch.empty((0, 2048))  # 根据你的 embedding 维度定义
            scores_keep = scores[remain_inds]
            scores_second = scores[inds_second]

            trust = (scores_keep - 0.6) / (1 - 0.6)
            af = 0.8
            dets_alpha = af + (1 - af) * (1 - trust)

            if len(dets) > 0:
                '''Detections'''
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
            else:
                detections = []

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed = []
            tracked_stracks = []  # type: list[STrack]
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)

            ''' Step 2: First association, with high score detection boxes'''
            # Predict the current location with KF
            strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
            trk_embs = [track.get_emb() for track in strack_pool]
            trk_embs = np.array(trk_embs)
            start_time = time.time()
            STrack.multi_predict(strack_pool)
            ref_bboxes = [STrack.tlwh_to_tlbr(track._tlwh) for track in strack_pool]
            dists = matching.iou_distance(strack_pool, detections)
            targets = (ref_bboxes, dets[:, :4])
            matches, u_track, u_detection = self.associate(
                strack_pool,
                detections,
                trk_embs,
                dets_embs,
                targets,
                imgs,
                0.3)
            # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.same_thresh)

            ref_box_t = []
            track_box_t = []
            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                emb = dets_embs[idet]
                alpha = dets_alpha[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    track.update_emb(emb, alpha)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            if len(ref_box_t) > 0:
                self.track_t = self.extract_mean_track_t(np.array(ref_box_t), np.array(track_box_t))
            ''' Step 3: Second association, with low score detection boxes'''
            # association the untrack to the low score detections
            if len(dets_second) > 0:
                '''Detections'''
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                     (tlbr, s) in zip(dets_second, scores_second)]
            else:
                detections_second = []
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dets_embs = [dets_embs[i] for i in u_detection]

            dists = matching.iou_distance(unconfirmed, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                emb = dets_embs[idet]
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                unconfirmed[itracked].update_emb(emb)
                activated_starcks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

            """ Step 4: Init new stracks"""
            for inew in u_detection:
                track = detections[inew]
                emb = dets_embs[inew]
                if track.score < self.det_thresh:
                    continue
                track.update_emb(emb)
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)
            """ Step 5: Update state"""
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)
            self.pre_imgs = cur_image
            self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
            self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
            self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
            self.lost_stracks.extend(lost_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
            self.removed_stracks.extend(removed_stracks)
            self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
            # get scores of lost tracks
        self.pre_features = cur_features
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks, association_time + time.time() - start_time

    def extract_feature(self, cur_image):
        fpn_outs = self.backbone(cur_image)
        cur_features = []
        for proj, l_feat in zip(self.feature_projs, fpn_outs):
            cur_features.append(proj(l_feat))
        mate_info = (cur_image.shape, cur_image.device, cur_image.dtype)
        return cur_features, mate_info

    def extract_mean_track_t(self, pre_box, cur_box):
        # "xyxy"
        pre_box = xyxy2cxcywh(pre_box)
        cur_box = xyxy2cxcywh(cur_box)
        abs_box = np.abs(pre_box - cur_box)
        abs_percent = np.sum(abs_box / (pre_box + 1e-5), axis=1) / 4
        track_t = np.mean(abs_percent)
        # min(max(int(track_t*1000),1),999)
        # min(max(int((np.exp(track_t)-1)/(np.exp(0)-1)*1000),1),999)
        # min(max(int(np.log(track_t+1)/np.log(2)*1000),1),999)
        return min(max(int(track_t * 1000), 1), 999)

    def diffusion_postprocess(self, diffusion_outputs, conf_scores, nms_thre=0.7, conf_thre=0.6):
        """ diffusion_outputs is a [n , 5] tensor, and [:, :4] is box[x, y, x, y], [:, 5] is box confidence
            conf_score is a [n , 1] tensor, which represents pre's and cur's relationship
        """
        pre_prediction, cur_prediction = diffusion_outputs.split(len(diffusion_outputs) // 2, dim=0)

        output = [None for _ in range(len(pre_prediction))]
        # computer every batch
        for i, (pre_image_pred, cur_image_pred, association_score) in enumerate(
                zip(pre_prediction, cur_prediction, conf_scores)):
            # This operation is transform [n , 1] to [n * 1]
            association_score = association_score.flatten()
            # If none are remaining => process next image
            if not pre_image_pred.size(0):
                continue
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.zeros((2, len(cur_image_pred), 7), dtype=cur_image_pred.dtype,
                                     device=cur_image_pred.device)
            # box[x ,y, x, y]
            detections[0, :, :4] = pre_image_pred[:, :4]
            detections[1, :, :4] = cur_image_pred[:, :4]
            # confidence
            detections[0, :, 4] = association_score
            detections[1, :, 4] = association_score
            # asso confidence, [n, 1] * [n]，and obtain a [n , 1] tensor
            detections[0, :, 5] = torch.sqrt(torch.sigmoid(pre_image_pred[:, 4]) * association_score)
            detections[1, :, 5] = torch.sqrt(torch.sigmoid(cur_image_pred[:, 4]) * association_score)

            score_out_index = association_score > conf_thre

            # strategy=torch.mean
            # value=strategy(detections[:,:,5],dim=0,keepdim=False)
            # score_out_index=value>conf_thre

            detections = detections[:, score_out_index, :]

            if not detections.size(1):
                output[i] = detections
                continue

            nms_out_index_3d = cluster_nms(
                detections[0, :, :4],
                detections[1, :, :4],
                # value[score_out_index],
                detections[0, :, 4],
                iou_threshold=nms_thre)

            detections = detections[:, nms_out_index_3d, :]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output[0][0], output[0][1], torch.cat([output[1][0], output[1][1]], dim=0) if len(output) >= 2 else None

    def diffusion_track_filt(self, ref_detections, track_detections, conf_thre=0.6, nms_thre=0.7):

        if not ref_detections.size(1):
            return ref_detections.cpu().numpy(), track_detections.cpu().numpy()

        scores = ref_detections[:, 5]
        score_out_index = scores > conf_thre
        ref_detections = ref_detections[score_out_index]
        track_detections = track_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
            ref_detections[:, :4],
            ref_detections[:, 5],
            ref_detections[:, 6],
            nms_thre,
        )
        return ref_detections[nms_out_index].cpu().numpy(), track_detections[nms_out_index].cpu().numpy()

    def diffusion_det_filt(self, diffusion_detections, conf_thre=0.6, nms_thre=0.7):

        if not diffusion_detections.size(1):
            return diffusion_detections.cpu().numpy()

        scores = diffusion_detections[:, 5]
        score_out_index = scores > conf_thre
        diffusion_detections = diffusion_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
            diffusion_detections[:, :4],
            diffusion_detections[:, 5],
            diffusion_detections[:, 6],
            nms_thre,
        )
        return diffusion_detections[nms_out_index].cpu().numpy()

    def proposal_schedule(self, num_ref_bboxes):
        # simple strategy
        return 16 * num_ref_bboxes

    def sampling_steps_schedule(self, num_ref_bboxes):
        min_sampling_steps = 1
        max_sampling_steps = 4
        min_num_bboxes = 10
        max_num_bboxes = 100
        ref_sampling_steps = (num_ref_bboxes - min_num_bboxes) * (max_sampling_steps - min_sampling_steps) / (
                    max_num_bboxes - min_num_bboxes) + min_sampling_steps

        return min(max(int(ref_sampling_steps), min_sampling_steps), max_sampling_steps)

    def vote_to_remove_candidate(self, track_ids, detections, vote_iou_thres=0.75, sorted=False, descending=False):

        box_pred_per_image, scores_per_image = detections[:, :4], detections[:, 4] * detections[:, 5]
        score_track_indices = torch.argsort((track_ids + scores_per_image), descending=True)
        track_ids = track_ids[score_track_indices]
        scores_per_image = scores_per_image[score_track_indices]
        box_pred_per_image = box_pred_per_image[score_track_indices]

        assert len(track_ids) == box_pred_per_image.shape[0]

        # vote guarantee only one track id in track candidates
        keep_mask = torch.zeros_like(scores_per_image, dtype=torch.bool)
        for class_id in torch.unique(track_ids):
            curr_indices = torch.where(track_ids == class_id)[0]
            curr_keep_indices = nms(box_pred_per_image[curr_indices], scores_per_image[curr_indices], vote_iou_thres)
            candidate_iou_indices = box_iou(box_pred_per_image[curr_indices],
                                            box_pred_per_image[curr_indices]) > vote_iou_thres
            counter = []
            for cluster_indice in candidate_iou_indices[curr_keep_indices]:
                cluster_scores = scores_per_image[curr_indices][cluster_indice]
                counter.append(len(cluster_scores) + torch.mean(cluster_scores))
            max_indice = torch.argmax(torch.tensor(counter).type(self.data_type))
            keep_mask[curr_indices[curr_keep_indices][max_indice]] = True

        keep_indices = torch.where(keep_mask)[0]
        track_ids = track_ids[keep_indices]
        box_pred_per_image = box_pred_per_image[keep_indices]
        scores_per_image = scores_per_image[keep_indices]

        if sorted and not descending:
            descending_indices = torch.argsort(track_ids)
            track_ids = track_ids[descending_indices]
            box_pred_per_image = box_pred_per_image[descending_indices]
            scores_per_image = scores_per_image[descending_indices]

        return track_ids.cpu().numpy(), box_pred_per_image.cpu().numpy(), scores_per_image.cpu().numpy()

    def prepare_input(self, pre_features, cur_features):
        inps_pre_features = []
        inps_cur_Features = []
        for l_pre_feat, l_cur_feat in zip(pre_features, cur_features):
            inps_pre_features.append(torch.cat([l_pre_feat.clone(), l_cur_feat.clone()], dim=0))
            inps_cur_Features.append(torch.cat([l_cur_feat.clone(), l_cur_feat.clone()], dim=0))
        return (inps_pre_features, inps_cur_Features)

    def associate(
            self,
            trackers,
            detections,
            trk_embs,
            det_embs,
            targets,
            imgs,
            iou_threshold=0.3,
            w_assoc_emb=0.5,
            aw_off=True,
            aw_param=0.5,
            return_cost=False  # optional flag for debugging
    ):
        if len(trackers) == 0 or len(detections) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(trackers)),
                np.arange(len(detections)),
            )

        # === 1. Compute IOU matrix ===
        iou_dists = matching.iou_distance(trackers, detections)
        iou_matrix = 1 - iou_dists  # higher is better

        # === 2. Compute ReID embedding similarity (cosine similarity) ===
        emb_cost = -self.asso_model.asso_generate(targets, imgs, trk_embs, det_embs)
        # === 3. One-to-one valid matching check (greedy bipartite) ===
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # === 4. Weight control for embedding similarity ===
                if emb_cost is None:
                    emb_cost = np.zeros_like(iou_matrix)
                else:
                    # Mask ReID similarity for poor IOU
                    emb_cost[iou_matrix <= iou_threshold] = -np.inf

                # Adaptive weighting or fixed scalar
                if not aw_off:
                    w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
                    emb_cost = emb_cost * w_matrix
                else:
                    emb_cost = emb_cost * w_assoc_emb

                # Final cost matrix: fusion of IOU and embedding similarity
                fusion_score = (1 - w_assoc_emb) * iou_matrix + emb_cost  # higher is better
                fusion_score = np.nan_to_num(fusion_score, nan=0.0, posinf=1.0, neginf=0.0)

                final_cost = 1 - fusion_score  # convert to cost (lower is better)
                matched_indices = desort_linear_assignment(final_cost)
        else:
            matched_indices = np.empty(shape=(0, 2), dtype=int)
            final_cost = np.zeros((len(trackers), len(detections)))  # dummy

        # === 5. Get unmatched trackers and detections ===
        unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 1]]
        unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 0]]

        # === 6. Filter out low IOU matches ===
        matches = []
        for m in matched_indices:
            t_idx, d_idx = m
            if iou_matrix[t_idx, d_idx] < iou_threshold:
                unmatched_trackers.append(t_idx)
                unmatched_detections.append(d_idx)
            else:
                matches.append([t_idx, d_idx])

        matches = np.array(matches, dtype=int) if len(matches) > 0 else np.empty((0, 2), dtype=int)
        unmatched_trackers = np.array(unmatched_trackers, dtype=int)
        unmatched_detections = np.array(unmatched_detections, dtype=int)

        if return_cost:
            return matches, unmatched_trackers, unmatched_detections, final_cost
        else:
            return matches, unmatched_trackers, unmatched_detections


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


from sklearn.metrics.pairwise import cosine_similarity


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    # if len(stracksa)>0 and len(stracksb)>0:
    #     # fix a derection bug
    #     pcosdist=cosine_similarity(
    #         [track.mean[4:6] for track in stracksa],
    #         [track.mean[4:6] for track in stracksb])
    #     pdist=(pdist+pcosdist)/2

    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if stracksa[p].mean is not None and stracksb[q].mean is not None:
            x, y = stracksa[p].mean[4:6], stracksa[p].mean[4:6]
            cosine_dist = 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-06)
            if cosine_dist > 0.15:
                continue
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def min_cost_matching(adap_flag=True, max_distance=0.45, tracks=None, detections=None, cost_matrix=None,
                      track_indices=None, detection_indices=None):
    # Far sure
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # Nothing to match.
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices
    # Adaptively set threshold
    if adap_flag:
        cost_matrix_min = np.min(cost_matrix)
        cost_matrix_max = np.max(cost_matrix)
        max_distance = set_threshold(cost_matrix, max_distance, 0., cost_matrix_max)

    # Adjust cost matrix
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # Hungarian algorithm
    indices = linear_assignment(cost_matrix)

    # Initialization
    matches, unmatched_tracks, unmatched_detections = [], [], []

    # Update matching results 1
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[0]:
            unmatched_tracks.append(track_idx)

    # Update matching results 2
    for row, col in np.concatenate([indices[0][:, None], indices[1][:, None]], axis=1):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return np.array(matches), unmatched_tracks, unmatched_detections


def set_threshold(dists, ori_threshold, min_anchor, max_anchor):
    # # More Sampling with linear assignment (Do not use)
    # indices = linear_assignment(dists)
    # dists = dists[indices[0], indices[1]]

    # Prepare
    threshold = ori_threshold
    dists_1d = dists.reshape(-1, 1)
    dists_1d = dists_1d[dists_1d < max_anchor]
    dists_1d = dists_1d[min_anchor < dists_1d]

    if len(dists_1d) > 0:
        # Prepare
        dists_1d = list(dists_1d) + [min_anchor, max_anchor]
        dists_1d = np.array(dists_1d).reshape(-1, 1)

        # Select Clustering
        model = KMeans(n_clusters=2, init=np.array([[min_anchor], [max_anchor]]), n_init=1, random_state=10000)
        # model = AgglomerativeClustering(n_clusters=2, linkage='ward')
        # model = SpectralClustering(n_clusters=2, assign_labels='kmeans', random_state=10000)
        # model = GaussianMixture(n_components=2, means_init=np.array([[min_anchor], [max_anchor]]), random_state=10000)

        # Fit
        result = model.fit_predict(dists_1d)

        # Rare exception (Only occurs with Gaussian mixture clustering)
        if np.sum(result == 0) == 0 or np.sum(result == 1) == 0:
            return ori_threshold

        # Set threshold
        threshold = min(np.max(dists_1d[result == 0]), np.max(dists_1d[result == 1])) + 1e-5
        # threshold = max(np.min(dists_1d[result == 0]), np.min(dists_1d[result == 1])) - 1e-5
        # threshold = (np.max(dists_1d[result == 0]) + np.min(dists_1d[result == 1])) / 2
        # threshold = (np.mean(dists_1d[result == 0]) + np.mean(dists_1d[result == 1])) / 2

    return threshold


def split_cosine_dist(dets, trks, affinity_thresh=0.55, top=False):
    # cos_dist = sp.distance.cdist(dets, trks, "cosine")
    # cos_sim = 1 - cos_dist
    if top:
        cos_dist, cos_sim = _nn_res_recons_cosine_distance(dets, trks, data_is_normalized=False)
    else:
        cos_dist = sp.distance.cdist(dets, trks, "cosine")
        cos_sim = 1 - cos_dist
    # 初始化一个矩阵来存储每对检测和跟踪之间的最大余弦相似度
    max_cos_sim = np.zeros_like(cos_sim)

    # 遍历所有的检测
    for i in range(len(dets)):
        # 找出当前行的最大余弦相似度及其索引
        if cos_sim[i].size == 0:
            continue  # 或者设成一个默认值，比如 max_val = 0
        max_val = np.max(cos_sim[i, :])
        max_idx = np.argmax(cos_sim[i, :])

        # 如果最大值大于阈值，则将该位置设为最大值，其余位置设为零
        if max_val > affinity_thresh:
            max_cos_sim[i, max_idx] = max_val
        else:
            max_cos_sim[i, :] = 0
    return max_cos_sim


def compute_aw_new_metric(emb_cost, w_association_emb=0.75, max_diff=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)
    w_emb_bonus = np.full_like(emb_cost, 0)

    # Needs two columns at least to make sense to boost
    if emb_cost.shape[1] >= 2:
        # Across all rows
        for idx in range(emb_cost.shape[0]):
            inds = np.argsort(-emb_cost[idx])
            # Row weight is difference between top / second top
            row_weight = min(emb_cost[idx, inds[0]] - emb_cost[idx, inds[1]], max_diff)
            # Add to row
            w_emb_bonus[idx] += row_weight / 2

    if emb_cost.shape[0] >= 2:
        for idj in range(emb_cost.shape[1]):
            inds = np.argsort(-emb_cost[:, idj])
            col_weight = min(emb_cost[inds[0], idj] - emb_cost[inds[1], idj], max_diff)
            w_emb_bonus[:, idj] += col_weight / 2

    return w_emb + w_emb_bonus


def _nn_res_recons_cosine_distance(x, y, tmp=100, data_is_normalized=False):
    if not data_is_normalized:
        x = np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True)
        y = np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)

    ftrk = torch.from_numpy(np.asarray(x)).half().cuda()
    fdet = torch.from_numpy(np.asarray(y)).half().cuda()
    aff = torch.mm(ftrk, fdet.transpose(0, 1))
    aff_td = F.softmax(tmp * aff, dim=1)
    aff_dt = F.softmax(tmp * aff, dim=0).transpose(0, 1)

    res_recons_ftrk = torch.mm(aff_td, fdet)
    res_recons_fdet = torch.mm(aff_dt, ftrk)

    sim = (torch.mm(ftrk, fdet.transpose(0, 1)) + torch.mm(res_recons_ftrk,
                                                           res_recons_fdet.transpose(0, 1))) / 2
    distances = 1 - sim

    distances = distances.detach().cpu().numpy()
    sim = sim.detach().cpu().numpy()

    return distances, sim


def deep_associate(
        trackers,
        detections,
        trk_embs,
        det_embs,
        iou_threshold=0.3,
        w_assoc_emb=0.5,
        aw_off=True,
        aw_param=0.5,
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(trackers)),
            np.empty((0, 5), dtype=int),
        )
    iou_dists = matching.iou_distance(trackers, detections)
    iou_matrix = 1 - iou_dists
    emb_cost = split_cosine_dist(trk_embs, det_embs)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost[iou_matrix <= 0.3] = 0
                pass
            if aw_off:
                w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
                emb_cost *= w_matrix
            else:
                emb_cost *= w_assoc_emb
            final_cost = 1 - (iou_matrix + emb_cost)
            matched_indices = desort_linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 0]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[1])
            unmatched_trackers.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.stack(matches, axis=0).reshape(-1, 2)
    return matches, np.array(unmatched_trackers), np.array(unmatched_detections),


def desort_linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

