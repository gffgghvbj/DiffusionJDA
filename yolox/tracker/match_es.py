import numpy as np
from scipy.optimize import linear_sum_assignment


def refine_association_matrix(current_detections,
                              prev_tracks,
                              assoc_matrix,
                              motion_weight=0.4,
                              iou_threshold=0.3,
                              confidence_threshold=0.6):
    """
    优化关联矩阵的后处理函数
    :param current_detections: 当前帧检测框 (N x 4) [x1,y1,x2,y2]
    :param prev_tracks: 上一帧跟踪结果 (M x 4) [x1,y1,x2,y2]
    :param assoc_matrix: 模型输出的关联矩阵 (M x N)
    :param motion_weight: 运动一致性权重 (0-1)
    :param iou_threshold: IOU匹配阈值
    :param confidence_threshold: 最终置信度阈值
    :return: 优化后的匹配对，未匹配检测，未匹配轨迹
    """
    # 1. 运动一致性计算
    motion_sim = compute_motion_similarity(current_detections, prev_tracks)

    # 2. 融合原始关联矩阵和运动一致性
    fused_matrix = (1 - motion_weight) * assoc_matrix + motion_weight * motion_sim

    # 3. 双向一致性校验
    refined_matrix = bidirectional_consistency_check(fused_matrix)

    # 4. 轨迹连续性约束
    final_matrix = apply_trajectory_constraints(refined_matrix, prev_tracks, current_detections)

    # 5. 执行匹配
    row_idx, col_idx = linear_sum_assignment(-final_matrix)
    matches = [(r, c) for r, c in zip(row_idx, col_idx)
               if final_matrix[r, c] > confidence_threshold]

    # 6. 处理未匹配项
    matched_rows = set(r for r, _ in matches)
    matched_cols = set(c for _, c in matches)

    unmatched_tracks = [r for r in range(final_matrix.shape[0]) if r not in matched_rows]
    unmatched_detections = [c for c in range(final_matrix.shape[1]) if c not in matched_cols]

    return matches, unmatched_detections, unmatched_tracks


def compute_motion_similarity(detections, tracks):
    """
    基于运动预测的相似度计算
    """
    # 预测当前位置（简单线性外推）
    pred_positions = []
    for track in tracks:
        if len(track.history) >= 2:
            # 使用最后两个位置计算速度
            prev1 = track.history[-1]
            prev2 = track.history[-2]
            velocity = (prev1[:2] - prev2[:2]) / (prev1[2:] + prev2[2:])  # 标准化速度
            pred_pos = prev1[:2] + velocity * (prev1[2:] + prev1[2:]) / 2
        else:
            pred_pos = track.current_bbox[:2]
        pred_positions.append(pred_pos)

    # 计算预测位置与检测的IOU
    similarity = np.zeros((len(tracks), len(detections)))
    for i, pred in enumerate(pred_positions):
        for j, det in enumerate(detections):
            similarity[i, j] = calculate_iou(pred, det)

    return similarity


def bidirectional_consistency_check(matrix):
    """
    双向匹配一致性校验
    """
    row_matches = np.argmax(matrix, axis=1)
    col_matches = np.argmax(matrix, axis=0)

    mask = np.zeros_like(matrix)
    for i, j in enumerate(row_matches):
        if col_matches[j] == i:
            mask[i, j] = matrix[i, j]

    return mask


def apply_trajectory_constraints(matrix, tracks, detections):
    """
    轨迹连续性约束
    """
    # 检查尺寸突变
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                track_size = tracks[i].current_bbox[2] * tracks[i].current_bbox[3]
                det_size = detections[j][2] * detections[j][3]
                size_ratio = max(track_size / det_size, det_size / track_size)
                if size_ratio > 4:  # 面积变化超过4倍视为异常
                    matrix[i, j] *= 0.5
    return matrix


def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框的IOU
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - intersection

    return intersection / (union + 1e-8)


def validate_association_matrix(assoc_matrix, confidence_thresh=0.8):
    """
    对模型输出的关联矩阵进行双向一致性验证和置信度过滤
    :param assoc_matrix: 模型输出的关联矩阵 (M tracks x N detections)
    :param confidence_thresh: 置信度阈值
    :return: 修正后的关联矩阵
    """
    # 深拷贝原始矩阵避免修改原数据
    validated_matrix = np.copy(assoc_matrix)

    # 第一步：置信度过滤
    validated_matrix[validated_matrix < confidence_thresh] = 0

    # 第二步：双向一致性检查
    # 从track到detection的最优匹配
    track_to_det = np.argmax(validated_matrix, axis=1)
    # 从detection到track的最优匹配
    det_to_track = np.argmax(validated_matrix, axis=0)

    # 创建修正矩阵
    corrected_matrix = np.zeros_like(validated_matrix)

    # 只保留双向一致的匹配
    for track_idx, det_idx in enumerate(track_to_det):
        if det_to_track[det_idx] == track_idx:
            corrected_matrix[track_idx, det_idx] = validated_matrix[track_idx, det_idx]

    return corrected_matrix


def motion_aware_association_refinement(assoc_matrix, tracks, detections,
                                        prev_tracks, prev_detections,
                                        motion_weight=0.3):
    """
    结合运动一致性修正关联矩阵
    :param tracks: 当前帧跟踪状态 [M x 4] (x1,y1,x2,y2)
    :param detections: 当前帧检测 [N x 4]
    :param prev_tracks: 上一帧跟踪状态
    :param prev_detections: 上一帧检测
    :param motion_weight: 运动一致性的权重
    """
    # 计算运动一致性矩阵
    motion_matrix = compute_motion_consistency(tracks, detections,
                                               prev_tracks, prev_detections)

    # 融合原始关联矩阵和运动一致性矩阵
    refined_matrix = (1 - motion_weight) * assoc_matrix + motion_weight * motion_matrix

    # 归一化
    refined_matrix = refined_matrix / np.max(refined_matrix, axis=1, keepdims=True)

    return refined_matrix


def compute_motion_consistency(tracks, detections, prev_tracks, prev_detections):
    """
    计算基于运动一致性的得分矩阵
    """
    M = len(tracks)
    N = len(detections)
    motion_matrix = np.zeros((M, N))

    # 计算所有track的运动向量
    track_motions = []
    for i, (track, prev_track) in enumerate(zip(tracks, prev_tracks)):
        if i >= len(prev_tracks):
            track_motions.append(np.zeros(2))
            continue
        track_motions.append(bbox_motion(track, prev_track))

    # 计算所有detection的运动向量
    det_motions = []
    for j, (det, prev_det) in enumerate(zip(detections, prev_detections)):
        if j >= len(prev_detections):
            det_motions.append(np.zeros(2))
            continue
        det_motions.append(bbox_motion(det, prev_det))

    # 计算运动一致性得分
    for i in range(M):
        for j in range(N):
            # 余弦相似度
            cos_sim = np.dot(track_motions[i], det_motions[j]) / (
                    np.linalg.norm(track_motions[i]) * np.linalg.norm(det_motions[j]) + 1e-8)
            motion_matrix[i, j] = (cos_sim + 1) / 2  # 归一化到[0,1]

    return motion_matrix


def bbox_motion(current_bbox, prev_bbox):
    """计算边界框中心点运动向量"""
    curr_center = np.array([(current_bbox[0] + current_bbox[2]) / 2,
                            (current_bbox[1] + current_bbox[3]) / 2])
    prev_center = np.array([(prev_bbox[0] + prev_bbox[2]) / 2,
                            (prev_bbox[1] + prev_bbox[3]) / 2])
    return curr_center - prev_center


class TrajectoryValidator:
    def __init__(self, smoothness_window=3, deviation_thresh=0.2):
        self.smoothness_window = smoothness_window
        self.deviation_thresh = deviation_thresh
        self.trajectories = {}  # {track_id: [positions]}

    def validate_associations(self, assoc_matrix, tracks, detections):
        """
        基于轨迹平滑性验证关联
        :return: 验证后的关联矩阵
        """
        validated_matrix = np.copy(assoc_matrix)

        # 获取检测框和跟踪框的中心
        det_centers = np.array([[(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in detections])
        trk_centers = np.array([[(t[0] + t[2]) / 2, (t[1] + t[3]) / 2] for t in tracks])

        # 对每个可能的关联进行检查
        for i in range(assoc_matrix.shape[0]):
            for j in range(assoc_matrix.shape[1]):
                if assoc_matrix[i, j] > 0:
                    if not self._check_smoothness(i, trk_centers[i], j, det_centers[j]):
                        validated_matrix[i, j] *= 0.5  # 惩罚不平滑的关联

        return validated_matrix

    def _check_smoothness(self, track_id, track_pos, det_id, det_pos):
        """检查关联是否满足轨迹平滑性"""
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []
            return True  # 新轨迹不检查

        if len(self.trajectories[track_id]) < 2:
            return True  # 历史点不足不检查

        # 预测下一位置 (简单线性外推)
        last = self.trajectories[track_id][-1]
        prev = self.trajectories[track_id][-2]
        velocity = last - prev
        predicted_pos = last + velocity

        # 计算预测偏差
        deviation = np.linalg.norm(det_pos - predicted_pos)
        avg_movement = np.linalg.norm(velocity)

        # 归一化偏差
        norm_deviation = deviation / (avg_movement + 1e-8)

        return norm_deviation < self.deviation_thresh

    def update_trajectories(self, tracks, assignments):
        """更新轨迹历史"""
        trk_centers = np.array([[(t[0] + t[2]) / 2, (t[1] + t[3]) / 2] for t in tracks])

        for i, j in assignments:
            if i >= len(trk_centers):
                continue
            if i not in self.trajectories:
                self.trajectories[i] = []
            self.trajectories[i].append(trk_centers[i])
            if len(self.trajectories[i]) > self.smoothness_window:
                self.trajectories[i].pop(0)


def full_association_postprocess(assoc_matrix, tracks, detections,
                                 prev_tracks=None, prev_detections=None,
                                 validator=None):
    """
    完整的关联矩阵后处理流程
    :param validator: 可选的TrajectoryValidator实例
    """
    # 第一步：基础验证
    processed_matrix = validate_association_matrix(assoc_matrix)

    # 如果有上一帧信息，进行运动一致性优化
    if prev_tracks is not None and prev_detections is not None:
        processed_matrix = motion_aware_association_refinement(
            processed_matrix, tracks, detections,
            prev_tracks, prev_detections)

    # 如果有轨迹验证器，进行平滑性验证
    if validator is not None:
        processed_matrix = validator.validate_associations(
            processed_matrix, tracks, detections)

    # 最终双向一致性检查
    final_matrix = validate_association_matrix(processed_matrix)
    # 提取最终匹配
    matches = []
    rows = []
    cols = []
    for i in range(final_matrix.shape[0]):
        j = np.argmax(final_matrix[i])
        if final_matrix[i, j] > 0:
            rows.append(i)
            cols.append(j)
    matches = np.array(list(zip(rows, cols)), dtype=int).reshape(-1, 2)

    # 更新轨迹验证器
    if validator is not None:
        validator.update_trajectories(tracks, matches)

    return matches, final_matrix


def asso_matrix_matches(matrix, threshold=0.8):
    """
    处理关联矩阵并返回三个结果：
    1. 超过阈值的元素位置 (n*2数组，第一列是行索引，第二列是列索引)
    2. 没有任何元素超过阈值的行索引
    3. 没有任何元素超过阈值的列索引

    参数:
        matrix: 二维numpy数组
        threshold: 阈值 (默认0.8)

    返回:
        tuple: (over_threshold, rows_below_threshold, cols_below_threshold)
    """
    # 转换为numpy数组以确保操作正确
    matrix = np.array(matrix)

    # 1. 找出超过阈值的元素位置
    over_threshold = np.argwhere(matrix > threshold)

    # 2. 找出没有任何元素超过阈值的行
    row_max = np.max(matrix, axis=1)  # 每行的最大值
    rows_below_threshold = np.where(row_max <= threshold)[0].tolist()

    # 3. 找出没有任何元素超过阈值的列
    col_max = np.max(matrix, axis=0)  # 每列的最大值
    cols_below_threshold = np.where(col_max <= threshold)[0].tolist()

    return over_threshold, rows_below_threshold, cols_below_threshold