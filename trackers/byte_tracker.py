import torch

from trackers.base_tracker import BaseTracker
from trackers.kalman_tracklet import KalmanTracklet

class ByteTracker(BaseTracker):  
    def __init__(self, label) -> None:
        super().__init__(label)
        self.Tracklet = KalmanTracklet

        self.detect_low_threshold = 0.2
        self.detect_high_threshold = 0.4
        self.detect_register_threshold = 0.6
    
    def __call__(self, detect_boxes, detect_scores):
        # Lowスコアで足切りして、High-DetectionとLow-Detectionに区分
        high_mask = detect_scores >= self.detect_high_threshold
        low_mask = detect_scores >= self.detect_low_threshold
        low_mask = torch.logical_xor(high_mask, low_mask)

        detect_high_boxes = detect_boxes[high_mask]
        detect_high_scores = detect_scores[high_mask]

        detect_low_boxes = detect_boxes[low_mask]
        detect_low_scores = detect_boxes[low_mask]
        
        # 全Trackletのカルマンフィルタ推定を行う
        self.predict()

        # Trackletと高スコアDetectionの紐付けを行い、Detectionの有効/無効なインデックスを作る。
        (active_detect_boxes, _,
         remain_detect_boxes, remain_detect_scores,
         active_high_tracklets, remain_high_tracklets)  = self.associate(detect_high_boxes, detect_high_scores, self.tracklets)

        # 紐付けされたTrackletsのみ更新を行う
        self.update(active_detect_boxes, active_high_tracklets)

        # 残ったTrackletと低スコアDetectionの紐付けを行い、Detectionの有効/無効なインデックスを作る。
        (active_detect_boxes, _,
         _, _,
         active_low_tracklets, remain_low_tracklets)  = self.associate(detect_low_boxes, detect_low_scores, remain_high_tracklets)

        # 紐付けされたTrackletsのみ更新を行う
        self.update(active_detect_boxes, active_low_tracklets)

        # 紐付けされなかった高スコアDetectionをTrackletに追加する。固有IDもここで決定される
        new_tracklets = self.register(remain_detect_boxes, remain_detect_scores)

        # active-high/active-low/newを結合して新たなactiveにする
        active_tracklets = active_high_tracklets + active_low_tracklets + new_tracklets

        # 紐付けできなかったTrackletを残し、必要なら削除を行う
        remain_tracklets = self.hold(remain_low_tracklets)

        # active/remainを結合してtrackletsにする
        self.tracklets = active_tracklets + remain_tracklets

        return active_tracklets, remain_tracklets


