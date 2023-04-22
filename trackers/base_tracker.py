import numpy as np
import torch
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

# from trackers.base_tracklet import BaseTracklet


class BaseTracker():
    latest_id = 0
    def __init__(self, label) -> None:
        self.Tracklet = None

        self.label = label
        self.tracklets = []
        self.max_lifetime = 10   # 紐付かなかったら除去するフレーム数
        self.iou_matrix = None

        self.detect_pass_threshold = 0.4
        self.detect_register_threshold = 0.6

        self.match_threshold = 0.5

    def __call__(self, detect_boxes, detect_scores):
        # Detectionスコアで足切り
        pass_mask = detect_scores >= self.detect_pass_threshold
        detect_boxes = detect_boxes[pass_mask]
        detect_scores = detect_scores[pass_mask]

        # 全Trackletのカルマンフィルタ推定を行う
        self.predict()

        # TrackletとDetectionの紐付けを行い、Detectionの有効/無効なインデックスを作る。
        (active_detect_boxes, _,
         remain_detect_boxes, remain_detect_scores,
         active_tracklets, remain_tracklets)  = self.associate(detect_boxes, detect_scores, self.tracklets)

        # 紐付けされたTrackletsのみ更新を行う
        self.update(active_detect_boxes, active_tracklets)

        # 紐付けできなかったTrackletを残し、必要なら削除を行う
        remain_tracklets = self.hold(remain_tracklets)

        # 紐付けされなかったがスコアの高いDetectionをTrackletに追加する。固有IDもここで決定される
        new_tracklets = self.register(remain_detect_boxes, remain_detect_scores)

        # active/newを結合して新たなactiveにする
        active_tracklets = active_tracklets + new_tracklets

        # active/remainを結合してtrackletsにする
        self.tracklets = active_tracklets + remain_tracklets

        return active_tracklets, remain_tracklets

    def predict(self):
        for tracklet in self.tracklets:
            tracklet.predict()

    def associate(self, detect_boxes, detect_scores, tracklets):
        if len(detect_boxes) == 0:
            active_detect_boxes = []
            active_detect_scores = []
            remain_detect_boxes = []
            remain_detect_scores = []
            active_tracklets = []
            remain_tracklets = tracklets
        elif len(self.tracklets) == 0:
            active_detect_boxes = []
            active_detect_scores = []
            remain_detect_boxes = detect_boxes
            remain_detect_scores = detect_scores
            active_tracklets = []
            remain_tracklets = []
        else:
            track_boxes, _, _ = BaseTracker.get_pascal_boxes_labels_ids(tracklets)     # active/remainを含めた全てのTracklet
            self.iou_matrix = box_iou(track_boxes, detect_boxes)    # box_iouを使うためにTensorであること

            row_indices, col_indices = linear_sum_assignment(self.iou_matrix.numpy(), maximize=True)    # linear_sum_assignmentを使うためにndarrayであること

            match_mask = self.iou_matrix.numpy()[row_indices, col_indices] >= self.match_threshold

            active_track_indices = row_indices[match_mask]
            remain_track_indices = np.setdiff1d(np.arange(len(track_boxes)), active_track_indices)

            active_tracklets = [tracklets[index] for index in active_track_indices]
            remain_tracklets = [tracklets[index] for index in remain_track_indices]

            active_detect_indices = col_indices[match_mask]
            remain_detect_indices = np.setdiff1d(np.arange(len(detect_boxes)), active_detect_indices)

            active_detect_boxes = detect_boxes[active_detect_indices]
            active_detect_scores = detect_scores[active_detect_indices]
            remain_detect_boxes = detect_boxes[remain_detect_indices]
            remain_detect_scores = detect_scores[remain_detect_indices]
        
        return active_detect_boxes, active_detect_scores, remain_detect_boxes, remain_detect_scores, active_tracklets, remain_tracklets

    def update(self, detect_boxes, tracklets):
        for detect_box, tracklet in zip(detect_boxes, tracklets):
            # updateされたtrackletのlifetimeは0にリセットされる
            tracklet.update(detect_box.numpy())
            
    def hold(self, tracklets):
        remain_tracklets = []
        for tracklet in tracklets:
            if tracklet.get_lifetime() < self.max_lifetime:
                # max_lifetime未満のtrackletのみ保留する
                tracklet.lifetime += 1
                remain_tracklets.append(tracklet)
        return remain_tracklets

    def register(self, detect_boxes, detect_scores):
        new_tracklets = []
        for box, score in zip(detect_boxes, detect_scores):
            if score > self.detect_register_threshold:
                # 新規登録されたTrackletのlifetimeは0で初期化される
                new_tracklets.append(self.Tracklet(box.numpy(), self.label, self.latest_id))
                self.__class__.latest_id += 1
        return new_tracklets

    def get_label(self):
        return self.label

    @staticmethod
    def get_pascal_boxes_labels_ids(tracklets):
        if len(tracklets) == 0:
            boxes = torch.empty([0, 4])
            labels = torch.empty(0, dtype=torch.int64)
            ids = torch.empty(0, dtype=torch.int64)
        else:
            boxes = []
            labels = []
            ids = []
            for tracklet in tracklets:
                boxes.append(tracklet.get_pascal_box())
                labels.append(tracklet.get_label())
                ids.append(tracklet.get_id())
            boxes = torch.tensor(np.stack(boxes))
            labels = torch.tensor(labels, dtype=torch.int64)
            ids = torch.tensor(ids, dtype=torch.int64)

        return boxes, labels, ids
