import numpy as np
import torch
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

from tracklet import Tracklet
from detection import Detection


# ByteTrackの場合
# self.detect_th_low = 0.1
# self.detect_th_high = 0.45
# self.detect_th_regist = 0.55

# self.match_th_high = 0.8
# self.match_th_low = 0.5

class Tracker():
    def __init__(self, classname) -> None:
        self.classname = classname
        self.tracklets = []
        self.max_lifetime = 10   # 紐付かなかったら消す
        self.latest_id = 0
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
        (active_detections, remain_detections,
         active_tracklets, remain_tracklets)  = self.associate(detect_boxes, detect_scores)

        # 紐付けされたTrackletsのみカルマンフィルタ更新を行う
        self.update(active_detections, active_tracklets)

        # 紐付けできなかったTrackletを残し、必要なら削除を行う
        remain_tracklets = self.hold(remain_tracklets)

        # 紐付けされなかったがスコアの高いDetectionをTrackletに追加する。固有IDもここで決定される
        new_tracklets = self.register(remain_detections)

        # active/remain/newを結合してtrackletsにする
        self.tracklets = active_tracklets + remain_tracklets + new_tracklets

        # Trackletsから矩形座標を取得する
        active_track_boxes, active_track_ids = self.get_pascal_boxes_and_ids(active_tracklets)
        remain_track_boxes, remain_track_ids = self.get_pascal_boxes_and_ids(remain_tracklets)

        return active_track_boxes, active_track_ids, remain_track_boxes, remain_track_ids

    def predict(self):
        for tracklet in self.tracklets:
            tracklet.predict()

    def associate(self, detect_boxes, detect_scores):
        """TrackletとDetectionの紐付けを行う

        - IoUマトリックスを作り記憶する
        - Detectionインスタンスタンスのリストを作成する

        Args:
            detect_boxes (_type_): _description_
            detect_scores (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(detect_boxes) == 0:
            active_detections = []
            remain_detections = []
            active_tracklets = []
            remain_tracklets = self.tracklets
        elif len(self.tracklets) == 0:
            active_detections = []
            remain_detections = [Detection(box, score) for box, score in zip(detect_boxes, detect_scores)]
            active_tracklets = []
            remain_tracklets = []
        else:
            track_boxes, _ = self.get_pascal_boxes_and_ids(self.tracklets)     # active/remainを含めた全てのTracklet
            self.iou_matrix = box_iou(track_boxes, detect_boxes)    # box_iouを使うためにTensorであること

            row_indices, col_indices = linear_sum_assignment(self.iou_matrix.numpy(), maximize=True)    # linear_sum_assignmentを使うためにndarrayであること

            match_mask = self.iou_matrix.numpy()[row_indices, col_indices] >= self.match_threshold

            active_track_indices = row_indices[match_mask]
            remain_track_indices = np.setdiff1d(np.arange(len(track_boxes)), active_track_indices)

            active_tracklets = [self.tracklets[index] for index in active_track_indices]
            remain_tracklets = [self.tracklets[index] for index in remain_track_indices]

            active_detect_indices = col_indices[match_mask]
            remain_detect_indices = np.setdiff1d(np.arange(len(detect_boxes)), active_detect_indices)

            active_detections = [Detection(detect_boxes[index], detect_scores[index]) for index in active_detect_indices]
            remain_detections = [Detection(detect_boxes[index], detect_scores[index]) for index in remain_detect_indices]
        
        return active_detections, remain_detections, active_tracklets, remain_tracklets

    def update(self, detections, tracklets):
        """紐付けされたDetectionとTrackletでカルマンフィルタの更新を行う

        Args:
            detections (_type_): trackletsと対応がとれた順番であること
            tracklets (_type_): detectionsと対応がとれた順番であること
        """
        for detection, tracklet in zip(detections, tracklets):
            # updateされたtrackletのlifetimeは0にリセットされる
            tracklet.update(detection)
            
    def hold(self, tracklets):
        """紐付けされなかったTrackletを保留する

        Args:
            tracklets (_type_): _description_

        Returns:
            _type_: _description_
        """
        remain_tracklets = []
        for tracklet in tracklets:
            if tracklet.get_lifetime() < self.max_lifetime:
                # max_lifetime未満のtrackletのみ保留する
                tracklet.lifetime += 1
                remain_tracklets.append(tracklet)
        return remain_tracklets

    def register(self, detections):
        """紐付けされなかったがスコアの高いDetectionをTrackletとして登録する

        Args:
            detections (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_tracklets = []
        for detection in detections:
            if detection.score > self.detect_register_threshold:
                # 新規登録されたTrackletのlifetimeは0で初期化される
                new_tracklets.append(Tracklet(detection, self.latest_id))
                self.latest_id += 1
        return new_tracklets

    def get_pascal_boxes_and_ids(self, tracklets):
        if len(tracklets) == 0:
            boxes = torch.empty([0, 4])
            ids = torch.empty(0, dtype=torch.int64)
        else:
            boxes = []
            ids = []
            for tracklet in tracklets:
                boxes.append(tracklet.get_pascal_box())
                ids.append(tracklet.get_id())
            boxes = torch.tensor(np.stack(boxes))
            ids = torch.tensor(ids, dtype=torch.int64)

        return boxes, ids
    
    def get_classname(self):
        return self.classname