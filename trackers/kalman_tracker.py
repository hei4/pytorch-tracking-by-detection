from trackers.base_tracker import BaseTracker
from trackers.kalman_tracklet import KalmanTracklet

class KalmanTracker(BaseTracker):  
    def __init__(self, label) -> None:
        super().__init__(label)
        self.Tracklet = KalmanTracklet
