import numpy as np
import torch
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

from trackers.base_tracker import BaseTracker
from trackers.particle_tracklet import ParticleTracklet

class ParticleTracker(BaseTracker):
    def __init__(self, label) -> None:
        super().__init__(label)
        self.Tracklet = ParticleTracklet

