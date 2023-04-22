import numpy as np
from scipy.stats import chi2

class BaseTracklet():
    def __init__(self, detect_box, label, id) -> None:
        self.label = label
        self.id = id
        self.lifetime = 0

        self.x = np.array([
            0.5 * (detect_box[2] + detect_box[0]),
            0.5 * (detect_box[3] + detect_box[1]),
            detect_box[2] - detect_box[0],
            detect_box[3] - detect_box[1],
            0.,
            0.,
            0.,
            0.
        ])
        
        dt = 1.    # 時間ステップ
        self.F = np.array([
            [1., 0., 0., 0., dt, 0., 0., 0.],
            [0., 1., 0., 0., 0., dt, 0., 0.],
            [0., 0., 1., 0., 0., 0., dt, 0.],
            [0., 0., 0., 1., 0., 0., 0., dt],
            [0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.],
        ])  # システム行列 [8x8]
        self.dt = dt

        self.H = np.array([
            [1., 0., -0.5, 0., 0., 0., 0., 0.],
            [0., 1., 0., -0.5, 0., 0., 0., 0.],
            [1., 0., 0.5, 0., 0., 0., 0., 0.],
            [0., 1., 0., 0.5, 0., 0., 0., 0.],
        ])  # 観測行列 [4x8]

        # プロセスノイズに関する係数
        self.alpha_x = 0.010
        self.alpha_y = 0.001
        self.alpha_w = 0.005
        self.alpha_h = 0.001

    def get_label(self):
        return self.label

    def get_id(self):
        return self.id

    def get_lifetime(self):
        return self.lifetime
    