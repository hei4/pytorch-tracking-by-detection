import numpy as np
from scipy.stats import chi2

from trackers.base_tracklet import BaseTracklet

class KalmanTracklet(BaseTracklet):
    def __init__(self, detect_box, label, id) -> None:
        super().__init__(detect_box, label, id)
        
        num_state = len(self.x)
        self.P = np.diag(np.full(num_state, fill_value=1e-6))   # 状態共分散 [8x8]
        
        # 観測ノイズに関する係数
        self.beta_x = 0.01
        self.beta_y = 0.01

        self.I = np.eye(num_state)    # 単位行列 [8x8]
    
    def predict(self):
        ax, bx, cx = self.calc_process_noise(self.alpha_x)
        ay, by, cy = self.calc_process_noise(self.alpha_y)
        aw, bw, cw = self.calc_process_noise(self.alpha_w)
        ah, bh, ch = self.calc_process_noise(self.alpha_h)

        # プロセスノイズ共分散
        Q = np.array([
            [ax, 0., 0., 0., bx, 0., 0., 0.],
            [0., ay, 0., 0., 0., by, 0., 0.],
            [0., 0., aw, 0., 0., 0., bw, 0.],
            [0., 0., 0., ah, 0., 0., 0., bh],
            [bx, 0., 0., 0., cx, 0., 0., 0.],
            [0., by, 0., 0., 0., cy, 0., 0.],
            [0., 0., bw, 0., 0., 0., cw, 0.],
            [0., 0., 0., bh, 0., 0., 0., ch],
        ])

        # 状態遷移
        self.x = self.F @ self.x    # 本来は F@x+G@u だがここでは制御は無視する

        # 共分散遷移
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, z):        
        # 観測ノイズ共分散 [4x4]
        ex = self.calc_observation_noise(self.beta_x)
        ey = self.calc_observation_noise(self.beta_y)
        R = np.array([
            [ex, 0., 0., 0.],
            [0., ey, 0., 0.],
            [0., 0., ex, 0.],
            [0., 0., 0., ey],
        ])

        # カルマンゲイン [8x4]
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)

        # 状態更新
        self.x = self.x + K @ (z - self.H @ self.x)

        # 共分散更新
        self.P = (self.I - K @ self.H) @ self.P @ (self.I - K @ self.H).T + K @ R @ K.T

        # ライフライムのリセット
        self.lifetime = 0   

    def calc_process_noise(self, alpha):
        height = self.x[3]
        sigma = alpha * height
        a = sigma**2 * self.dt**5 / 20.
        b = sigma**2 * self.dt**4 / 8.
        c = sigma**2 * self.dt**3 / 3.
        return a, b, c
    
    def calc_observation_noise(self, beta):
        height = self.x[3]
        epsilon = beta * height
        return epsilon**2

    def get_yolo_box(self):
        return self.x[:4]
    
    def get_pascal_box(self):
        return self.H @ self.x
    
    def get_ellipse(self):
        mean = self.x[:2]
        cov = self.P[:2, :2]
        
        eigenvalue, eigenvector = np.linalg.eig(cov)

        if eigenvalue[0] >= eigenvalue[1]:
            value = eigenvalue
            vector = eigenvector[:, 0]
        else:
            value = eigenvalue[::-1]
            vector = eigenvector[:, 1]

        # intervalの信頼区間は両側で算出されるので、片側にするには2倍する
        confidece_interval = chi2.interval(confidence=(1-0.01*2), df=2)[1]

        size = np.sqrt(confidece_interval * value)
        angle = np.arctan2(vector[1], vector[0])

        return mean, size[0], size[1], np.rad2deg(angle)
