import numpy as np

from trackers.base_tracklet import BaseTracklet

class ParticleTracklet(BaseTracklet):
    def __init__(self, detect_box, label, id) -> None:
        super().__init__(detect_box, label, id)

        self.N = 65536
        self.W = np.full(self.N, 1. / self.N)    # 粒子重み

        self.zeros = np.zeros_like(self.x)
        self.x = np.repeat(self.x[:, np.newaxis], self.N, axis=1)   # パーティクル [8, N]
        
        self.tau = 10.  # 尤度用の係数
    
    def predict(self):  # motion update
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

        noise = np.random.multivariate_normal(mean=self.zeros, cov=Q, size=self.N)

        # 状態遷移
        self.x = self.F @ self.x + noise.T

    def update(self, z):        
        # 尤度 [N]
        L = np.exp(-np.sum(((z[:, np.newaxis] - self.H @ self.x) / self.tau)**2., axis=0))

        self.W *= L
        sum_weights = np.sum(self.W)
        if np.isnan(sum_weights) == True:
            self.W = np.full(self.N, 1. / self.N)
        else:
            self.W /= sum_weights    # 重みの正規化

        # リサンプリング
        indices = np.random.choice(np.arange(self.N), size=self.N, replace=True, p=self.W)
        self.x = self.x[:, indices]
        self.W = 1. / self.N

        # ライフライムのリセット
        self.lifetime = 0          

    def expect(self):
        return np.sum(self.x * self.W, axis=1)
    
    def calc_process_noise(self, alpha):
        expect_x = self.expect()
        height = expect_x[3]
        sigma = alpha * height
        a = sigma**2 * self.dt**5 / 20.
        b = sigma**2 * self.dt**4 / 8.
        c = sigma**2 * self.dt**3 / 3.
        return a, b, c

    def get_yolo_box(self):
        return self.expect()[:4]
    
    def get_pascal_box(self):
        return self.H @ self.expect()
    
    def get_points(self):
        return self.x[:2]
    