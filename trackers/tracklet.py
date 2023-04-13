import numpy as np

class Tracklet():
    def __init__(self, detection, id) -> None:
        self.id = id
        self.lifetime = 0

        yolo_box = detection.get_yolo_box()
        num_observation = len(yolo_box)

        self.x = np.concatenate([yolo_box, np.zeros(num_observation)])   # 状態ベクトル [8]
        num_state = len(self.x)

        self.P = np.zeros([num_state, num_state])   # 状態共分散 [8x8]

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

        self.H = np.array([
            [1., 0., -0.5, 0., 0., 0., 0., 0.],
            [0., 1., 0., -0.5, 0., 0., 0., 0.],
            [1., 0., 0.5, 0., 0., 0., 0., 0.],
            [0., 1., 0., 0.5, 0., 0., 0., 0.],
        ])  # 観測行列 [4x8]

        # プロセスノイズの標準偏差
        self.std_process_xy_position = 0.05
        self.std_process_wh_position = 0.02
        self.std_process_xy_velocity = 4. * self.std_process_xy_position / (dt**2)
        self.std_process_wh_velocity = 4. * self.std_process_wh_position / (dt**2)

        # 観測ノイズの標準偏差
        self.std_observation_xy = 0.05

        self.I = np.eye(num_state)    # 単位行列 [8x8]
    
    def predict(self):
        # 状態方程式
        self.x = self.F @ self.x    # 本来は F@x+G@u だがここでは制御は無視する

        # プロセスノイズ共分散
        box_height = self.x[3]
        Q_std = np.concatenate([
            np.full(2, self.std_process_xy_position * box_height),
            np.full(2, self.std_process_wh_position * box_height),
            np.full(2, self.std_process_xy_velocity * box_height),
            np.full(2, self.std_process_wh_velocity * box_height),
        ])
        Q = np.diag(np.square(Q_std))

        # 共分散遷移
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, detection):
        # 観測値ベクトル [4]
        z = detection.get_pascal_box()
        
        # 観測ノイズ共分散
        box_height = self.x[3]
        R_std = np.full(4, self.std_observation_xy * box_height)
        R = np.diag(np.square(R_std))  # 観測ノイズ共分散 [4x4]

        # カルマンゲイン [8x4]
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)

        # 状態更新
        self.x = self.x + K @ (z - self.H @ self.x)

        # 共分散更新
        # self.P = (self.I - K @ self.H) @ self.P @ (self.I - K @ self.H).T + K @ self.R @ K.T
        self.P = (self.I - K @ self.H) @ self.P @ (self.I - K @ self.H).T + K @ R @ K.T

        # ライフライムのリセット
        self.lifetime = 0   

    def get_id(self):
        return self.id

    def get_lifetime(self):
        return self.lifetime
    
    def get_yolo_box(self):
        return self.x[:4]
    
    def get_pascal_box(self):
        return self.H @ self.x