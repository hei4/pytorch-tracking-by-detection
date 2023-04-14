# pytorch-tracking-by-detection

**Tracking-by-Detection with PyTorch**


https://user-images.githubusercontent.com/26747044/231817109-aef7156b-9e26-4729-af2c-877cdca476a3.mp4




## How to use

```shell
python run.py <data_root> <result_root>
```

## Requirements

```shell
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install scipy

pip install matplotlib
```

## Kalman filter

**State space (YOLO box style and velocity)**

$$
\bf{x} =
\begin{bmatrix}
x_c & y_c & w & h & \frac{dx_c}{dt} & \frac{dy_c}{dt} & \frac{dw}{dt} & \frac{dh}{dt}
\end{bmatrix}^T
$$

**Observation space (Pascal box style)**

$$
\bf{z} = 
\begin{bmatrix}
x_1 & y_1 & x_2 & y_2
\end{bmatrix}^T
$$

**State transition matrix**

$$
\bf{F} = 
\begin{bmatrix}
1 & 0 & 0 & 0 & dt & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & dt & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & dt & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & dt \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

**Observation matrix**

$$
\bf{H} = 
\begin{bmatrix}
1 & 0 & -\frac{1}{2} & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & -\frac{1}{2} & 0 & 0 & 0 & 0 \\
1 & 0 & \frac{1}{2} & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & \frac{1}{2} & 0 & 0 & 0 & 0 \\
\end{bmatrix}
$$

**State extrapolation equation**

$$
\bf{x}_{t+1} = \bf{F} \bf{x}_t
$$

**Covariance extrapolation equation**

$$
\bf{P}_{t+1} = \bf{F} \bf{P}_t \bf{F}^T + \bf{Q}_t 
$$

**Kalman gain**

$$
\bf{K}_t = \bf{P}_{t-1} \bf{H}^T \lparen \bf{H} \bf{P}_{t-1} \bf{H}^T + \bf{R}_t \rparen^{-1}
$$

**State update equation**

$$
\bf{x}_t = \bf{x}_{t-1} + \bf{K}_t \lparen \bf{z}_t - \bf{H} \bf{x}_{t-1} \rparen
$$

**Covariance update equation**

$$
\bf{P}_t = \lparen \bf{I} - \bf{K}_t \bf{H} \rparen \bf{P}_{t-1} \lparen \bf{I} - \bf{K}_t \bf{H} \rparen^T + \bf{K}_t \bf{R}_t \bf{K}_t^T
$$
