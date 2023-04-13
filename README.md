# pytorch-tracking-by-detection

**Tracking-by-Detection with PyTorch**



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

**State space**

$$
\bm{x} =
\begin{bmatrix}
x_c & y_c & w & h & \frac{dx_c}{dt} & \frac{dy_c}{dt} & \frac{dw}{dt} & \frac{dh}{dt}
\end{bmatrix}^T
$$

**Observation space (Pascal box style)**

$$
\bm{z} = 
\begin{bmatrix}
x_1 & y_1 & x_2 & y_2
\end{bmatrix}^T
$$

**State transition matrix (YOLO box style and velocity)**

$$
\bm{F} = 
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
\bm{H} = 
\begin{bmatrix}
1 & 0 & -\frac{1}{2} & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & -\frac{1}{2} & 0 & 0 & 0 & 0 \\
1 & 0 & \frac{1}{2} & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & \frac{1}{2} & 0 & 0 & 0 & 0 \\
\end{bmatrix}
$$

**State extrapolation equation**

$$
\bm{x}_{t+1} = \bm{F} \bm{x}_t
$$

**Covariance extrapolation equation**

$$
\bm{P}_{t+1} = \bm{F} \bm{P}_t \bm{F}^T + \bm{Q} 
$$

**Kalman gain**

$$
\bm{K}_t = \bm{P}_{t-1} \bm{H}^T \lparen \bm{H} \bm{P}_{t-1} \bm{H}^T + \bm{R}_t \rparen^{-1}
$$

**State update equation**

$$
\bm{x}_t = \bm{x}_{t-1} + \bm{K}_t \lparen \bm{z}_t - \bm{H} \bm{x}_{t-1} \rparen
$$

**Covariance update equation**

$$
\bm{P}_t = \lparen \bm{I} - \bm{K}_t \bm{H} \rparen \bm{P}_{t-1} \lparen \bm{I} - \bm{K}_t \bm{H} \rparen^T + \bm{K}_t \bm{R}_t \bm{K}_t^T
$$