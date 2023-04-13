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
\bold{x} =
\begin{bmatrix}
x_c & y_c & w & h & \frac{dx_c}{dt} & \frac{dy_c}{dt} & \frac{dw}{dt} & \frac{dh}{dt}
\end{bmatrix}^T
$$

**Observation space (Pascal box style)**

$$
\bold{z} = 
\begin{bmatrix}
x_1 & y_1 & x_2 & y_2
\end{bmatrix}^T
$$

**State transition matrix (YOLO box style and velocity)**

$$
\bold{F} = 
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
\bold{H} = 
\begin{bmatrix}
1 & 0 & -\frac{1}{2} & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & -\frac{1}{2} & 0 & 0 & 0 & 0 \\
1 & 0 & \frac{1}{2} & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & \frac{1}{2} & 0 & 0 & 0 & 0 \\
\end{bmatrix}
$$

**State extrapolation equation**

$$
\bold{x}_{t+1} = \bold{F} \bold{x}_t
$$

**Covariance extrapolation equation**

$$
\bold{P}_{t+1} = \bold{F} \bold{P}_t \bold{F}^T + \bold{Q} 
$$

**Kalman gain**

$$
\bold{K}_t = \bold{P}_{t-1} \bold{H}^T \lparen \bold{H} \bold{P}_{t-1} \bold{H}^T + \bold{R}_t \rparen^{-1}
$$

**State update equation**

$$
\bold{x}_t = \bold{x}_{t-1} + \bold{K}_t \lparen \bold{z}_t - \bold{H} \bold{x}_{t-1} \rparen
$$

**Covariance update equation**

$$
\bold{P}_t = \lparen \bold{I} - \bold{K}_t \bold{H} \rparen \bold{P}_{t-1} \lparen \bold{I} - \bold{K}_t \bold{H} \rparen^T + \bold{K}_t \bold{R}_t \bold{K}_t^T
$$