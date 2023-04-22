# pytorch-tracking-by-detection

**Tracking-by-Detection with PyTorch**

## How to use

```shell
python run.py <tracking> <data_root> <result_root>
```

- tracking
  - kalman: Kalman filter tracking
  - byte: Byte track
  - particle: Particle filter tracking

## Requirements

```shell
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install scipy

pip install matplotlib
```

## Kalman filter

**State vector (YOLO box coordinate and velocity)**

$$
\bf{x} =
\begin{bmatrix}
    x_c & y_c & w & h & \frac{dx_c}{dt} & \frac{dy_c}{dt} & \frac{dw}{dt} & \frac{dh}{dt}
\end{bmatrix}^T
$$

**Observation vector (Pascal box coordinate)**

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

**Process noise uncertainty**

$$
\begin{align*}
    \sigma_x &= \alpha_x h \\
    \sigma_y &= \alpha_y h \\
    \sigma_w &= \alpha_w h \\
    \sigma_h &= \alpha_h h \\
    \bf{Q} &= 
    \begin{bmatrix}
        \frac{\sigma_x^2 \Delta t^5}{20} & 0 & 0 & 0 & \frac{\sigma_x^2 \Delta t^4}{8} & 0 & 0 & 0 \\
        0 & \frac{\sigma_y^2 \Delta t^5}{20} & 0 & 0 & 0 & \frac{\sigma_y^2 \Delta t^4}{8} & 0 & 0 \\
        0 & 0 & \frac{\sigma_w^2 \Delta t^5}{20} & 0 & 0 & 0 & \frac{\sigma_w^2 \Delta t^4}{8} & 0 \\
        0 & 0 & 0 & \frac{\sigma_h^2 \Delta t^5}{20} & 0 & 0 & 0 & \frac{\sigma_h^2 \Delta t^4}{8} \\
        \frac{\sigma_x^2 \Delta t^4}{8} & 0 & 0 & 0 & \frac{\sigma_x^2 \Delta t^3}{3} & 0 & 0 & 0 \\
        0 & \frac{\sigma_y^2 \Delta t^4}{8} & 0 & 0 & 0 & \frac{\sigma_y^2 \Delta t^3}{3} & 0 & 0 \\
        0 & 0 & \frac{\sigma_w^2 \Delta t^4}{8} & 0 & 0 & 0 & \frac{\sigma_w^2 \Delta t^3}{3} & 0 \\
        0 & 0 & 0 & \frac{\sigma_h^2 \Delta t^4}{8} & 0 & 0 & 0 & \frac{\sigma_h^2 \Delta t^3}{3} \\
    \end{bmatrix}
\end{align*}
$$

**Observation noise uncertainty**

$$
\begin{align*}
    \epsilon_x &= \beta_x h \\
    \epsilon_y &= \beta_y h \\
    \bf{R} &= 
    \begin{bmatrix}
        \epsilon_{x}^2 & 0 & 0 & 0 \\
        0 & \epsilon_{y}^2 & 0 & 0 \\
        0 & 0 & \epsilon_{x}^2 & 0 \\
        0 & 0 & 0 & \epsilon_{y}^2 \\
    \end{bmatrix}
\end{align*}
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

## Particle filter

**Likelihood**

$$
\begin{align*}
    &\bf{H}\bf{x} = \begin{bmatrix}
        \hat{x}_1 & \hat{y}_1 & \hat{x}_2 & \hat{y}_2
    \end{bmatrix}^T \\
    &\mathcal L(\mathbf{z}, \mathbf{x}) = \exp\left[ -\left(\frac{x_1 - \hat{x}_1}{\tau} \right)^2 -\left(\frac{y_1 - \hat{y}_1}{\tau} \right)^2 -\left(\frac{x_2 - \hat{x}_2}{\tau} \right)^2 -\left(\frac{y_2 - \hat{y}_2}{\tau} \right)^2\right]
\end{align*}
$$