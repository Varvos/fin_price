# Introduction to Variance Swaps and VIX Options

This document provides an introduction to variance swaps and VIX options, focusing on their pricing and the underlying mathematical concepts. We explore the relationship between variance swaps and VIX options, and how to price them using various models.

## Cont-Kokholm Model

### Forward Variance Swap Rate

The forward variance swap rate, quoted at date t for the period [T_i, T_{i+1}], is the strike that sets the value of a forward variance swap running from T₁ to T₂ to zero at time t. It is given by:

$$
V^{T_1, T_{2}}_t := \frac{\mathbb{E}[[\log S]_{T_{2}} - [\log S]_{T_1}|\mathcal{F}_t]}{T_{2}-T_1}
$$

Forward variance swap rates are martingales.

We define the forward variance swap rate over the time interval [T_i, T_{i+1}] as:
$$
V^i_t := V^{T_i, T_{i+1}}_t
$$

## Variance Swap Dynamics

Given that the forward variance swap rate is a (positive) martingale under the pricing measure, we model it as:

$$V_{t}^{i}=V_{0}^{t} e^{X_{i}}
=V_{0}^{i} \exp \left\{{\color{red}\int_{0}^{t} \mu_{s}^{i} d s}+{\color{blue}\int_{0}^{t}\omega e^{-k_{1}\left(T_{i}-s\right)} d Z_{s}}+{\color{green}\int_{0}^{t} \int_{\mathbb{R}} e^{-k_{2}\left(T_{i}-s\right)} x J(d x ds)}\right\}$$

where:
- **J(dx dt)** is a Poisson random measure with compensator ν(dx)dt
- **Z** is a Wiener process independent of J
- The **red term** represents the drift component
- The **blue term** represents the diffusion component
- The **green term** represents the jump component

The martingale condition imposes:
$$
\mu_{t}^{i}=-\frac{1}{2} \omega^{2} e^{-2 k_{1}\left(T_{i}-t\right)}-\int_{\mathbb{R}}\left(\exp \left\{e^{-k_{2}\left(T_{i}-t\right)} x\right\}-1\right) v(d x)
$$

For t > T_i, we let V^i_t = V^i_{T_i}.

In the case of finite jump intensity, equation (3.1) reduces to:

$$
V_{t}^{i}=V_{0}^{i} \exp \left\{\int_{0}^{t} \mu_{s}^{i} d s+\int_{0}^{t} \omega e^{-k_{1}\left(T_{i}-s\right)} d Z_{s}+\sum_{j=0}^{N_{t}} e^{-k_{2}\left(T_{i}-\tau_{j}\right)} Y_{j}\right\}
$$

## Jump Distributions

### Gaussian Jumps (Merton Model)
$$
v(d x)=\lambda \frac{1}{\delta \sqrt{2 \pi}} e^{-\frac{(x-m)^{2}}{2 \delta^{2}}} d x
$$

where:
- **λ** is the jump intensity
- **m** is the mean jump size
- **δ** is the standard deviation of jump sizes

### Kou's Two-Sided Exponential Distribution
$$
v(d x)=\lambda\left(p \alpha_{+} e^{-\alpha_{+} x} \mathbf{1}_{x \geq 0}+(1-p) \alpha_{-} e^{-\alpha_{-}|x|} \mathbf{1}_{x<0}\right) d x
$$

where:
- **λ** is the jump intensity  
- **p** is the probability of an upward jump
- **α₊** is the rate parameter for upward jumps
- **α₋** is the rate parameter for downward jumps

## Drift Coefficient

The drift coefficient is given by:
$$
 \mu^i(t) := -\frac{1}{2}\omega^2 e^{-2k_1(T_i-t)} - \int_{\mathbb{R}}(exp(-k_2(T_i-t) x) - 1)\nu(dx)
$$

The total drift term in the variance swap dynamics is calculated by integrating over time:

### For Merton Model:
$$\int_0^{T_i} \mu^i(t)dt = -\frac{\omega^2 }{2 }\times\frac{(1- e^{-2k_1 T_i})}{2 k_1}\,-\, \lambda \int_0^{T_i} \left(e^{m e^{-k_2 (T_i-s)}+ \frac{1}{2} e^{-2 k_2 (T_i-s)} \delta^2}-1 \right)ds$$

### For Kou Model:
$$\int_0^{T_i} \mu^i(t)dt = -\frac{\omega^2 }{2 }\times \frac{(1- e^{-2k_1 T_i})}{2 k_1}\,-\,
\lambda \int_0^{T_i} \left(p \frac{\alpha_{+}}{\alpha_{+} - e^{-k_2 (T_i-s)}}+(1-p)\frac{\alpha_{-}}{\alpha_{-} + e^{-k_2(T_i-s)}}-1\right)ds$$

## Characteristic Functions

Characteristic functions of variance swaps and VIX futures can be derived from the Cont-Kokholm model and are essential for Fourier-based pricing methods. These functions provide the foundation for pricing various derivative instruments on volatility indices.

## Model Parameters

The model is characterized by the following key parameters:

- **ω**: Volatility of the jump process
- **k₁**: Rate parameter for diffusion mean reversion  
- **k₂**: Rate parameter for jump size decay
- **λ**: Jump intensity (frequency of jumps)

### Additional Parameters by Model:
- **Merton Model**: m (mean jump size), δ (jump size standard deviation)
- **Kou Model**: p (upward jump probability), α₊ (upward jump rate), α₋ (downward jump rate)