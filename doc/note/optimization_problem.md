# 两组分囊泡相场模型 — 优化问题

问题：给定初始的物理设置后，如果可能发生相分析，最后相分离的状态是什么？

## 区域与决策变量

$$\Omega = [-\pi,\pi]^3 \subset \mathbb{R}^3$$

$$\phi = \phi(x),\quad \eta = \eta(x),\qquad x\in\Omega$$

零水平集:
$$\Gamma = \{x:\phi(x)=0\},\qquad \Gamma_\perp=\{x:\eta(x)=0\},\qquad \gamma_0=\{x:\phi(x)=\eta(x)=0\}$$

## 物理系数

$$k(\eta) = k + c\,\tanh(\eta/\xi),\qquad k_1=k+c,\quad k_2=k-c$$

$\eta, \xi$ 定义的diffuse边界参数，$\epsilon = \xi$

$$2c_0(\eta) = (c_1+c_2) + (c_1-c_2)\tanh(\eta/\xi)$$

## 能量泛函

弯曲能 (Eq. 3):
$$
E(\phi,\eta) = \int_\Omega \frac{k(\eta)}{2\epsilon}\left(\epsilon\Delta\phi + \left(\frac{1}{\epsilon}\phi + c_0(\eta)\sqrt{2}\right)(1-\phi^2)\right)^2 dx
$$

线张力能 (Eq. 4):
$$
L(\phi,\eta) = \int_\Omega \delta\left(\frac{\xi}{2}|\nabla\eta|^2 + \frac{1}{4\xi}(\eta^2-1)^2\right)\left(\frac{\epsilon}{2}|\nabla\phi|^2 + \frac{1}{4\epsilon}(\phi^2-1)^2\right)dx
$$

## 约束泛函

基于 ${a_0, a_d, v_0}$ 的三条等式约束

总面积 (Eq. 5):
$$
A(\phi) = \int_\Omega \left(\frac{\epsilon}{2}|\nabla\phi|^2 + \frac{1}{4\epsilon}(\phi^2-1)^2\right)dx = a_0
$$

体积差 (Eq. 6):
$$
V(\phi) = \int_\Omega \phi\,dx = v_d
$$

面积差 (Eq. 7):
$$
D(\phi,\eta) = \int_\Omega \tanh\!\frac{\eta}{\xi}\left(\frac{\epsilon}{2}|\nabla\phi|^2 + \frac{1}{4\epsilon}(\phi^2-1)^2\right)dx = a_d
$$

### 理论收敛情况

$$
E(\phi,\eta) \to \frac{4\sqrt{2}}{3}\sum_{j=1}^{2}\int_{\Gamma_j} k_j(H-c_j)^2\,ds
$$

$$
V(\phi)\to|\Omega_i|-|\Omega_e|,\qquad A(\phi)\to\frac{2\sqrt{2}}{3}|\Gamma|
$$

$$
L(\phi,\eta)\to\frac{8}{9}\int_{\gamma_0}\delta\,dl,\qquad D(\phi,\eta)\to\frac{2\sqrt{2}}{3}\big(|\Gamma_1|-|\Gamma_2|\big)
$$

## 正则泛函

正交性 (Eq. 13): 约束两Function的正交性，作为惩罚想。

$$
N(\phi,\eta) = \int_\Omega \frac{\epsilon}{2}|\nabla\phi\cdot\nabla\eta|^2\,dx
$$

$\eta$ 剖面 (Eq. 14): 正则项，描述$\eta$ 的分布和$\tanh$的分布一致性。

$$
P(\eta) = \int_\Omega \left(\frac{\xi}{2}|\nabla\eta|^2 - \frac{1}{4\xi}(\eta^2-1)^2\right)^2 dx
$$

## 约束最小化问题

$$\min_{\phi,\eta}\ \ \mathcal{E}(\phi,\eta) = E(\phi,\eta) + L(\phi,\eta)$$

$$\text{s.t.}\quad V(\phi)=v_d,\quad A(\phi)=a_0,\quad D(\phi,\eta)=a_d$$

$$N(\phi,\eta)\approx 0,\qquad P(\eta)\approx 0$$

### Lagrange Form

$$
\mathcal{E}_M(\phi,\eta) = E(\phi,\eta) + L(\phi,\eta) + \tfrac{1}{2}M_1\big(V(\phi)-v_d\big)^2+\tfrac{1}{2}M_2\big(A(\phi)-a_0\big)^2 \ + \tfrac{1}{2}M_3\big(D(\phi,\eta)-a_d\big)^2
+ \tfrac{1}{2}M_4\big(N(\phi,\eta)\big)^2
+ \tfrac{1}{2}M_5\big(P(\eta)\big)^2
$$

$$
(\phi^*,\eta^*) = \arg\min_{\phi,\eta}\ \mathcal{E}_M(\phi,\eta)
$$

