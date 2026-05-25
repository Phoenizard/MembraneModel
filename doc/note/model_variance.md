# 模型与变分导数

> 两组分囊泡相场模型的能量 $E_M$，及其变分导数 $\delta E_M/\delta\phi,\ \delta E_M/\delta\eta$ 的**完整推导**。
> 这是梯度流 $\phi_t=-\delta E_M/\delta\phi,\ \eta_t=-\delta E_M/\delta\eta$ 的右端，**所有求解算法共享**；
> 如何把梯度流积分到稳态见各算法文档（如 [`quasi_newton.md`](quasi_newton.md)）。
> 代码：[`model/energy.py`](../../model/energy.py)、[`model/fields.py`](../../model/fields.py)。
>
> 简化设定：$c_1=c_2=0 \Rightarrow c_0(\eta)\equiv 0$，$\xi=\epsilon$，周期边界。$(f,g)=\int_\Omega fg\,dx$。

## 记号

$$
g(\phi)=\tfrac1\epsilon\phi(1-\phi^2),\quad
g'(\phi)=\tfrac1\epsilon(1-3\phi^2),\quad
m(\phi)=\epsilon\Delta\phi+g(\phi),
$$

$$
W(\phi)=\tfrac\epsilon2|\nabla\phi|^2+\tfrac1{4\epsilon}(\phi^2-1)^2,\qquad
k(\eta)=k+c\,\tanh(\eta/\xi),\quad k'(\eta)=\tfrac c\xi\,\mathrm{sech}^2(\eta/\xi).
$$

代码：`g, gp, W_density, curvature`（即 $m$）在 `fields.py`；`k_eta, kp_eta, tanh_eta, sech2_eta` 在 `energy.py`。

## 1. 能量泛函

弯曲能 $E$ 与线张力能 $L$：

$$
E=\int_\Omega\frac{k(\eta)}{2\epsilon}\,m(\phi)^2\,dx,
$$

$$
L=\int_\Omega \delta\Big(\tfrac\xi2|\nabla\eta|^2+\tfrac1{4\xi}(\eta^2-1)^2\Big)\,W(\phi)\,dx.
$$

约束泛函——体积 $V$、总面积 $A$、面积差 $D$：

$$
V=\int_\Omega\phi\,dx,\quad
A=\int_\Omega W(\phi)\,dx,\quad
D=\int_\Omega\tanh(\eta/\xi)\,W(\phi)\,dx.
$$

正则泛函——正交项 $N$、$\eta$ 剖面项 $P$：

$$
N=\int_\Omega\tfrac\epsilon2|\nabla\phi\cdot\nabla\eta|^2\,dx,\quad
P=\int_\Omega\Pi(\eta)^2\,dx,\quad
\Pi(\eta):=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2.
$$

罚函数总能量（约束 $V\to v_d,\ A\to a_0,\ D\to a_d$ 以罚项施加，$N,P$ 为正则）：

$$
\boxed{\ \mathcal E_M=E+L
+\tfrac{M_1}2(V-v_d)^2+\tfrac{M_2}2(A-a_0)^2+\tfrac{M_3}2(D-a_d)^2
+\tfrac{M_4}2N^2+\tfrac{M_5}2P^2\ }
$$

代码：`E_bending, E_line, volume, area, area_diff, N_func, P_func, E_total, energy_breakdown`，均在 `energy.py`。

## 2. 变分导数

### 2.0 方法（三步）

泛函 $F$ 在 $\phi$ 处沿方向 $u$ 的方向导数

$$
dF(\phi)[u]:=\frac{d}{d\varepsilon}F(\phi+\varepsilon u)\Big|_{\varepsilon=0}.
$$

若方向导数在 $L^2$ 中连续，由 Riesz 定理可整理成

$$
dF=\int_\Omega(\,\cdot\,)\,u\,dx,
$$

且令 $(\,\cdot\,)=:\dfrac{\delta F}{\delta\phi}$ 为变分导数。固定三步：

1. **代入** $\phi+\varepsilon u$，对 $\varepsilon$ 求导取 $\varepsilon=0$；
2. **分部积分**，把 $u$ 上的导数（$\nabla u,\Delta u$）转移走（周期边界，边界项消失）；
3. **读系数**，整理成 $\int_\Omega(\cdot)\,u\,dx$。

罚项 $\tfrac12 M(G-g_*)^2$ 用外层链式 $d=M(G-g_*)\,dG[u]$，对内层泛函 $G$ 再走三步。全程直接对泛函求导，不引入辅助变量。

### 2.1 对 φ 变分

此时 $\eta$ 冻结，记

$$
\Phi_\eta:=\delta\big(\tfrac\xi2|\nabla\eta|^2+\tfrac1{4\xi}(\eta^2-1)^2\big).
$$

#### 弯曲能 $E$

方向导数

$$
dE[u]=\int_\Omega\tfrac{k(\eta)}\epsilon\,m\,\big(\epsilon\Delta u+g'(\phi)u\big)\,dx.
$$

含 $\Delta u$ 的项两次分部积分（$\int\Psi\,\Delta u=\int(\Delta\Psi)\,u$），读系数得

$$
\frac{\delta E}{\delta\phi}=\Delta\big(k(\eta)\,m\big)+\tfrac{k(\eta)}\epsilon\,g'(\phi)\,m .
$$

#### 线张力 $L$

由 $L=\int_\Omega\Phi_\eta\,W(\phi)\,dx$，方向导数

$$
dW[u]=\epsilon\,\nabla\phi\cdot\nabla u+\tfrac1\epsilon\phi(\phi^2-1)\,u.
$$

含 $\nabla u$ 的项分部积分，读系数得

$$
\frac{\delta L}{\delta\phi}=-\epsilon\,\nabla\!\cdot\!\big(\Phi_\eta\nabla\phi\big)+\tfrac1\epsilon\phi(\phi^2-1)\,\Phi_\eta .
$$

#### 约束 $V,A,D$

内层导数同 $dW$ 的处理（权重不同，$\nabla u$ 项分部积分）：

$$
\frac{\delta V}{\delta\phi}=1,\quad
\frac{\delta A}{\delta\phi}=-\epsilon\Delta\phi+\tfrac1\epsilon\phi(\phi^2-1),\quad
\frac{\delta D}{\delta\phi}=-\epsilon\,\nabla\!\cdot\!\big(\tanh(\eta/\xi)\nabla\phi\big)+\tfrac1\epsilon\phi(\phi^2-1)\tanh(\eta/\xi).
$$

#### 正交项 $N$

方向导数

$$
dN[u]=\int_\Omega\epsilon\,(\nabla\phi\cdot\nabla\eta)(\nabla u\cdot\nabla\eta)\,dx.
$$

含 $\nabla u$ 的项分部积分，读系数得

$$
\frac{\delta N}{\delta\phi}=-\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\nabla\eta\big).
$$

#### 合计

罚项外层乘 $M_i(G-g_*)$，合并各项：

$$
\boxed{
\begin{aligned}
\frac{\delta\mathcal E_M}{\delta\phi}
=\;&\Delta\big(k(\eta)m\big)+\tfrac{k(\eta)}\epsilon g'(\phi)m
+\Big[-\epsilon\nabla\!\cdot\!(\Phi_\eta\nabla\phi)+\tfrac1\epsilon\phi(\phi^2-1)\Phi_\eta\Big]\\
&+M_1(V-v_d)
+M_2(A-a_0)\big(-\epsilon\Delta\phi+\tfrac1\epsilon\phi(\phi^2-1)\big)\\
&+M_3(D-a_d)\big(-\epsilon\nabla\!\cdot\!(\tanh(\eta/\xi)\nabla\phi)+\tfrac1\epsilon\phi(\phi^2-1)\tanh(\eta/\xi)\big)\\
&-M_4\,N\,\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\nabla\eta\big)
\end{aligned}}
$$

### 2.2 对 η 变分

此时 $\phi$ 冻结，故 $\Delta\phi,\ g(\phi),\ m,\ W(\phi)$ 均为已知系数。

#### 弯曲能 $E$

仅 $k(\eta)$ 含 $\eta$，无 $u$ 的导数，直接读系数得

$$
\frac{\delta E}{\delta\eta}=\frac{k'(\eta)}{2\epsilon}\,m^2 .
$$

#### 线张力 $L$

$W$ 作权重，方向导数

$$
d(\cdots)[u]=\xi\,\nabla\eta\cdot\nabla u+\tfrac1\xi\eta(\eta^2-1)\,u.
$$

含 $\nabla u$ 的项分部积分，读系数得

$$
\frac{\delta L}{\delta\eta}=-\delta\xi\,\nabla\!\cdot\!\big(W(\phi)\nabla\eta\big)+\delta\tfrac1\xi W(\phi)\,\eta(\eta^2-1).
$$

#### 面积差 $D$

仅 $\tanh(\eta/\xi)$ 含 $\eta$，无导数，直接读系数得

$$
\frac{\delta D}{\delta\eta}=\tfrac1\xi W(\phi)\,\mathrm{sech}^2(\eta/\xi).
$$

#### 正交项 $N$

方向导数

$$
dN[u]=\int_\Omega\epsilon\,(\nabla\phi\cdot\nabla\eta)(\nabla\phi\cdot\nabla u)\,dx.
$$

含 $\nabla u$ 的项分部积分，读系数得

$$
\frac{\delta N}{\delta\eta}=-\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\nabla\phi\big).
$$

#### η 剖面项 $P$

方向导数

$$
dP[u]=\int_\Omega 2\Pi(\eta)\big(\xi\,\nabla\eta\cdot\nabla u-\tfrac1\xi\eta(\eta^2-1)\,u\big)\,dx.
$$

含 $\nabla u$ 的项分部积分，读系数得

$$
\frac{\delta P}{\delta\eta}=-2\xi\,\nabla\!\cdot\!\big(\Pi\nabla\eta\big)-\tfrac2\xi\Pi\,\eta(\eta^2-1).
$$

#### 合计

$$
\boxed{
\begin{aligned}
\frac{\delta\mathcal E_M}{\delta\eta}
=\;&\frac{k'(\eta)}{2\epsilon}m^2
+\Big[-\delta\xi\nabla\!\cdot\!(W\nabla\eta)+\delta\tfrac1\xi W\,\eta(\eta^2-1)\Big]\\
&+M_3(D-a_d)\tfrac1\xi W\,\mathrm{sech}^2(\eta/\xi)
-M_4\,N\,\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\nabla\phi\big)\\
&+M_5\,P\big(-2\xi\nabla\!\cdot\!(\Pi\nabla\eta)-\tfrac2\xi\Pi\,\eta(\eta^2-1)\big)
\end{aligned}}
$$
