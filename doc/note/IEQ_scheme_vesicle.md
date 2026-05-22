# 两组分囊泡相场模型 — IEQ 一阶解耦数值方案

求解目标：梯度流演化至稳态，得到 $\arg\min_{\phi,\eta}\mathcal{E}_M(\phi,\eta)$。

时间离散：一阶 IEQ（Invariant Energy Quadratization）。空间离散：Fourier 谱（FFT），周期边界，$64^3$ 网格。场更新：解耦交替。

---

## 0. 简化条件与记号

本文档在以下简化下推导：

$$c_1=c_2=0\ \Rightarrow\ c_0(\eta)\equiv 0,\qquad \epsilon=\xi.$$

记 $L^2$ 内积 $(f,g)=\int_\Omega fg\,dx$，范数 $\|f\|^2=(f,f)$。所有场周期。

引入两个反复出现的非线性量：

$$
g(\phi):=\Big(\tfrac{1}{\epsilon}\phi\Big)(1-\phi^2),\qquad
W(\phi):=\tfrac{\epsilon}{2}|\nabla\phi|^2+\tfrac{1}{4\epsilon}(\phi^2-1)^2 .
$$

$g$ 来自弯曲能（$c_0\equiv0$ 后括号内的非线性部分），$W$ 是 Cahn–Hilliard 型面积密度。物理系数：

$$k(\eta)=k+c\,\tanh(\eta/\xi),\qquad k,c\ \text{为符号参数}.$$

---

## 1. 能量泛函（简化后）

**弯曲能**

$$
E(\phi,\eta)=\int_\Omega \frac{k(\eta)}{2\epsilon}\,\big(\epsilon\Delta\phi+g(\phi)\big)^2\,dx .
$$

**线张力能**

$$
L(\phi,\eta)=\int_\Omega \delta\,\Big(\tfrac{\xi}{2}|\nabla\eta|^2+\tfrac{1}{4\xi}(\eta^2-1)^2\Big)\,W(\phi)\,dx .
$$

**约束泛函**

$$
A(\phi)=\int_\Omega W(\phi)\,dx,\qquad
V(\phi)=\int_\Omega \phi\,dx,\qquad
D(\phi,\eta)=\int_\Omega \tanh(\eta/\xi)\,W(\phi)\,dx .
$$

**正则泛函**

$$
N(\phi,\eta)=\int_\Omega \frac{\epsilon}{2}\,|\nabla\phi\cdot\nabla\eta|^2\,dx,\qquad
P(\eta)=\int_\Omega \Big(\tfrac{\xi}{2}|\nabla\eta|^2-\tfrac{1}{4\xi}(\eta^2-1)^2\Big)^2 dx .
$$

**罚函数总能量**

$$
\mathcal{E}_M(\phi,\eta)=E+L
+\tfrac{1}{2}M_1(V-v_d)^2
+\tfrac{1}{2}M_2(A-a_0)^2
+\tfrac{1}{2}M_3(D-a_d)^2
+\tfrac{1}{2}M_4 N^2
+\tfrac{1}{2}M_5 P^2 .
$$

---

## 2. IEQ 辅助变量

对每个非线性势引入辅助变量，使能量成为辅助变量的**二次型**。

### 2.1 弯曲能 — 变量 $q$

$$
\boxed{\,q(\phi,\eta):=\sqrt{\tfrac{k(\eta)}{\epsilon}}\;\big(\epsilon\Delta\phi+g(\phi)\big)\,}
\qquad\Longrightarrow\qquad
E=\tfrac12\|q\|^2 .
$$

要求 $k(\eta)>0$（取 $k>|c|$，保证根式有定义）。

### 2.2 线张力能 — 变量 $r$

线张力能是两个非负密度之积。对 $\phi$ 侧引入辅助变量。注意 $W(\phi)$ 在 bulk 区（$|\phi|\to1$）趋于 $0$，若直接取 $r=\sqrt{2W}$，后续时间格式中 $1/r$ 会在大片网格点除零。故引入正则常数 $C_0>0$：

$$
\boxed{\,r(\phi):=\sqrt{2\,W(\phi)+C_0}\,}\qquad\Longrightarrow\qquad W(\phi)=\tfrac12\big(r^2-C_0\big).
$$

$C_0$ 为常数，对变分导数无贡献（常数变分为零），仅在能量值上平移，不影响梯度流与稳态；$C_0$ 取值列入待标定量。则

$$
L=\int_\Omega \frac{\delta}{2}\,\Big(\tfrac{\xi}{2}|\nabla\eta|^2+\tfrac{1}{4\xi}(\eta^2-1)^2\Big)\,\big(r^2-C_0\big)\,dx .
$$

同一个 $r$ 同时服务于 $A,D$：

$$
A=\tfrac12\int_\Omega\big(r^2-C_0\big)\,dx,\qquad D=\tfrac12\int_\Omega\tanh(\eta/\xi)\,\big(r^2-C_0\big)\,dx .
$$

### 2.3 正交正则项 $N$ — 变量 $s$

$$
\boxed{\,\mathbf{s}(\phi,\eta):=\sqrt{\epsilon}\,(\nabla\phi\cdot\nabla\eta)\,}\qquad\Longrightarrow\qquad N=\tfrac12\|\mathbf s\|^2 .
$$

（$\mathbf s$ 为标量场；$\nabla\phi\cdot\nabla\eta$ 是点积。）

### 2.4 $\eta$ 剖面正则项 $P$ — 变量 $p$

被平方的括号为差，对 $\eta$ 既非凸亦无下界，故**不**对括号内取根，而是直接令

$$
\boxed{\,p(\eta):=\tfrac{\xi}{2}|\nabla\eta|^2-\tfrac{1}{4\xi}(\eta^2-1)^2\,}\qquad\Longrightarrow\qquad P=\|p\|^2 .
$$

$p$ 可正可负，但 $P=\|p\|^2$ 仍是 $p$ 的二次型，满足 IEQ 所需结构。

### 2.5 二次化总能量

$$
\mathcal{E}_M=\tfrac12\|q\|^2+L(r,\eta)
+\tfrac{M_1}{2}(V-v_d)^2+\tfrac{M_2}{2}(A-a_0)^2+\tfrac{M_3}{2}(D-a_d)^2
+\tfrac{M_4}{2}N^2+\tfrac{M_5}{2}P^2 ,
$$

其中 $E,N,P$ 已为对应辅助变量的二次型；$A,L,D$ 经 $W(\phi)=\tfrac12(r^2-C_0)$ 为 $r$ 的二次型（含常数项 $-\tfrac12C_0$，对变分无贡献）。

---

## 3. 辅助变量的演化方程

辅助变量随 $\phi,\eta$ 演化，由其定义对时间求导（链式法则）得到。下列为**恒等式**，离散时用于消去辅助变量。

### 3.1 $q$

记 $g'(\phi)=\dfrac{1}{\epsilon}(1-3\phi^2)$，$\;a(\eta):=\sqrt{k(\eta)/\epsilon}$，$\;a'(\eta)=\dfrac{k'(\eta)}{2\epsilon\,a(\eta)}$，$\;k'(\eta)=\dfrac{c}{\xi}\,\mathrm{sech}^2(\eta/\xi)$。

$$
q_t=a(\eta)\big(\epsilon\Delta\phi_t+g'(\phi)\phi_t\big)+a'(\eta)\big(\epsilon\Delta\phi+g(\phi)\big)\,\eta_t .
$$

### 3.2 $r$

$r=\sqrt{2W(\phi)+C_0}$。对时间求导，链式法则（先 $\sqrt{\ }$，再 $W$ 逐点求导；$C_0$ 为常数，$\partial_t C_0=0$）：

$$
r_t=\frac{1}{\sqrt{2W(\phi)+C_0}}\,W_t,\qquad
W_t=\epsilon\,\nabla\phi\cdot\nabla\phi_t+\tfrac1\epsilon\phi(\phi^2-1)\,\phi_t ,
$$

即（注意 $\sqrt{2W(\phi)+C_0}=r$）

$$
r_t=\frac{1}{r}\Big(\epsilon\,\nabla\phi\cdot\nabla\phi_t+\tfrac1\epsilon\phi(\phi^2-1)\,\phi_t\Big).
$$

这是严格的逐点恒等式，第一项含 $\nabla\phi_t$。离散格式中 $\phi_t$ 上的梯度由分部积分转走，$1/r$ 取旧层值 $1/r^n$ 冻结为已知系数；$C_0>0$ 保证 $r^n$ 处处有正下界 $r^n\ge\sqrt{C_0}$，$1/r^n$ 不奇异（详见时间格式章节）。

### 3.3 $\mathbf s$

$$
\mathbf s_t=\sqrt\epsilon\,\big(\nabla\phi_t\cdot\nabla\eta+\nabla\phi\cdot\nabla\eta_t\big).
$$

### 3.4 $p$

$p=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2$。对时间求导：

$$
p_t=\xi\,\nabla\eta\cdot\nabla\eta_t-\tfrac1\xi\eta(\eta^2-1)\,\eta_t .
$$

---

## 4. 梯度流与变分导数

求稳态采用 $L^2$ 梯度流：

$$
\phi_t=-\frac{\delta\mathcal{E}_M}{\delta\phi},\qquad
\eta_t=-\frac{\delta\mathcal{E}_M}{\delta\eta}.
$$

### 4.0 变分导数的定义与计算流程

对泛函 $F$，其在 $\phi$ 处沿方向 $u$ 的方向导数定义为

$$
\mathrm dF(\phi)[u]:=\left.\frac{d}{d\varepsilon}\,F(\phi+\varepsilon u)\right|_{\varepsilon=0}.
$$

若该方向导数可整理成 $L^2$ 内积形式 $\mathrm dF(\phi)[u]=\int_\Omega(\,\cdot\,)\,u\,dx$，则被 $u$ 乘的那个函数称为变分导数 $\dfrac{\delta F}{\delta\phi}$。**计算流程固定为三步**：

1. 代入 $\phi+\varepsilon u$，对 $\varepsilon$ 求导取 $\varepsilon=0$；
2. 分部积分，把作用在 $u$ 上的一切导数（$\nabla u,\Delta u$）转移走（周期边界，边界项消失）；
3. 整理成 $\int_\Omega(\,\cdot\,)u\,dx$，读出 $u$ 的系数。

本节所有变分导数均直接对**泛函**（$E,L,A,D,V,N,P$ 及罚项，输出为数）执行此三步。辅助变量 $q,r,\mathbf s,p$ 是函数值映射（输出为函数），不是泛函，不对它们单独定义变分导数；它们只在第 4 节末与离散格式中作为中间量出现。

下面 $\mathrm dF(\phi)[u]$ 中 $u$ 为 $\phi$ 方向的扰动，$\mathrm dF(\eta)[u]$ 中 $u$ 为 $\eta$ 方向的扰动。

---

### 4.1 对 $\phi$ 的变分导数

求 $\delta\phi$ 变分时 $\eta$ 冻结，含 $\eta$ 的量均为已知系数。本节将冻结的 $\eta$ 系数封装为

$$
\Phi_\eta:=\delta\Big(\tfrac\xi2|\nabla\eta|^2+\tfrac1{4\xi}(\eta^2-1)^2\Big)\qquad(\text{已知系数，仅 4.1 节有效}).
$$

注意：$\Phi_\eta$ 仅在本节有效。4.2 节对 $\eta$ 变分时 $\eta$ 不冻结，该表达式须参与求导，届时保留完整写法、不用 $\Phi_\eta$。

#### 4.1.1 弯曲能 $E$

$$
E(\phi,\eta)=\int_\Omega\frac{k(\eta)}{2\epsilon}\big(\epsilon\Delta\phi+g(\phi)\big)^2\,dx .
$$

**第一步：方向导数。**

$$
\mathrm dE(\phi)[u]
=\int_\Omega\frac{k(\eta)}{2\epsilon}\cdot 2\big(\epsilon\Delta\phi+g(\phi)\big)
\left.\frac{d}{d\varepsilon}\big(\epsilon\Delta(\phi+\varepsilon u)+g(\phi+\varepsilon u)\big)\right|_{\varepsilon=0}dx .
$$

其中

$$
\left.\frac{d}{d\varepsilon}\big(\epsilon\Delta(\phi+\varepsilon u)+g(\phi+\varepsilon u)\big)\right|_{\varepsilon=0}
=\epsilon\Delta u+g'(\phi)\,u,\qquad g'(\phi)=\tfrac1\epsilon(1-3\phi^2),
$$

故

$$
\mathrm dE(\phi)[u]=\int_\Omega\frac{k(\eta)}{\epsilon}\big(\epsilon\Delta\phi+g(\phi)\big)\big(\epsilon\Delta u+g'(\phi)u\big)\,dx .
$$

**第二步：分部积分。** 含 $\Delta u$ 的部分，记被乘因子为 $\Psi:=\dfrac{k(\eta)}{\epsilon}\big(\epsilon\Delta\phi+g(\phi)\big)\cdot\epsilon$，分部积分两次（周期边界，边界项消失）：

$$
\int_\Omega\Psi\,\Delta u\,dx=-\int_\Omega\nabla\Psi\cdot\nabla u\,dx=\int_\Omega(\Delta\Psi)\,u\,dx .
$$

含 $g'(\phi)u$ 的部分已不含 $u$ 的导数。

**第三步：读系数。**

$$
\boxed{\ \frac{\delta E}{\delta\phi}
=\epsilon\,\Delta\!\Big(k(\eta)\big(\epsilon\Delta\phi+g(\phi)\big)\Big)\Big/\epsilon
+\frac{k(\eta)}{\epsilon}\,g'(\phi)\big(\epsilon\Delta\phi+g(\phi)\big)\ }
$$

整理（提出 $\epsilon$）：

$$
\frac{\delta E}{\delta\phi}
=\Delta\!\Big(k(\eta)\big(\epsilon\Delta\phi+g(\phi)\big)\Big)
+\frac{k(\eta)}{\epsilon}\,g'(\phi)\big(\epsilon\Delta\phi+g(\phi)\big).
$$

**用辅助变量 $a,q$ 改写。** 由 2.1 节 $q=a(\eta)\big(\epsilon\Delta\phi+g(\phi)\big)$、3.1 节 $a(\eta)=\sqrt{k(\eta)/\epsilon}$，有 $a(\eta)^2=k(\eta)/\epsilon$，故

$$
k(\eta)\big(\epsilon\Delta\phi+g(\phi)\big)=\epsilon\,a(\eta)^2\big(\epsilon\Delta\phi+g(\phi)\big)=\epsilon\,a(\eta)\,q,
$$

$$
\frac{k(\eta)}{\epsilon}\big(\epsilon\Delta\phi+g(\phi)\big)=a(\eta)^2\big(\epsilon\Delta\phi+g(\phi)\big)=a(\eta)\,q .
$$

代入得等价形式：

$$
\boxed{\ \frac{\delta E}{\delta\phi}=\epsilon\,\Delta\big(a(\eta)\,q\big)+g'(\phi)\,a(\eta)\,q\ }
$$

#### 4.1.2 线张力能 $L$

$$
L(\phi,\eta)=\int_\Omega\Phi_\eta\,W(\phi)\,dx .
$$

**第一步：方向导数。**

$$
\mathrm dL(\phi)[u]
=\int_\Omega\Phi_\eta\left.\frac{d}{d\varepsilon}W(\phi+\varepsilon u)\right|_{\varepsilon=0}dx .
$$

其中

$$
\left.\frac{d}{d\varepsilon}W(\phi+\varepsilon u)\right|_{\varepsilon=0}
=\epsilon\,\nabla\phi\cdot\nabla u+\tfrac1\epsilon\phi(\phi^2-1)\,u .
$$

**第二步：分部积分。** 含 $\nabla u$ 的项（周期边界，边界项消失）：

$$
\int_\Omega\Phi_\eta\,\epsilon\,\nabla\phi\cdot\nabla u\,dx
=-\int_\Omega u\;\epsilon\,\nabla\!\cdot\!\big(\Phi_\eta\,\nabla\phi\big)\,dx .
$$

**第三步：读系数。**

$$
\boxed{\ \frac{\delta L}{\delta\phi}
=-\epsilon\,\nabla\!\cdot\!\big(\Phi_\eta\,\nabla\phi\big)
+\tfrac1\epsilon\phi(\phi^2-1)\,\Phi_\eta\ }
$$

#### 4.1.3 面积约束罚项 $\tfrac{M_2}2(A-a_0)^2$

$A(\phi)=\int_\Omega W(\phi)\,dx$。外层是平方，内层 $A$ 经三步流程求 $\delta A/\delta\phi$。

**外层（普通链式法则）。**

$$
\mathrm d\Big[\tfrac{M_2}2(A-a_0)^2\Big](\phi)[u]
=M_2\,(A-a_0)\;\mathrm dA(\phi)[u].
$$

$(A-a_0)$ 是标量数。

**内层 $\mathrm dA(\phi)[u]$。** $A=\int W$，三步流程（与 $L$ 中 $W$ 部分相同，无 $\eta$ 权重）：

$$
\mathrm dA(\phi)[u]=\int_\Omega\big(\epsilon\,\nabla\phi\cdot\nabla u+\tfrac1\epsilon\phi(\phi^2-1)u\big)dx
=\int_\Omega\big(-\epsilon\Delta\phi+\tfrac1\epsilon\phi(\phi^2-1)\big)u\,dx,
$$

故

$$
\frac{\delta A}{\delta\phi}=-\epsilon\Delta\phi+\tfrac1\epsilon\phi(\phi^2-1).
$$

**合并。**

$$
\boxed{\ \frac{\delta}{\delta\phi}\Big[\tfrac{M_2}2(A-a_0)^2\Big]
=M_2(A-a_0)\Big(-\epsilon\Delta\phi+\tfrac1\epsilon\phi(\phi^2-1)\Big)\ }
$$

#### 4.1.4 面积差约束罚项 $\tfrac{M_3}2(D-a_d)^2$

$D(\phi,\eta)=\int_\Omega\tanh(\eta/\xi)\,W(\phi)\,dx$，$\tanh(\eta/\xi)$ 冻结为已知系数。

**外层。**

$$
\mathrm d\Big[\tfrac{M_3}2(D-a_d)^2\Big](\phi)[u]=M_3(D-a_d)\;\mathrm dD(\phi)[u].
$$

**内层 $\mathrm dD(\phi)[u]$。** 三步流程，$\tanh(\eta/\xi)$ 作权重：

$$
\mathrm dD(\phi)[u]=\int_\Omega\tanh(\eta/\xi)\big(\epsilon\nabla\phi\cdot\nabla u+\tfrac1\epsilon\phi(\phi^2-1)u\big)dx,
$$

含 $\nabla u$ 项分部积分：

$$
\frac{\delta D}{\delta\phi}=-\epsilon\,\nabla\!\cdot\!\big(\tanh(\eta/\xi)\nabla\phi\big)+\tfrac1\epsilon\phi(\phi^2-1)\tanh(\eta/\xi).
$$

**合并。**

$$
\boxed{\ \frac{\delta}{\delta\phi}\Big[\tfrac{M_3}2(D-a_d)^2\Big]
=M_3(D-a_d)\Big(-\epsilon\,\nabla\!\cdot\!\big(\tanh(\eta/\xi)\nabla\phi\big)+\tfrac1\epsilon\phi(\phi^2-1)\tanh(\eta/\xi)\Big)\ }
$$

#### 4.1.5 体积约束罚项 $\tfrac{M_1}2(V-v_d)^2$

$V(\phi)=\int_\Omega\phi\,dx$ 是线性泛函。

$$
\mathrm dV(\phi)[u]=\int_\Omega u\,dx\ \Rightarrow\ \frac{\delta V}{\delta\phi}=1 .
$$

外层链式：

$$
\boxed{\ \frac{\delta}{\delta\phi}\Big[\tfrac{M_1}2(V-v_d)^2\Big]=M_1(V-v_d)\ }
$$

#### 4.1.6 正交正则项罚项 $\tfrac{M_4}2N^2$

$N(\phi,\eta)=\int_\Omega\tfrac\epsilon2|\nabla\phi\cdot\nabla\eta|^2\,dx$，$\eta$ 冻结。

**外层。**

$$
\mathrm d\Big[\tfrac{M_4}2N^2\Big](\phi)[u]=M_4\,N\;\mathrm dN(\phi)[u].
$$

**内层 $\mathrm dN(\phi)[u]$。** 第一步：

$$
\mathrm dN(\phi)[u]=\int_\Omega\frac\epsilon2\cdot 2(\nabla\phi\cdot\nabla\eta)
\left.\frac{d}{d\varepsilon}\big(\nabla(\phi+\varepsilon u)\cdot\nabla\eta\big)\right|_{\varepsilon=0}dx
=\int_\Omega\epsilon\,(\nabla\phi\cdot\nabla\eta)\,(\nabla u\cdot\nabla\eta)\,dx .
$$

第二步：含 $\nabla u$，分部积分（周期边界）：

$$
\int_\Omega\epsilon\,(\nabla\phi\cdot\nabla\eta)\,(\nabla u\cdot\nabla\eta)\,dx
=-\int_\Omega u\;\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\,\nabla\eta\big)\,dx .
$$

第三步：

$$
\frac{\delta N}{\delta\phi}=-\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\,\nabla\eta\big).
$$

**合并。**

$$
\boxed{\ \frac{\delta}{\delta\phi}\Big[\tfrac{M_4}2N^2\Big]
=-M_4\,N\,\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\,\nabla\eta\big)\ }
$$

#### 4.1.7 合计

$$
\boxed{
\begin{aligned}
\frac{\delta\mathcal E_M}{\delta\phi}
=\;&\epsilon\,\Delta\big(a(\eta)\,q\big)+g'(\phi)\,a(\eta)\,q\\[4pt]
&-\epsilon\,\nabla\!\cdot\!\big(\Phi_\eta\,\nabla\phi\big)
+\tfrac1\epsilon\phi(\phi^2-1)\,\Phi_\eta\\[4pt]
&+M_1(V-v_d)\\[4pt]
&+M_2(A-a_0)\Big(-\epsilon\Delta\phi+\tfrac1\epsilon\phi(\phi^2-1)\Big)\\[4pt]
&+M_3(D-a_d)\Big(-\epsilon\,\nabla\!\cdot\!\big(\tanh(\eta/\xi)\nabla\phi\big)+\tfrac1\epsilon\phi(\phi^2-1)\tanh(\eta/\xi)\Big)\\[4pt]
&-M_4\,N\,\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\,\nabla\eta\big)
\end{aligned}}
$$

其中弯曲项 $q=a(\eta)\big(\epsilon\Delta\phi+g(\phi)\big)$，$a(\eta)=\sqrt{k(\eta)/\epsilon}$；$\Phi_\eta=\delta\big(\tfrac\xi2|\nabla\eta|^2+\tfrac1{4\xi}(\eta^2-1)^2\big)$ 为冻结系数。

---

### 4.2 对 $\eta$ 的变分导数

求 $\delta\eta$ 变分时 $\phi$ 冻结，含 $\phi$ 的量（$\Delta\phi,g(\phi),W(\phi)$）均为已知系数。

#### 4.2.1 弯曲能 $E$

$\eta$ 仅经 $k(\eta)$ 进入。$\epsilon\Delta\phi+g(\phi)$ 冻结。

**第一步：方向导数。** 仅 $k(\eta)$ 含 $\eta$：

$$
\mathrm dE(\eta)[u]
=\int_\Omega\frac{1}{2\epsilon}\big(\epsilon\Delta\phi+g(\phi)\big)^2
\left.\frac{d}{d\varepsilon}k(\eta+\varepsilon u)\right|_{\varepsilon=0}dx,
$$

$$
\left.\frac{d}{d\varepsilon}k(\eta+\varepsilon u)\right|_{\varepsilon=0}=k'(\eta)\,u,
\qquad k'(\eta)=\tfrac c\xi\,\mathrm{sech}^2(\eta/\xi).
$$

**第二步：无 $u$ 的导数，不需分部积分。**

**第三步：读系数。**

$$
\boxed{\ \frac{\delta E}{\delta\eta}=\frac{k'(\eta)}{2\epsilon}\big(\epsilon\Delta\phi+g(\phi)\big)^2\ }
$$

**用辅助变量 $q$ 改写。** 由 $q=a(\eta)\big(\epsilon\Delta\phi+g(\phi)\big)$、$a(\eta)^2=k(\eta)/\epsilon$，有 $\big(\epsilon\Delta\phi+g(\phi)\big)^2=q^2/a(\eta)^2=\epsilon\,q^2/k(\eta)$，代入得等价形式：

$$
\boxed{\ \frac{\delta E}{\delta\eta}=\frac{k'(\eta)}{2k(\eta)}\,q^2\ }
$$

#### 4.2.2 线张力能 $L$

$$
L(\phi,\eta)=\int_\Omega\delta\Big(\tfrac\xi2|\nabla\eta|^2+\tfrac1{4\xi}(\eta^2-1)^2\Big)W(\phi)\,dx,
$$

$W(\phi)$ 冻结为已知系数。

**第一步：方向导数。**

$$
\mathrm dL(\eta)[u]
=\int_\Omega\delta\,W(\phi)\left.\frac{d}{d\varepsilon}
\Big(\tfrac\xi2|\nabla\eta+\varepsilon\nabla u|^2+\tfrac1{4\xi}\big((\eta+\varepsilon u)^2-1\big)^2\Big)\right|_{\varepsilon=0}dx,
$$

$$
\left.\frac{d}{d\varepsilon}(\cdots)\right|_{\varepsilon=0}
=\xi\,\nabla\eta\cdot\nabla u+\tfrac1\xi\eta(\eta^2-1)\,u .
$$

**第二步：分部积分。** 含 $\nabla u$ 项（周期边界）：

$$
\int_\Omega\delta\,W(\phi)\,\xi\,\nabla\eta\cdot\nabla u\,dx
=-\int_\Omega u\;\delta\,\xi\,\nabla\!\cdot\!\big(W(\phi)\nabla\eta\big)\,dx .
$$

**第三步：读系数。**

$$
\boxed{\ \frac{\delta L}{\delta\eta}
=-\delta\,\xi\,\nabla\!\cdot\!\big(W(\phi)\nabla\eta\big)+\delta\,\tfrac1\xi\,W(\phi)\,\eta(\eta^2-1)\ }
$$

#### 4.2.3 面积差约束罚项 $\tfrac{M_3}2(D-a_d)^2$

$D(\phi,\eta)=\int_\Omega\tanh(\eta/\xi)\,W(\phi)\,dx$，$W(\phi)$ 冻结。

**外层。**

$$
\mathrm d\Big[\tfrac{M_3}2(D-a_d)^2\Big](\eta)[u]=M_3(D-a_d)\;\mathrm dD(\eta)[u].
$$

**内层 $\mathrm dD(\eta)[u]$。** 仅 $\tanh(\eta/\xi)$ 含 $\eta$：

$$
\mathrm dD(\eta)[u]=\int_\Omega W(\phi)\left.\frac{d}{d\varepsilon}\tanh\!\big((\eta+\varepsilon u)/\xi\big)\right|_{\varepsilon=0}dx
=\int_\Omega W(\phi)\,\tfrac1\xi\,\mathrm{sech}^2(\eta/\xi)\,u\,dx,
$$

无 $u$ 的导数，

$$
\frac{\delta D}{\delta\eta}=\tfrac1\xi\,W(\phi)\,\mathrm{sech}^2(\eta/\xi).
$$

**合并。**

$$
\boxed{\ \frac{\delta}{\delta\eta}\Big[\tfrac{M_3}2(D-a_d)^2\Big]
=M_3(D-a_d)\,\tfrac1\xi\,W(\phi)\,\mathrm{sech}^2(\eta/\xi)\ }
$$

#### 4.2.4 正交正则项罚项 $\tfrac{M_4}2N^2$

$N(\phi,\eta)=\int_\Omega\tfrac\epsilon2|\nabla\phi\cdot\nabla\eta|^2\,dx$，$\phi$ 冻结。

**外层。**

$$
\mathrm d\Big[\tfrac{M_4}2N^2\Big](\eta)[u]=M_4\,N\;\mathrm dN(\eta)[u].
$$

**内层 $\mathrm dN(\eta)[u]$。**

$$
\mathrm dN(\eta)[u]=\int_\Omega\epsilon\,(\nabla\phi\cdot\nabla\eta)\,(\nabla\phi\cdot\nabla u)\,dx,
$$

含 $\nabla u$，分部积分（周期边界）：

$$
\frac{\delta N}{\delta\eta}=-\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\,\nabla\phi\big).
$$

**合并。**

$$
\boxed{\ \frac{\delta}{\delta\eta}\Big[\tfrac{M_4}2N^2\Big]
=-M_4\,N\,\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\,\nabla\phi\big)\ }
$$

#### 4.2.5 $\eta$ 剖面正则项罚项 $\tfrac{M_5}2P^2$

$$
P(\eta)=\int_\Omega\Big(\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2\Big)^2 dx .
$$

记内层括号 $\,\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2\,$，它是 $\eta$ 的泛函（其平方积分给出 $P$）。

**外层（$P^2$ 平方）。**

$$
\mathrm d\Big[\tfrac{M_5}2P^2\Big](\eta)[u]=M_5\,P\;\mathrm dP(\eta)[u].
$$

**内层 $\mathrm dP(\eta)[u]$。** 第一步，$P=\int(\text{内层括号})^2$：

$$
\mathrm dP(\eta)[u]
=\int_\Omega 2\Big(\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2\Big)
\left.\frac{d}{d\varepsilon}\Big(\tfrac\xi2|\nabla\eta+\varepsilon\nabla u|^2-\tfrac1{4\xi}\big((\eta+\varepsilon u)^2-1\big)^2\Big)\right|_{\varepsilon=0}dx,
$$

$$
\left.\frac{d}{d\varepsilon}(\cdots)\right|_{\varepsilon=0}
=\xi\,\nabla\eta\cdot\nabla u-\tfrac1\xi\eta(\eta^2-1)\,u .
$$

记权重 $\Pi(\eta):=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2$，则

$$
\mathrm dP(\eta)[u]=\int_\Omega 2\,\Pi(\eta)\Big(\xi\,\nabla\eta\cdot\nabla u-\tfrac1\xi\eta(\eta^2-1)\,u\Big)dx .
$$

第二步，含 $\nabla u$ 项分部积分（周期边界）：

$$
\int_\Omega 2\,\Pi(\eta)\,\xi\,\nabla\eta\cdot\nabla u\,dx
=-\int_\Omega u\;2\,\xi\,\nabla\!\cdot\!\big(\Pi(\eta)\nabla\eta\big)\,dx .
$$

第三步：

$$
\frac{\delta P}{\delta\eta}
=-2\,\xi\,\nabla\!\cdot\!\big(\Pi(\eta)\nabla\eta\big)-2\,\tfrac1\xi\,\Pi(\eta)\,\eta(\eta^2-1).
$$

**合并。**

$$
\boxed{\ \frac{\delta}{\delta\eta}\Big[\tfrac{M_5}2P^2\Big]
=M_5\,P\Big(-2\,\xi\,\nabla\!\cdot\!\big(\Pi(\eta)\nabla\eta\big)-\tfrac2\xi\,\Pi(\eta)\,\eta(\eta^2-1)\Big),
\qquad \Pi(\eta)=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2\ }
$$

#### 4.2.6 合计

$$
\boxed{
\begin{aligned}
\frac{\delta\mathcal E_M}{\delta\eta}
=\;&\frac{k'(\eta)}{2k(\eta)}\,q^2\\[4pt]
&-\delta\,\xi\,\nabla\!\cdot\!\big(W(\phi)\nabla\eta\big)+\delta\,\tfrac1\xi\,W(\phi)\,\eta(\eta^2-1)\\[4pt]
&+M_3(D-a_d)\,\tfrac1\xi\,W(\phi)\,\mathrm{sech}^2(\eta/\xi)\\[4pt]
&-M_4\,N\,\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\,\nabla\phi\big)\\[4pt]
&+M_5\,P\Big(-2\,\xi\,\nabla\!\cdot\!\big(\Pi(\eta)\nabla\eta\big)-\tfrac2\xi\,\Pi(\eta)\,\eta(\eta^2-1)\Big)
\end{aligned}}
$$

其中 $q=a(\eta)\big(\epsilon\Delta\phi+g(\phi)\big)$，$\Pi(\eta)=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2$。

---

### 4.3 说明

- 4.1、4.2 全部用"方向导数 → 分部积分 → 读系数"三步直接对**泛函**执行，不引入辅助变量的"变分导数"记号。
- 4.1.1 弯曲能的变分导数含 $\Delta(k(\eta)(\epsilon\Delta\phi+g(\phi)))$，为 $\phi$ 的六阶非线性算子；这是后续时间格式须用 IEQ 二次化、半隐式线性化处理的对象。
- 含罚项的 $A,D,N,P$ 各项，其变分导数中外层标量因子 $(A-a_0),(D-a_d),N,P$ 在时间离散时取旧层值冻结成常数，内层经辅助变量 $q,r,\mathbf s,p$ 线性化——此为后续时间格式章节的内容。
- 4.2.1 中 $\dfrac{\delta E}{\delta\eta}=\dfrac{k'(\eta)}{2k(\eta)}q^2$ 不含 $\eta$ 的导数，是纯乘法型，在 $\eta$-子步作为已知系数项处理。

---

## 5. 一阶解耦 IEQ 时间格式与谱空间离散（完整 IEQ）

### 5.0 总体设计与约定

时间步 $\tau$。一个时间步从 $(\phi^n,\eta^n)$ 推进到 $(\phi^{n+1},\eta^{n+1})$，分两个解耦子步：

$$
(\phi^n,\eta^n)\ \xrightarrow{\ \phi\text{-子步：固定 }\eta=\eta^n\ }\ \phi^{n+1}
\ \xrightarrow{\ \eta\text{-子步：固定 }\phi=\phi^{n+1}\ }\ \eta^{n+1}.
$$

本方案为**完整 IEQ**（路线 B）：能量中含 $W(\phi)$ 的弯曲能 $E$ 与线张力/面积/面积差 $L,A,D$ 分别经辅助变量 $q,r$ 二次化，正交项 $N$ 经 $\mathbf s$、$\eta$ 剖面项 $P$ 经 $p$ 二次化；所有辅助变量在 $n+1$ 层隐式，用其线性演化恒等式回代消元。本章约定：

- **D1（辅助变量隐式回代）**：$q,r,\mathbf s,p$ 在 $n+1$ 层取隐式，由第 3 节演化恒等式的离散式回代入场方程消元。
- **D2（系数因子冻结）**：辅助变量演化恒等式与变分导数中的非线性系数因子（$a(\eta),g'(\phi),1/r,\Phi_\eta,\tanh(\eta/\xi),W(\phi)$ 等）取旧层值。这是使 $n+1$ 层辅助变量与未知场成**线性**关系的必要条件。
- **D3（解耦完全冻结）**：$\phi$-子步中 $\eta$ 的一切出现取 $\eta^n$；$\eta$-子步中 $\phi$ 的一切出现取 $\phi^{n+1}$。
- **D4（罚项外层 Newton 线性化）**【标记：Newton 线性化】：罚项 $\tfrac{M_i}2(G-g_*)^2$（$G=A,D$）的外层标量，不取旧层冻结，而作一阶 Newton 线性化
  $$
  G^{n+1}-g_*\approx (G^n-g_*)+\mathrm dG(\cdot^n)[\,\delta(\cdot)\,],
  $$
  使罚项变分中产生的 $\mathrm dG[\delta(\cdot)]\cdot\mathrm dG[\,\cdot\,]$ 部分为 $\mathcal C^*\mathcal C$ 型、天然半正定，进入 LHS 不破坏算子正性。**动机**：旧层冻结下 $(G^n-g_*)$ 暂态可负，负标量乘 $-\epsilon\Delta$ 给出反扩散，破坏谱算子正性（见 5.3）；Newton 线性化规避此问题。体积约束 $V$ 为线性泛函，$(V-v_d)$ 直接隐式，无需 Newton 线性化。
- **D5（双井项 Eyre 凸劈分 + 稳定化）**【标记：Eyre 稳定化】：非线性双井项（$\phi$ 的 $\tfrac1\epsilon\phi(\phi^2-1)$、$\eta$ 的 $\tfrac1\xi\eta(\eta^2-1)$）采用 Eyre 凸劈分思想，在场方程两端加稳定化项 $\sigma(\cdot^{n+1}-\cdot^n)$：LHS 加 $\sigma\mathcal I$、RHS 加 $\sigma(\cdot^n)$，净效果为稳定化、不改变稳态不动点。稳定化常数 $\sigma>0$ 列入待标定量。
- **D6（分部积分严格化）**：辅助变量演化式中场增量上携带的 $\nabla(\cdot)$，在回代入场方程并与测试函数作 $L^2$ 内积时，由分部积分（周期边界，边界项消失）转移；本章每处分部积分均显式写出。

记号：场增量 $\delta\phi:=\phi^{n+1}-\phi^n$，$\delta\eta:=\eta^{n+1}-\eta^n$。$r$ 含正则常数 $C_0$：$r=\sqrt{2W(\phi)+C_0}$，$r^n\ge\sqrt{C_0}>0$。

---

### 5.1 $\phi$-子步：辅助变量演化恒等式的离散

固定 $\eta=\eta^n$（D3）。$\phi$-子步涉及辅助变量 $q,r,\mathbf s$（$p$ 仅 $\eta$ 相关，不涉及）。由第 3 节恒等式离散，$\eta$ 冻结使含 $\eta_t$ 的项为零，系数因子冻结到旧层（D2）。

**$q$ 的离散演化（由 3.1 节，$\eta_t=0$）：**

$$
q^{n+1}=q^n+a(\eta^n)\big(\epsilon\Delta+g'(\phi^n)\big)\,\delta\phi .
\tag{5.1a}
$$

**$r$ 的离散演化（由 3.2 节，$1/r$ 取 $1/r^n$）：**

$$
r^{n+1}=r^n+\frac{1}{r^n}\Big(\epsilon\,\nabla\phi^n\cdot\nabla\,\delta\phi+\tfrac1\epsilon\phi^n((\phi^n)^2-1)\,\delta\phi\Big).
\tag{5.1b}
$$

**$\mathbf s$ 的离散演化（由 3.3 节，$\eta_t=0$）：**

$$
\mathbf s^{n+1}=\mathbf s^n+\sqrt\epsilon\,\nabla\,\delta\phi\cdot\nabla\eta^n .
\tag{5.1c}
$$

(5.1a)(5.1b)(5.1c) 右端均为 $\delta\phi$ 的线性表达式。为简记，定义线性算子

$$
\mathcal C^n_q[\psi]:=a(\eta^n)\big(\epsilon\Delta\psi+g'(\phi^n)\psi\big),\qquad
\mathcal C^n_r[\psi]:=\frac1{r^n}\Big(\epsilon\,\nabla\phi^n\cdot\nabla\psi+\tfrac1\epsilon\phi^n((\phi^n)^2-1)\psi\Big),
$$

$$
\mathcal C^n_s[\psi]:=\sqrt\epsilon\,\nabla\psi\cdot\nabla\eta^n ,
$$

则 $q^{n+1}=q^n+\mathcal C^n_q[\delta\phi]$，$r^{n+1}=r^n+\mathcal C^n_r[\delta\phi]$，$\mathbf s^{n+1}=\mathbf s^n+\mathcal C^n_s[\delta\phi]$。

---

### 5.2 $\phi$-子步：场方程与回代消元

$L^2$ 梯度流一阶离散，加 Eyre 稳定化（D5）：

$$
\frac{\phi^{n+1}-\phi^n}{\tau}+\sigma\,\delta\phi
=-\Big(\frac{\delta\mathcal E_M}{\delta\phi}\Big)_{\text{IEQ}}+\sigma\,\delta\phi\Big|_{\text{显式抵消}} ,
$$

实际写法为：LHS 含 $\tfrac1\tau\delta\phi+\sigma\delta\phi$ 与隐式算子，RHS 含 $\sigma$ 项的显式抵消已并入。下面逐项给出 $\big(\delta\mathcal E_M/\delta\phi\big)_{\text{IEQ}}$。

#### 5.2.1 弯曲项（经 $q$）

4.1.1：$\dfrac{\delta E}{\delta\phi}=\epsilon\Delta(a(\eta)q)+g'(\phi)a(\eta)q$。IEQ 离散：系数 $a(\eta^n),g'(\phi^n)$ 冻结，$q$ 取 $q^{n+1}$：

$$
\Big(\frac{\delta E}{\delta\phi}\Big)_{\text{IEQ}}
=\epsilon\Delta\big(a(\eta^n)q^{n+1}\big)+g'(\phi^n)a(\eta^n)q^{n+1}.
$$

代入 (5.1a) $q^{n+1}=q^n+\mathcal C^n_q[\delta\phi]$。隐式算子部分

$$
\mathcal A^n_E[\psi]:=\big(\epsilon\Delta\,a(\eta^n)\cdot+g'(\phi^n)a(\eta^n)\cdot\big)\,\mathcal C^n_q[\psi],
$$

显式部分 $\big(\epsilon\Delta\,a(\eta^n)\cdot+g'(\phi^n)a(\eta^n)\cdot\big)q^n$ 入右端。

注：$\mathcal A^n_E$ 是 $\mathcal C^{n*}_q\mathcal C^n_q$ 型（弯曲项变分导数算子恰为 $\mathcal C^n_q$ 的伴随作用），对称半正定。

#### 5.2.2 线张力项（经 $r$，加权二次型）

线张力能 $L=\int\tfrac\delta2\,\Phi'_\eta\,(r^2-C_0)\,dx$，其中 $\Phi'_\eta:=\tfrac\xi2|\nabla\eta|^2+\tfrac1{4\xi}(\eta^2-1)^2$（即 $\Phi_\eta=\delta\Phi'_\eta$）。$L$ 对 $r$ 是**加权二次型**（权重 $\Phi_\eta$）。

$L$ 对 $r$ 的变分密度为 $\Phi_\eta\,r$。IEQ 离散：权重 $\Phi_\eta$ 冻结（$\phi$-子步 $\eta$ 冻结，D3；$\Phi_\eta$ 为已知正系数），$r$ 取 $r^{n+1}$。$L$ 对 $\phi$ 的变分经 $r$ 的链式作用——由 $L$ 对 $r$ 变分 $\Phi_\eta r$ 与 $r$ 演化 (5.1b) 复合：

$$
\Big(\frac{\delta L}{\delta\phi}\Big)_{\text{IEQ}}
=\big(\mathcal C^n_r\big)^{*}\big[\Phi_\eta\,r^{n+1}\big],
$$

其中 $(\mathcal C^n_r)^*$ 为 $\mathcal C^n_r$ 的伴随（由分部积分，D6）：对任意 $\psi,h$，

$$
\big(\mathcal C^n_r[\psi],\,h\big)
=\int_\Omega\Big(\tfrac{\epsilon}{r^n}\nabla\phi^n\cdot\nabla\psi+\tfrac1{\epsilon r^n}\phi^n((\phi^n)^2-1)\psi\Big)h\,dx,
$$

含 $\nabla\psi$ 项分部积分（周期边界，边界项消失）：

$$
\big(\mathcal C^n_r[\psi],\,h\big)
=\int_\Omega\psi\Big(-\epsilon\,\nabla\!\cdot\!\big(\tfrac{h}{r^n}\nabla\phi^n\big)+\tfrac1{\epsilon r^n}\phi^n((\phi^n)^2-1)\,h\Big)dx,
$$

故

$$
\big(\mathcal C^n_r\big)^{*}[h]
=-\epsilon\,\nabla\!\cdot\!\big(\tfrac{h}{r^n}\nabla\phi^n\big)+\tfrac1{\epsilon r^n}\phi^n((\phi^n)^2-1)\,h .
$$

代入 (5.1b) $r^{n+1}=r^n+\mathcal C^n_r[\delta\phi]$，线张力项

$$
\Big(\frac{\delta L}{\delta\phi}\Big)_{\text{IEQ}}
=\big(\mathcal C^n_r\big)^{*}\big[\Phi_\eta(r^n+\mathcal C^n_r[\delta\phi])\big].
$$

隐式算子部分

$$
\mathcal A^n_L[\psi]:=\big(\mathcal C^n_r\big)^{*}\big[\Phi_\eta\,\mathcal C^n_r[\psi]\big],
$$

显式部分 $(\mathcal C^n_r)^*[\Phi_\eta\,r^n]$ 入右端。$\mathcal A^n_L$ 形如 $\mathcal C^*(\Phi_\eta\,\mathcal C)$，$\Phi_\eta>0$ 时对称半正定。

#### 5.2.3 面积约束罚项（经 $r$，Newton 线性化）

$\tfrac{M_2}2(A-a_0)^2$，$A=\tfrac12\int(r^2-C_0)dx$。

**外层 Newton 线性化（D4）。** $A$ 对 $\phi$ 的方向导数 $\mathrm dA(\phi^n)[\delta\phi]$ 经 $r$ 链式：$A$ 对 $r$ 变分密度为 $r$，故 $\mathrm dA(\phi^n)[\delta\phi]=\big(r^n,\,\mathcal C^n_r[\delta\phi]\big)=\big((\mathcal C^n_r)^*[r^n],\,\delta\phi\big)$（数值标量）。Newton 线性化：

$$
A^{n+1}-a_0\approx(A^n-a_0)+\mathrm dA(\phi^n)[\delta\phi].
$$

罚项变分 $\dfrac{\delta}{\delta\phi}[\tfrac{M_2}2(A-a_0)^2]=M_2(A-a_0)\,\dfrac{\delta A}{\delta\phi}$。其中 $\dfrac{\delta A}{\delta\phi}$ 经 $r$ 为 $(\mathcal C^n_r)^*[r^{n+1}]$（与 5.2.2 同构，权重为 $1$）。代入 Newton 线性化的 $(A-a_0)$ 与 (5.1b)：

$$
\Big(\frac{\delta}{\delta\phi}\Big[\tfrac{M_2}2(A-a_0)^2\Big]\Big)_{\text{IEQ}}
=M_2\Big[(A^n-a_0)+\mathrm dA(\phi^n)[\delta\phi]\Big]\cdot(\mathcal C^n_r)^*[r^n+\mathcal C^n_r[\delta\phi]].
$$

展开，**隐式部分取至 $\delta\phi$ 一次**（Newton 线性化只保留一阶）：

- $\mathcal C^*\mathcal C$ 型半正定项（进 LHS）：$M_2\,\mathrm dA(\phi^n)[\delta\phi]\cdot(\mathcal C^n_r)^*[r^n]$。注意 $\mathrm dA(\phi^n)[\delta\phi]=\big((\mathcal C^n_r)^*[r^n],\delta\phi\big)$，记 $e^n:=(\mathcal C^n_r)^*[r^n]$，则此项为 $M_2\,(e^n,\delta\phi)\,e^n$，即秩一算子 $M_2\,e^n\!\otimes\!e^n$，对称半正定。
- 旧层标量乘隐式项（进 LHS）：$M_2(A^n-a_0)\,(\mathcal C^n_r)^*[\mathcal C^n_r[\delta\phi]]$。此项系数 $(A^n-a_0)$ 可负——见下。

**符号说明【标记：Newton 线性化】。** 上式第二项 $(A^n-a_0)$ 暂态可负，但它乘的是 $\mathcal C^*\mathcal C$ 型算子 $(\mathcal C^n_r)^*\mathcal C^n_r$（半正定）。负系数乘半正定算子得半负定，可能破坏 LHS 正性。处理：将该项也并入 Newton 一阶余项的统一表达——严格的 Newton 线性化下，罚项 $\tfrac{M_2}2(A-a_0)^2$ 的二阶变分（Hessian）为

$$
M_2\,\big(\mathrm dA[\cdot]\big)\otimes\big(\mathrm dA[\cdot]\big)+M_2(A-a_0)\,\mathrm d^2A[\cdot,\cdot],
$$

第一项 $\mathcal C^*\mathcal C$ 型半正定；第二项含 $(A-a_0)$。Newton-IEQ 实现取**仅保留半正定的第一项进 LHS**，第二项 $M_2(A^n-a_0)\,\mathrm d^2A$ 显式置于右端（旧层）。即隐式只保留秩一半正定算子

$$
\mathcal A^n_A[\psi]:=M_2\,(e^n,\psi)\,e^n,\qquad e^n:=(\mathcal C^n_r)^*[r^n],
$$

其余 $(A^n-a_0)$ 相关项全部显式入右端。这样 LHS 严格半正定，谱算子正性得保（见 5.3）。

#### 5.2.4 面积差约束罚项（经 $r$，Newton 线性化）

$\tfrac{M_3}2(D-a_d)^2$，$D=\tfrac12\int\tanh(\eta/\xi)(r^2-C_0)dx$。$\phi$-子步 $\tanh(\eta^n/\xi)$ 冻结。

与 5.2.3 同构。$D$ 对 $r$ 变分密度为 $\tanh(\eta^n/\xi)\,r$。记

$$
e^n_D:=(\mathcal C^n_r)^*\big[\tanh(\eta^n/\xi)\,r^n\big],
$$

$\mathrm dD(\phi^n)[\delta\phi]=(e^n_D,\delta\phi)$。Newton-IEQ：隐式只保留半正定秩一项

$$
\mathcal A^n_D[\psi]:=M_3\,(e^n_D,\psi)\,e^n_D ,
$$

含 $(D^n-a_d)$ 的二阶项显式入右端。

#### 5.2.5 体积约束罚项

$V=\int\phi\,dx$ 线性泛函，$\delta V/\delta\phi=1$，无需 Newton 线性化。$V^{n+1}$ 隐式：

$$
\Big(\frac{\delta}{\delta\phi}\Big[\tfrac{M_1}2(V-v_d)^2\Big]\Big)_{\text{IEQ}}=M_1(V^{n+1}-v_d),
\qquad V^{n+1}=\int_\Omega\phi^{n+1}\,dx .
$$

隐式算子部分为秩一非局部算子 $\mathcal A^n_V[\psi]:=M_1\,(1,\psi)\cdot 1=M_1\,\mathbf 1\!\otimes\!\mathbf 1[\psi]$（$\mathbf 1$ 为常值场），显式部分 $-M_1 v_d$ 与 $M_1(1,\phi^n)$ 处理见 5.2.7。

#### 5.2.6 正交约束罚项（经 $\mathbf s$）

$\tfrac{M_4}2N^2$，$N=\tfrac12\|\mathbf s\|^2$。$N$ 是 $\mathbf s$ 的纯二次型。

外层标量按 D4 同样作 Newton 线性化：$N$ 对 $\phi$ 方向导数 $\mathrm dN(\phi^n)[\delta\phi]=(\mathbf s^n,\mathcal C^n_s[\delta\phi])=((\mathcal C^n_s)^*[\mathbf s^n],\delta\phi)$。$N$ 对 $\mathbf s$ 变分为 $\mathbf s$，$N$ 对 $\phi$ 变分经 $\mathbf s$ 为 $(\mathcal C^n_s)^*[\mathbf s^{n+1}]$。

$(\mathcal C^n_s)^*$ 由分部积分（D6）：$\mathcal C^n_s[\psi]=\sqrt\epsilon\,\nabla\psi\cdot\nabla\eta^n$，

$$
(\mathcal C^n_s)^*[h]=-\sqrt\epsilon\,\nabla\!\cdot\!\big(h\,\nabla\eta^n\big).
$$

记 $e^n_N:=(\mathcal C^n_s)^*[\mathbf s^n]$。Newton-IEQ：隐式只保留半正定秩一项

$$
\mathcal A^n_N[\psi]:=M_4\,(e^n_N,\psi)\,e^n_N ,
$$

含 $N^n$ 的二阶项显式入右端。

#### 5.2.7 $\phi^{n+1}$ 的线性方程

合并 5.2.1–5.2.6，并入 Eyre 稳定化（D5）。$\phi^{n+1}$ 满足

$$
\boxed{\
\Big(\tfrac1\tau\mathcal I+\sigma\mathcal I+\mathcal A^n_E+\mathcal A^n_L+\mathcal A^n_A+\mathcal A^n_D+\mathcal A^n_N+\mathcal A^n_V\Big)\phi^{n+1}=\mathcal R^n_\phi\
}
$$

其中各隐式算子如上定义，均对称半正定（$\mathcal A^n_E,\mathcal A^n_L$ 为 $\mathcal C^*\mathcal C$ 型；$\mathcal A^n_A,\mathcal A^n_D,\mathcal A^n_N,\mathcal A^n_V$ 为半正定秩一算子），故 LHS 算子对称正定。右端

$$
\begin{aligned}
\mathcal R^n_\phi=\;&\tfrac1\tau\phi^n+\sigma\phi^n
-\big(\epsilon\Delta\,a(\eta^n)+g'(\phi^n)a(\eta^n)\big)q^n
-(\mathcal C^n_r)^*[\Phi_\eta\,r^n]\\
&-(\text{5.2.3、5.2.4、5.2.6 中含 }(A^n-a_0),(D^n-a_d),N^n\text{ 的二阶显式项})\\
&+M_1 v_d+(\text{秩一算子 }\mathcal A^n_V\text{ 作用于 }\phi^n\text{ 的移项，并入 LHS 已计})\\
&-(\text{Eyre 与 Newton 线性化产生的其余旧层显式项}).
\end{aligned}
$$

（右端各显式项由对应小节的"显式入右端"汇集；$\mathcal A^n_V$ 作用于未知 $\phi^{n+1}$ 已在 LHS。）

解出 $\phi^{n+1}$ 后，由 (5.1a)(5.1b)(5.1c) 显式更新 $q^{n+1},r^{n+1},\mathbf s^{n+1}$。

---

### 5.3 谱算子正性

经 5.2.3、5.2.4、5.2.6 的 Newton-IEQ 处理，LHS 所有隐式算子均对称半正定：

- $\mathcal A^n_E=\mathcal C^{n*}_q\mathcal C^n_q$、$\mathcal A^n_L=\mathcal C^{n*}_r(\Phi_\eta\,\mathcal C^n_r)$（$\Phi_\eta>0$）——$\mathcal C^*\mathcal C$ 型，半正定；
- $\mathcal A^n_A,\mathcal A^n_D,\mathcal A^n_N$——半正定秩一算子 $e\!\otimes\!e$；
- $\mathcal A^n_V$——半正定秩一算子 $\mathbf 1\!\otimes\!\mathbf 1$；
- $\tfrac1\tau\mathcal I+\sigma\mathcal I$——严格正定。

故 LHS $=\tfrac1\tau\mathcal I+\sigma\mathcal I+\sum(\text{半正定})$ 严格正定，对任意 $\tau,\sigma>0$、任意暂态 $(A^n-a_0),(D^n-a_d),N^n$ 符号均成立。**含 $(A^n-a_0)$ 等可变号标量的二阶项已全部显式置于右端，不进 LHS**，因此不存在反扩散破坏正性的问题。

---

### 5.4 $\eta$-子步：辅助变量演化恒等式的离散

固定 $\phi=\phi^{n+1}$（D3）。$\eta$-子步涉及辅助变量 $p,\mathbf s$（$q$ 已在 $\phi$-子步定值，$\eta$-子步弯曲项 4.2.1 中 $q$ 取 $q^{n+1}$ 为已知；$r$ 同理已定值）。

**$p$ 的离散演化（由 3.4 节，系数冻结到 $\eta^n$）：**

$$
p^{n+1}=p^n+\xi\,\nabla\eta^n\cdot\nabla\,\delta\eta-\tfrac1\xi\,\eta^n\zeta^n\,\delta\eta,
\qquad \zeta^n:=(\eta^n)^2-1.
\tag{5.4a}
$$

**$\mathbf s$ 的离散演化（由 3.3 节，$\phi$-子步 $\phi_t=0$；$\eta$-子步 $\phi$ 冻结）：**

$$
\mathbf s^{n+1}=\mathbf s^{n+1}_{(\phi)}+\sqrt\epsilon\,\nabla\phi^{n+1}\cdot\nabla\,\delta\eta ,
\tag{5.4b}
$$

其中 $\mathbf s^{n+1}_{(\phi)}$ 为 $\phi$-子步结束时的 $\mathbf s$ 值（5.2.7）。

线性算子：

$$
\mathcal D^n_p[\psi]:=\xi\,\nabla\eta^n\cdot\nabla\psi-\tfrac1\xi\eta^n\zeta^n\psi ,\qquad
\mathcal D^n_s[\psi]:=\sqrt\epsilon\,\nabla\phi^{n+1}\cdot\nabla\psi ,
$$

则 $p^{n+1}=p^n+\mathcal D^n_p[\delta\eta]$，$\mathbf s^{n+1}=\mathbf s^{n+1}_{(\phi)}+\mathcal D^n_s[\delta\eta]$。

---

### 5.5 $\eta$-子步：场方程与回代消元

$L^2$ 梯度流一阶离散，加 Eyre 稳定化（D5）：LHS 含 $\tfrac1\tau\delta\eta+\sigma\delta\eta$。逐项给出 $\big(\delta\mathcal E_M/\delta\eta\big)_{\text{IEQ}}$。

#### 5.5.1 弯曲项

4.2.1：$\dfrac{\delta E}{\delta\eta}=\dfrac{k'(\eta)}{2k(\eta)}q^2$。不含 $\eta$ 的导数。$q^{n+1}$ 已由 $\phi$-子步算出，系数 $\dfrac{k'(\eta^n)}{2k(\eta^n)}$ 冻结。整项纯已知，入右端：

$$
\Big(\frac{\delta E}{\delta\eta}\Big)_{\text{IEQ}}=\frac{k'(\eta^n)}{2k(\eta^n)}\,(q^{n+1})^2 .
$$

#### 5.5.2 线张力项（经 $r$ 已定值；$\eta$ 的二次型 + 双井 Eyre）

4.2.2：$\dfrac{\delta L}{\delta\eta}=-\delta\xi\nabla\!\cdot\!(W(\phi)\nabla\eta)+\delta\tfrac1\xi W(\phi)\,\eta(\eta^2-1)$。

$W(\phi^{n+1})=\tfrac12((r^{n+1})^2-C_0)$ 为已知系数。第一项 $-\delta\xi\nabla\!\cdot\!(W(\phi^{n+1})\nabla\eta^{n+1})$ 为 $\eta^{n+1}$ 的线性主部，隐式（变系数 $W(\phi^{n+1})>0$ 时半正定）。第二项双井 $\eta(\eta^2-1)$ 按 D5 Eyre 凸劈分：系数 $\zeta^n=(\eta^n)^2-1$ 冻结，线性因子 $\eta^{n+1}$ 隐式，并加稳定化（$\zeta^n<0$ 时该零阶项系数为负，Eyre 稳定化常数 $\sigma$ 补偿其反向作用）：

$$
\Big(\frac{\delta L}{\delta\eta}\Big)_{\text{IEQ}}
=-\delta\xi\,\nabla\!\cdot\!\big(W(\phi^{n+1})\nabla\eta^{n+1}\big)+\delta\tfrac1\xi W(\phi^{n+1})\,\zeta^n\,\eta^{n+1}.
$$

隐式算子部分

$$
\mathcal B^n_L[\psi]:=-\delta\xi\,\nabla\!\cdot\!\big(W(\phi^{n+1})\nabla\psi\big)+\delta\tfrac1\xi W(\phi^{n+1})\,\zeta^n\,\psi .
$$

注：$\mathcal B^n_L$ 的零阶项系数 $\delta\tfrac1\xi W\zeta^n$ 在 $\zeta^n<0$ 时为负；其负性由 LHS 的 $\sigma\mathcal I$ 稳定化项与扩散主部共同压住（见 5.6 正性讨论）。

#### 5.5.3 面积差约束罚项（Newton 线性化）

$\tfrac{M_3}2(D-a_d)^2$，$\eta$-子步 $D=\tfrac12\int\tanh(\eta/\xi)((r^{n+1})^2-C_0)dx$，$r^{n+1}$ 冻结，$\eta$ 经 $\tanh(\eta/\xi)$ 进入。

$D$ 对 $\eta$ 方向导数：$\mathrm dD(\eta^n)[\delta\eta]=\int W(\phi^{n+1})\tfrac1\xi\mathrm{sech}^2(\eta^n/\xi)\,\delta\eta\,dx$。记

$$
f^n_D:=\tfrac1\xi\,W(\phi^{n+1})\,\mathrm{sech}^2(\eta^n/\xi),
$$

则 $\mathrm dD(\eta^n)[\delta\eta]=(f^n_D,\delta\eta)$，$\delta D/\delta\eta=f^n_D$（纯乘法型，无导数）。Newton-IEQ（D4）：隐式只保留半正定秩一项

$$
\mathcal B^n_D[\psi]:=M_3\,(f^n_D,\psi)\,f^n_D ,
$$

含 $(D^n-a_d)$ 的二阶项显式入右端。

#### 5.5.4 正交约束罚项（经 $\mathbf s$，Newton 线性化）

$\tfrac{M_4}2N^2$。$\eta$-子步 $\phi=\phi^{n+1}$ 冻结。$N$ 对 $\eta$ 经 $\mathbf s$：$\mathrm dN(\eta^n)[\delta\eta]=(\mathbf s^{n+1}_{(\phi)},\mathcal D^n_s[\delta\eta])=((\mathcal D^n_s)^*[\mathbf s^{n+1}_{(\phi)}],\delta\eta)$。

$(\mathcal D^n_s)^*$ 由分部积分（D6）：$\mathcal D^n_s[\psi]=\sqrt\epsilon\,\nabla\phi^{n+1}\cdot\nabla\psi$，

$$
(\mathcal D^n_s)^*[h]=-\sqrt\epsilon\,\nabla\!\cdot\!\big(h\,\nabla\phi^{n+1}\big).
$$

记 $e^n_{N\eta}:=(\mathcal D^n_s)^*[\mathbf s^{n+1}_{(\phi)}]$。Newton-IEQ：隐式只保留半正定秩一项

$$
\mathcal B^n_N[\psi]:=M_4\,(e^n_{N\eta},\psi)\,e^n_{N\eta},
$$

含 $N$ 的二阶项显式入右端。

#### 5.5.5 $\eta$ 剖面正则罚项（经 $p$，Newton 线性化）

$\tfrac{M_5}2P^2$，$P=\|p\|^2$。$P$ 是 $p$ 的纯二次型。

$P$ 对 $\eta$ 经 $p$：$\mathrm dP(\eta^n)[\delta\eta]=2(p^n,\mathcal D^n_p[\delta\eta])=2((\mathcal D^n_p)^*[p^n],\delta\eta)$。

$(\mathcal D^n_p)^*$ 由分部积分（D6）：$\mathcal D^n_p[\psi]=\xi\nabla\eta^n\cdot\nabla\psi-\tfrac1\xi\eta^n\zeta^n\psi$，

$$
(\mathcal D^n_p)^*[h]=-\xi\,\nabla\!\cdot\!\big(h\,\nabla\eta^n\big)-\tfrac1\xi\eta^n\zeta^n\,h .
$$

记 $e^n_P:=(\mathcal D^n_p)^*[p^n]$。Newton-IEQ：隐式只保留半正定秩一项

$$
\mathcal B^n_P[\psi]:=M_5\cdot 4\,(e^n_P,\psi)\,e^n_P ,
$$

（因子 $4$ 来自 $P=\|p\|^2$ 与 $\mathrm dP=2(p,\cdot)$ 的两次出现：$\tfrac{M_5}2P^2$ 的 Hessian 半正定主项为 $M_5(\mathrm dP)\otimes(\mathrm dP)=M_5\cdot 4\,e^n_P\!\otimes\!e^n_P$。）含 $P^n$ 的二阶项显式入右端。

#### 5.5.6 $\eta^{n+1}$ 的线性方程

合并 5.5.1–5.5.5，并入 Eyre 稳定化。$\eta^{n+1}$ 满足

$$
\boxed{\
\Big(\tfrac1\tau\mathcal I+\sigma\mathcal I+\mathcal B^n_L+\mathcal B^n_D+\mathcal B^n_N+\mathcal B^n_P\Big)\eta^{n+1}=\mathcal R^n_\eta\
}
$$

其中 $\mathcal B^n_D,\mathcal B^n_N,\mathcal B^n_P$ 为半正定秩一算子；$\mathcal B^n_L$ 含扩散主部（半正定）与可变号零阶项。右端

$$
\begin{aligned}
\mathcal R^n_\eta=\;&\tfrac1\tau\eta^n+\sigma\eta^n
-\frac{k'(\eta^n)}{2k(\eta^n)}(q^{n+1})^2\\
&-(\text{5.5.3、5.5.4、5.5.5 中含 }(D^n-a_d),N^n,P^n\text{ 的二阶显式项})\\
&-(\text{Eyre 与 Newton 线性化产生的其余旧层显式项}).
\end{aligned}
$$

解出 $\eta^{n+1}$ 后，由 (5.4a)(5.4b) 更新 $p^{n+1},\mathbf s^{n+1}$。

---

### 5.6 Fourier 谱空间离散

#### 5.6.1 谱表示

周期区域 $\Omega=[-\pi,\pi]^3$，$64^3$ 网格。场展开

$$
\phi(x)=\sum_{\mathbf k}\hat\phi_{\mathbf k}\,e^{i\mathbf k\cdot x},\qquad
k_j\in\{-32,\dots,31\}.
$$

微分算子对角化：$\widehat{\nabla\psi}=i\mathbf k\hat\psi$，$\widehat{\Delta\psi}=-|\mathbf k|^2\hat\psi$，$\widehat{\Delta^2\psi}=|\mathbf k|^4\hat\psi$。

#### 5.6.2 常系数隐式 + 变系数显式劈分

**【标记：FFT 对角化关键步骤】** 5.2、5.5 节的隐式算子（$\mathcal A^n_E,\mathcal A^n_L,\mathcal B^n_L$ 等）含变系数（$a(\eta^n)^2,\Phi_\eta,W(\phi^{n+1}),1/r^n,\nabla\phi^n$ 等随空间变化），变系数算子在谱空间非对角，不能逐波数求逆。

处理：将变系数劈为常数代表值加余项，常数部分隐式（谱对角）、余项显式入右端。以弯曲项最高阶为例，$\mathcal A^n_E$ 主部含 $a(\eta^n)^2\epsilon^2\Delta^2$，取常数 $\bar a^2$，

$$
a(\eta^n)^2\epsilon^2\Delta^2\psi
=\underbrace{\bar a^2\epsilon^2\Delta^2\psi^{n+1}}_{\text{常系数，隐式}}
+\underbrace{\big(a(\eta^n)^2-\bar a^2\big)\epsilon^2\Delta^2\psi^n}_{\text{变系数余项，显式入右端}}.
$$

其余隐式算子（$\mathcal A^n_L,\mathcal B^n_L$ 等）同此处理；秩一算子 $\mathcal A^n_A,\mathcal A^n_D,\dots$ 为非局部低秩项，可直接在谱空间作低秩更新（Sherman–Morrison）或并入迭代。常数代表值（$\bar a^2$ 等）取法——空间均值或上界——为待标定量。

#### 5.6.3 隐式骨架谱符号

经劈分，$\phi$-子步常系数隐式骨架谱符号

$$
\widehat{\mathcal L}(\mathbf k)=\frac1\tau+\sigma+\bar a^2\epsilon^2|\mathbf k|^4+(\text{低阶常系数项})>0 .
$$

$\tfrac1\tau+\sigma>0$ 与最高阶 $\bar a^2\epsilon^2|\mathbf k|^4\ge0$ 保证 $\widehat{\mathcal L}(\mathbf k)>0$ 对所有 $\mathbf k$；可变号的罚项二阶项已显式入右端（5.3），不影响 $\widehat{\mathcal L}$ 正性。$\eta$-子步同构，记 $\widehat{\mathcal M}(\mathbf k)>0$；其中 $\mathcal B^n_L$ 可变号零阶项的常数代表值若为负，由 $\sigma$ 保证 $\widehat{\mathcal M}>0$（$\sigma$ 取足够大）。

体积约束秩一算子 $\mathcal A^n_V$ 仅作用零波数 $\mathbf k=\mathbf0$，单独处理。

#### 5.6.4 求解步骤（每时间步）

$\phi$-子步：

1. 实空间计算冻结系数与显式项（含 5.6.2 变系数余项、5.3 罚项二阶显式项），组装右端 $\mathcal R^n_\phi$；
2. FFT 得 $\widehat{\mathcal R^n_\phi}$；
3. 非零波数逐波数求逆 $\widehat{\phi^{n+1}}_{\mathbf k}=\widehat{\mathcal R^n_\phi}_{\mathbf k}/\widehat{\mathcal L}(\mathbf k)$；秩一项 $\mathcal A^n_V$ 等用 Sherman–Morrison 低秩修正；零波数单独解；
4. IFFT 得 $\phi^{n+1}$；
5. 由 (5.1a)(5.1b)(5.1c) 更新 $q^{n+1},r^{n+1},\mathbf s^{n+1}$。

变系数余项显式，若 $\tau$ 受限可改迭代：以 $\widehat{\mathcal L}(\mathbf k)^{-1}$ 为预条件子作不动点迭代或共轭梯度，重复 1–4 至收敛。$\eta$-子步同构。

#### 5.6.5 算法流程

```
输入: φ⁰, η⁰; ε=ξ=0.1964, δ=10, c₁=c₂=0;
      v_d=-216.52, a₀=29.46, a_d=0.23;
      M₁..M₅; τ; σ; C₀; tol; k,c (k>|c|>0)

初始化辅助变量:
  q⁰ = √(k(η⁰)/ε)·(εΔφ⁰ + g(φ⁰))
  r⁰ = √(2 W(φ⁰) + C₀)
  s⁰ = √ε·(∇φ⁰·∇η⁰)
  p⁰ = (ξ/2)|∇η⁰|² − (1/4ξ)((η⁰)²−1)²

预计算: 波数 k, 谱符号 L̂(k), M̂(k)

for n = 0,1,2,... :
  ── φ-子步 (固定 η=ηⁿ, D3) ──
  1. 冻结系数 a(ηⁿ),g'(φⁿ),Φ_η,tanh(ηⁿ/ξ),1/rⁿ; 算 Aⁿ,Dⁿ,Nⁿ,Vⁿ
  2. 算辅助算子伴随作用 eⁿ=(C_r)*[rⁿ] 等; 组装右端 R_φ
     (含变系数余项 + 罚项二阶显式项 + Eyre/Newton 显式项)
  3. FFT → 非零波数除 L̂(k) + 秩一 Sherman-Morrison 修正 → IFFT ⇒ φⁿ⁺¹
  4. 更新 qⁿ⁺¹,rⁿ⁺¹,sⁿ⁺¹  (5.1a,5.1b,5.1c)

  ── η-子步 (固定 φ=φⁿ⁺¹, D3) ──
  5. 冻结系数 k'(ηⁿ)/2k(ηⁿ),W(φⁿ⁺¹),ζⁿ; 算 Dⁿ,Nⁿ,Pⁿ
  6. 组装右端 R_η
  7. FFT → 逐波数除 M̂(k) + 秩一修正 → IFFT ⇒ ηⁿ⁺¹
  8. 更新 pⁿ⁺¹,sⁿ⁺¹  (5.4a,5.4b)

  ── 收敛判据 (求稳态) ──
  if ‖φⁿ⁺¹−φⁿ‖ + ‖ηⁿ⁺¹−ηⁿ‖ < tol : break

输出: 稳态 (φ*, η*)
  Γ={φ*=0}, Γ⊥={η*=0}, γ₀={φ*=η*=0}
```

---

### 5.7 罚因子 continuation

罚因子 $M_i$ 越大约束越准但系统越刚性、变系数劈分余项越大。采用 continuation：$M_i$ 从小值起，每收敛一次按 $M_i\leftarrow\rho M_i$（$\rho\sim2$–$10$）放大并以当前解重启，直至约束残差 $|V-v_d|,|A-a_0|,|D-a_d|$ 与 $N,P$ 均小于容差。

---

### 5.8 待实验验证的可调量

以下为推导中引入、需数值实验确定的量，非推导结论：

- 正则常数 $C_0>0$（5.1b 中 $1/r^n$ 的下界由 $\sqrt{C_0}$ 控制）；
- Eyre 稳定化常数 $\sigma>0$（D5；$\eta$-子步 $\mathcal B^n_L$ 负零阶项的正性补偿亦依赖 $\sigma$）；
- 常系数劈分（5.6.2）中常数代表值 $\bar a^2$ 等的取法（均值 / 上界 / 其它）；
- 时间步 $\tau$ 的实际可用范围；
- 罚因子 $M_i$ 的初值、放大比 $\rho$ 与目标容差；
- Newton 线性化中是否需对 $\mathcal B^n_L$ 可变号零阶项作额外稳定化；
- 是否需对变系数项改用预条件迭代（5.6.4 末）。

### 5.9 说明

- 本章为完整 IEQ（路线 B）：$E,L,A,D$ 经 $q,r$、$N$ 经 $\mathbf s$、$P$ 经 $p$ 全部二次化，辅助变量 $n+1$ 层隐式回代，$L,A,D$ 与 $E,N,P$ 同享 IEQ 的线性格式结构。
- 罚项外层标量经 Newton 线性化（D4），仅半正定 Hessian 主项进 LHS，可变号的二阶项显式置右端，保证谱算子正性（5.3、5.6.3）。
- 双井项经 Eyre 凸劈分 + 稳定化常数 $\sigma$（D5）；$\eta$ 的 $(\eta^2-1)^2$ 不单独引辅助变量，由 Eyre 处理。
- 无条件能量稳定性证明留待数值实验后单独补充。
