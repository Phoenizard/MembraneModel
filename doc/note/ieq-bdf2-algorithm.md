# 经典 IEQ 算法（BDF2 二阶版）—— 两组分囊泡相场模型

> 目标：以二阶时间精度求解梯度流 $(\phi,\eta)$ 的演化及其稳态。
> 时间离散：**二阶向后差分（BDF2）解耦 IEQ**。空间离散：Fourier 谱（FFT），周期边界，$64^3$。
> 模型简化：$c_1=c_2=0\Rightarrow c_0(\eta)\equiv0$，$\epsilon=\xi$。

---

## §0 记号

定义双井相关函数：

$$
g(\phi)=\tfrac1\epsilon\phi(1-\phi^2),\qquad
g'(\phi)=\tfrac1\epsilon(1-3\phi^2),\qquad
W(\phi)=\tfrac\epsilon2|\nabla\phi|^2+\tfrac1{4\epsilon}(\phi^2-1)^2
$$

变系数弯曲刚度及相关量：

$$
k(\eta)=k+c\,\tanh(\eta/\xi),\qquad
a(\eta)=\sqrt{k(\eta)/\epsilon},\qquad
k'(\eta)=\tfrac c\xi\,\mathrm{sech}^2(\eta/\xi)
$$

$L^2$ 内积 $(f,g)=\int_\Omega fg\,dx$，范数 $\|f\|^2=(f,f)$。

**时间标记** BDF2 为三层格式，每步涉及 $\phi^{n-1},\phi^n,\phi^{n+1}$（$\eta$ 同）。场增量
$$
\delta^n\phi:=\phi^{n+1}-\phi^n,\qquad
\delta^{n-1}\phi:=\phi^n-\phi^{n-1},
$$

$\delta^n\eta,\delta^{n-1}\eta$ 同义。

**二阶外推算子** 构造差分定义
$$
\mathrm{Ext}[\,f^n,f^{n-1}\,]:=2f^n-f^{n-1} = f^n - (f^n-f^{n-1}),
$$

对光滑解满足 $\mathrm{Ext}[f^n,f^{n-1}]=f^{n+1}+O(\tau^2)$，即用两层旧值二阶逼近 $n+1$ 层值。下文凡以外推估到 $n+1$ 层的量记上标 $\star$，例如 $\phi^\star:=\mathrm{Ext}[\phi^n,\phi^{n-1}]$。

---

## §1 梯度流与能量

求解 $L^2$ 梯度流至稳态：

$$
\phi_t=-\frac{\delta\mathcal E_M}{\delta\phi},\qquad
\eta_t=-\frac{\delta\mathcal E_M}{\delta\eta}
$$

总能量为弯曲能、线张力能与五个约束/正则罚项之和：

$$
\mathcal E_M=\underbrace{E}_{\text{弯曲}}+\underbrace{L}_{\text{线张力}}
+\tfrac{M_1}2(V-v_d)^2+\tfrac{M_2}2(A-a_0)^2+\tfrac{M_3}2(D-a_d)^2
+\tfrac{M_4}2N^2+\tfrac{M_5}2P^2
$$

其中 $V$ 为体积、$A$ 为总面积、$D$ 为面积差、$N$ 为正交约束泛函、$P$ 为 $\eta$ 剖面正则泛函；$v_d,a_0,a_d$ 为目标值，$M_1,\dots,M_5$ 为罚因子。

---

## §2 变分导数

### §2.1 对 $\phi$ 的变分导数

记冻结系数 $\Phi_\eta:=\delta\big(\tfrac\xi2|\nabla\eta|^2+\tfrac1{4\xi}(\eta^2-1)^2\big)$（线张力能 $L$ 中与 $\eta$ 相关的部分）。

$$
\begin{aligned}
\frac{\delta\mathcal E_M}{\delta\phi}
=\;&\Delta\!\big(k(\eta)(\epsilon\Delta\phi+g(\phi))\big)
+\tfrac{k(\eta)}\epsilon g'(\phi)(\epsilon\Delta\phi+g(\phi))\\
&-\epsilon\,\nabla\!\cdot\!(\Phi_\eta\nabla\phi)+\tfrac1\epsilon\phi(\phi^2-1)\Phi_\eta\\
&+M_1(V-v_d)\\
&+M_2(A-a_0)\big(-\epsilon\Delta\phi+\tfrac1\epsilon\phi(\phi^2-1)\big)\\
&+M_3(D-a_d)\big(-\epsilon\nabla\!\cdot\!(\tanh(\eta/\xi)\nabla\phi)+\tfrac1\epsilon\phi(\phi^2-1)\tanh(\eta/\xi)\big)\\
&-M_4 N\,\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\nabla\eta\big)
\end{aligned}
$$

### §2.2 对 $\eta$ 的变分导数

记 $\Pi(\eta)=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2$（$\eta$ 剖面正则泛函 $P$ 的被积密度）。

$$
\begin{aligned}
\frac{\delta\mathcal E_M}{\delta\eta}
=\;&\tfrac{k'(\eta)}{2\epsilon}(\epsilon\Delta\phi+g(\phi))^2\\
&-\delta\xi\,\nabla\!\cdot\!(W(\phi)\nabla\eta)+\delta\tfrac1\xi W(\phi)\,\eta(\eta^2-1)\\
&+M_3(D-a_d)\tfrac1\xi W(\phi)\,\mathrm{sech}^2(\eta/\xi)\\
&-M_4 N\,\epsilon\,\nabla\!\cdot\!\big((\nabla\phi\cdot\nabla\eta)\nabla\phi\big)\\
&+M_5 P\big(-2\xi\nabla\!\cdot\!(\Pi(\eta)\nabla\eta)-\tfrac2\xi\Pi(\eta)\,\eta(\eta^2-1)\big)
\end{aligned}
$$

## §3 IEQ 二次化：辅助变量

IEQ 的核心是对每个非线性势引入辅助变量，使能量成为辅助变量的二次型。本模型引入四个辅助变量：

$$
\begin{array}{lll}
\text{弯曲} & q:=a(\eta)\big(\epsilon\Delta\phi+g(\phi)\big) & E=\tfrac12\|q\|^2\\[4pt]
\text{线张力/面积} & r:=\sqrt{2W(\phi)+C_0} & W(\phi)=\tfrac12(r^2-C_0)\\[4pt]
\text{正交项} & s:=\sqrt\epsilon\,(\nabla\phi\cdot\nabla\eta) & N=\tfrac12\|s\|^2\\[4pt]
\eta\ \text{剖面} & p:=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2 & P=\|p\|^2
\end{array}
$$

正定性要求：$k(\eta)>0$（取 $k>|c|$，保证 $a(\eta)$ 实值），$C_0>0$（避免后续 $1/r$ 出现奇异）。

面积 $A$ 与面积差 $D$ 经同一辅助变量 $r$ 表达：

$$
A=\tfrac12\int_\Omega(r^2-C_0)\,dx,\qquad
D=\tfrac12\int_\Omega\tanh(\eta/\xi)(r^2-C_0)\,dx
$$

### §3.1 辅助变量的演化与变分重构

辅助变量有两类关系：一类是对时间求导得到的**演化恒等式**，一类是变分导数经辅助变量改写得到的**线性重构算子**。下文以弯曲项辅助变量 $q$ 为主线，其余辅助变量同构。

**时间演化。** 由辅助变量定义对时间求导（链式法则）即得。记 $\mathcal D(\cdot):=\dfrac{d(\cdot)}{dt}$，各辅助变量的时间导数为：

$$
\mathcal D q=a(\eta)\big(\epsilon\Delta\phi_t+g'(\phi)\,\phi_t\big)
$$

$$
\mathcal D r=\tfrac1r\big(\epsilon\,\nabla\phi\cdot\nabla\phi_t+\tfrac1\epsilon\phi(\phi^2-1)\,\phi_t\big)
$$

$$
\mathcal D s=\sqrt\epsilon\,\big(\nabla\phi_t\cdot\nabla\eta+\nabla\phi\cdot\nabla\eta_t\big)
$$

$$
\mathcal D p=\xi\,\nabla\eta\cdot\nabla\eta_t-\tfrac1\xi\eta(\eta^2-1)\,\eta_t
$$

每式右端对场的时间导数 $\phi_t,\eta_t$ 线性 —— 这是 IEQ 能将时间格式线性化的根本。其中 $s$ 同时依赖 $\phi$ 与 $\eta$，故其全导数含两项；解耦子步与 BDF2 离散在 §4.3 一步到位。

**变分重构。** §2 的变分导数对场是非线性的；引入辅助变量后，可将其改写成"对辅助变量线性的算子作用在该辅助变量上"。以弯曲能为例，由 §2.1，$\delta E/\delta\phi=\epsilon\Delta\big(a(\eta)^2(\epsilon\Delta\phi+g(\phi))\big)+a(\eta)^2g'(\phi)(\epsilon\Delta\phi+g(\phi))$；代入 $q=a(\eta)(\epsilon\Delta\phi+g(\phi))$（即 $a(\eta)(\epsilon\Delta\phi+g(\phi))=q$），改写为

$$
\frac{\delta E}{\delta\phi}=\mathcal G_q[q],\qquad
\mathcal G_q[h]:=\epsilon\Delta\big(a(\eta)\,h\big)+g'(\phi)\,a(\eta)\,h
$$

其中 $\mathcal G_q$ 对辅助变量 $q$ **线性**，系数 $a(\eta),g'(\phi)$ 为已知冻结场。非线性并未消失，而是被分流进了"$q$ 如何依赖 $\phi$"（即上面的演化 $\mathcal D q$）；$\mathcal G_q$ 只承担线性外壳。$r,s$ 的变分重构同理，在 §4 各项推导时按需给出。

---

## §4 时间格式：BDF2 一步 $(\phi^{n-1},\phi^n)\to\phi^{n+1}$

### §4.0 BDF2 离散

IEQ 方案确定取隐式构造：变分导数取在未知层 $n+1$。梯度流 $\phi_t=-\delta\mathcal E_M/\delta\phi$，时间导数用三层 BDF2 公式：

$$
\phi_t\Big|^{n+1}\approx\frac{3\phi^{n+1}-4\phi^n+\phi^{n-1}}{2\tau}=-\frac{\delta\mathcal E_M}{\delta\phi}\Big|^{n+1}
$$

增量化（$\delta^n\phi=\phi^{n+1}-\phi^n$，$\delta^{n-1}\phi=\phi^n-\phi^{n-1}$）：

$$
3\phi^{n+1}-4\phi^n+\phi^{n-1}=3(\phi^{n+1}-\phi^n)-(\phi^n-\phi^{n-1})=3\,\delta^n\phi-\delta^{n-1}\phi
$$

代入上式移项即得：

$$
\frac{3}{2\tau}\,\delta^n\phi=-\frac{\delta\mathcal E_M}{\delta\phi}\Big|^{n+1}+\frac{1}{2\tau}\,\delta^{n-1}\phi
$$

在 $\phi$-子步里，$\dfrac{\delta\mathcal E_M}{\delta\phi}\big|^{n+1}$ 这个表达式里同时含 $\phi^{n+1}$ 和 $\eta^{n+1}$：

- $\phi^{n+1}$ 是这一子步正在求的未知量，作为变量保留（再通过 IEQ 辅助变量线性化处理）；
- $\eta^{n+1}$ 在这一子步还没解出，没有值，于是用外推 $\eta^\star=2\eta^n-\eta^{n-1}$ 顶替它，不破坏 BDF2 精度）。
### §4.1 $\phi$-子步

固定 $\eta$。本节先以弯曲项为例完整推出其线性方程，其余项同构。

提示：$\phi$-子步中凡含 $\eta$ 的系数取外推 $\eta^\star=2\eta^n-\eta^{n-1}$、含 $\phi$ 的系数取 $\phi^\star=2\phi^n-\phi^{n-1}$（$=\,$对应 $n+1$ 层值 $+O(\tau^2)$），故 §3.1 的 $\mathcal D_q,\mathcal G_q$ 成为已知系数（$a(\eta^\star),g'(\phi^\star)$）的线性算子；唯一保留为新层未知的是辅助变量 $q^{n+1}$ 与场增量 $\delta^n\phi$。
#### 弯曲项

从 §4.0 的 BDF2 增量式出发，仅保留弯曲能贡献 $E=\tfrac12\|q\|^2$：

$$
\frac{3}{2\tau}\,\delta^n\phi=-\frac{\delta E}{\delta\phi}\Big|^{n+1}+\frac1{2\tau}\,\delta^{n-1}\phi
$$

**变分重构**（§3.1）：$\delta E/\delta\phi=\mathcal G_q[q]$，取在 $n+1$ 层（系数已外推冻结）即 $\delta E/\delta\phi|^{n+1}=\mathcal G_q[q^{n+1}]$，代入：

$$
\frac{3}{2\tau}\,\delta^n\phi=-\mathcal G_q[q^{n+1}]+\frac1{2\tau}\,\delta^{n-1}\phi
\tag{1}
$$

**$q^{n+1}$ 的 BDF2 递推**（§3.1 演化 $\mathcal D q$ 配 §4.0 同款 BDF2，§4.3 详述）：

$$
q^{n+1}=\tfrac13\big(4q^n-q^{n-1}\big)+\mathcal D_q[\delta^n\phi]-\tfrac13\mathcal D_q[\delta^{n-1}\phi]
\tag{2}
$$

右端仅 $\mathcal D_q[\delta^n\phi]$ 含未知 $\delta^n\phi$，其余 $\tfrac13(4q^n-q^{n-1})$ 与 $\tfrac13\mathcal D_q[\delta^{n-1}\phi]$ 全为旧层已知量。

**(2) 代入 (1)**，$\mathcal G_q$ 线性逐项作用：

$$
\frac{3}{2\tau}\,\delta^n\phi
=-\mathcal G_q\mathcal D_q[\delta^n\phi]
-\mathcal G_q\big[\tfrac13(4q^n-q^{n-1})\big]
+\tfrac13\mathcal G_q\mathcal D_q[\delta^{n-1}\phi]
+\frac1{2\tau}\,\delta^{n-1}\phi
$$

含 $\delta^n\phi$ 的项移左、已知项留右，得弯曲项的线性方程：

$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+\mathcal G_q\mathcal D_q\Big)\,\delta^n\phi
=-\mathcal G_q\big[\tfrac13(4q^n-q^{n-1})\big]
+\tfrac13\mathcal G_q\mathcal D_q[\delta^{n-1}\phi]
+\tfrac1{2\tau}\,\delta^{n-1}\phi
\;}
$$

左端隐式算子记 $\mathcal A_E:=\mathcal G_q\mathcal D_q$。

#### 线张力项

线张力能 $L$ 的 $\phi$-相关密度为 $\Phi_\eta W(\phi)$，其中 $W(\phi)=\tfrac12(r^2-C_0)$ 经辅助变量 $r$ 表达，$\Phi_\eta$ 为线张力能中与 $\eta$ 相关的密度因子（不含 $\phi$，在 $\phi$-子步为冻结权重）：

$$
\Phi_\eta:=\delta\Big(\tfrac\xi2|\nabla\eta|^2+\tfrac1{4\xi}(\eta^2-1)^2\Big)
$$

$\phi$-子步取外推 $\eta^\star$，记 $\Phi^\star_\eta=\delta\big(\tfrac\xi2|\nabla\eta^\star|^2+\tfrac1{4\xi}((\eta^\star)^2-1)^2\big)$。

**变分重构**（§3.1）：$\delta L/\delta\phi=\mathcal G_r[\Phi_\eta\,r]$，其中

$$
\mathcal D_r[\psi]:=\tfrac1r\big(\epsilon\nabla\phi\cdot\nabla\psi+\tfrac1\epsilon\phi(\phi^2-1)\psi\big),\qquad
\mathcal G_r[h]:=-\epsilon\,\nabla\!\cdot\!\big(\tfrac h r\nabla\phi\big)+\tfrac1{\epsilon r}\phi(\phi^2-1)\,h
$$

（系数 $r,\phi$ 取外推 $r^\star,\phi^\star$。$\mathcal G_r$ 中的 $1/r$ 与代入的 $h=\Phi_\eta r$ 之 $r$ 精确抵消，故变分重构本身不奇异；$1/r$ 的潜在奇异只在演化算子 $\mathcal D_r$ 中——此处 $C_0>0$ 保证 $r\ge\sqrt{C_0}$，避免界面外平坦区 $W\to0$ 致 $1/r$ 发散。）

BDF2 增量式（线张力部分），变分重构取 $n+1$ 层（$\Phi_\eta$ 冻结为 $\Phi^\star_\eta$）：

$$
\frac{3}{2\tau}\,\delta^n\phi=-\mathcal G_r[\Phi^\star_\eta\,r^{n+1}]+\frac1{2\tau}\,\delta^{n-1}\phi
\tag{1'}
$$

$r^{n+1}$ 的 BDF2 递推（§3.1 演化 $\mathcal D r$ 配 BDF2，§4.3 详述）：

$$
r^{n+1}=\tfrac13\big(4r^n-r^{n-1}\big)+\mathcal D_r[\delta^n\phi]-\tfrac13\mathcal D_r[\delta^{n-1}\phi]
\tag{2'}
$$

(2') 代入 (1')，$\mathcal G_r[\Phi^\star_\eta\,\cdot\,]$ 线性逐项作用，含 $\delta^n\phi$ 移左、已知留右，得线张力项的线性方程：

$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+\mathcal G_r\big[\Phi^\star_\eta\,\mathcal D_r\big]\Big)\,\delta^n\phi
=-\mathcal G_r\big[\Phi^\star_\eta\cdot\tfrac13(4r^n-r^{n-1})\big]
+\tfrac13\mathcal G_r\big[\Phi^\star_\eta\,\mathcal D_r[\delta^{n-1}\phi]\big]
+\tfrac1{2\tau}\,\delta^{n-1}\phi
\;}
$$

左端隐式算子记 $\mathcal A_L:=\mathcal G_r[\Phi^\star_\eta\,\mathcal D_r]$。

#### 惩罚项

##### Newton 线性化

罚项 $\tfrac{M_j}2(C_j-c_j)^2$（$C_j$ 为约束泛函 $V,A,D,N$，$c_j$ 为目标值）。其变分导数（一次求导，链式法则）：
$$
\frac{\delta}{\delta\phi}\Big[\tfrac{M_j}2(C_j-c_j)^2\Big]=M_j(C_j-c_j)\,\frac{\delta C_j}{\delta\phi}
$$

其中对于变量$\phi$, 在C和其变分中都出现。无法保证对这些项的变分是Linear的，因此为了IEQ构建线性方程，需要对每一项的变分做线性近似。

##### 面积 $A$

$A=\int_\Omega W(\phi)\,dx$，$W(\phi)=\tfrac\epsilon2|\nabla\phi|^2+\tfrac1{4\epsilon}(\phi^2-1)^2$，$\phi$ 非线性出现，故 $\mathrm d^2A\ne0$。

一阶变分（$W$ 对 $\phi$ 求变分，梯度项分部积分）：

$$
\frac{\delta A}{\delta\phi}=-\epsilon\Delta\phi+\tfrac1\epsilon\phi(\phi^2-1)
$$

二阶变分（对上式再求方向导数；$-\epsilon\Delta$ 线性、导数为自身，$\tfrac1\epsilon(\phi^3-\phi)$ 求导得 $\tfrac1\epsilon(3\phi^2-1)$）：

$$
\mathrm d^2A[\psi]=-\epsilon\Delta\psi+\tfrac1\epsilon(3\phi^2-1)\,\psi
$$

线性化即对精确梯度 $G(\phi):=M_2(A-a_0)\dfrac{\delta A}{\delta\phi}$ 在已知态 $\phi^\star$ 处作一阶泰勒展开，增量 $\Delta\phi:=\phi^{n+1}-\phi^\star$：

$$
G(\phi^{n+1})\approx G(\phi^\star)+G'(\phi^\star)[\Delta\phi]
$$

$G(\phi)$ 是两个含 $\phi$ 的因子之积：标量 $M_2(A-a_0)$ 与场 $\dfrac{\delta A}{\delta\phi}$。对它求导用乘积法则：

$$
G'[\psi]=M_2\underbrace{\Big(\tfrac{\delta A}{\delta\phi},\psi\Big)}_{\text{标量 }(A-a_0)\text{ 的方向导数}}\tfrac{\delta A}{\delta\phi}
+M_2(A-a_0)\underbrace{\mathrm d^2A[\psi]}_{\tfrac{\delta A}{\delta\phi}\text{ 的方向导数}}
$$

第一项来自标量因子求导：$\dfrac{d}{ds}A(\phi+s\psi)=\big(\tfrac{\delta A}{\delta\phi},\psi\big)$（$L^2$ 内积），乘上原场因子 $\dfrac{\delta A}{\delta\phi}$，给出 $\big(\tfrac{\delta A}{\delta\phi}\otimes\tfrac{\delta A}{\delta\phi}\big)[\psi]$ 型秩一主项；第二项来自场因子求导 $\mathrm d^2A[\psi]$，乘上标量 $(A-a_0)$，即二阶项。

精确梯度 $G(\phi)=M_2(A-a_0)\dfrac{\delta A}{\delta\phi}$ 的一阶泰勒展开（中心取外推 $\phi^\star$，保二阶）：

$$
G(\phi^{n+1})\approx\underbrace{M_2(A^\star-a_0)\big(\tfrac{\delta A}{\delta\phi}\big)^\star}_{\text{零阶项（精确负梯度）}}
+\underbrace{M_2\big(\tfrac{\delta A}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta A}{\delta\phi}\big)^\star[\phi^{n+1}-\phi^\star]}_{\text{主项, 秩一半正定}}
+\underbrace{M_2(A^\star-a_0)\,\mathrm d^2A^\star[\phi^{n+1}-\phi^\star]}_{\text{二阶项}}
$$

二阶项含变号标量 $(A^\star-a_0)$ 与定号 $\mathrm d^2A^\star$，符号不定、稳态为零，按 Gauss–Newton **整项丢弃**。

代入 BDF2 增量式（仅面积罚贡献），并用 $\phi^{n+1}-\phi^\star=\delta^n\phi-\delta^{n-1}\phi$：

$$
\frac{3}{2\tau}\,\delta^n\phi
=-M_2(A^\star-a_0)\big(\tfrac{\delta A}{\delta\phi}\big)^\star
-M_2\big(\tfrac{\delta A}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta A}{\delta\phi}\big)^\star[\delta^n\phi-\delta^{n-1}\phi]
+\frac1{2\tau}\,\delta^{n-1}\phi
$$

主项线性，拆开二阶差分、含 $\delta^n\phi$ 移左，得面积罚的线性方程：

$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+M_2\big(\tfrac{\delta A}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta A}{\delta\phi}\big)^\star\Big)\,\delta^n\phi
=-M_2(A^\star-a_0)\big(\tfrac{\delta A}{\delta\phi}\big)^\star
+M_2\big(\tfrac{\delta A}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta A}{\delta\phi}\big)^\star[\delta^{n-1}\phi]
+\tfrac1{2\tau}\,\delta^{n-1}\phi
\;}
$$

左端隐式算子记 $\mathcal A_A:=M_2\big(\tfrac{\delta A}{\delta\phi}\big)^\star\otimes\big(\tfrac{\delta A}{\delta\phi}\big)^\star$（秩一、半正定），其中

$$
\Big(\tfrac{\delta A}{\delta\phi}\Big)^\star=-\epsilon\Delta\phi^\star+\tfrac1\epsilon\phi^\star((\phi^\star)^2-1),\qquad
A^\star=\tfrac12\int_\Omega((r^\star)^2-C_0)\,dx
$$

##### 面积差 $D$

$$
D=\tfrac12\int_\Omega\tanh(\eta/\xi)(r^2-C_0)\,dx
$$

$$
\big(\tfrac{\delta D}{\delta\phi}\big)^\star=-\epsilon\nabla\!\cdot\!(\tanh(\eta^\star/\xi)\nabla\phi^\star)+\tfrac1\epsilon\phi^\star((\phi^\star)^2-1)\tanh(\eta^\star/\xi)
$$

$$
D^\star=\tfrac12\int_\Omega\tanh(\eta^\star/\xi)((r^\star)^2-C_0)\,dx
$$

Gauss–Newton 后线性方程：
$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+M_3\big(\tfrac{\delta D}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta D}{\delta\phi}\big)^\star\Big)\,\delta^n\phi
=-M_3(D^\star-a_d)\big(\tfrac{\delta D}{\delta\phi}\big)^\star
+M_3\big(\tfrac{\delta D}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta D}{\delta\phi}\big)^\star[\delta^{n-1}\phi]
+\tfrac1{2\tau}\,\delta^{n-1}\phi
\;}
$$

左端隐式算子记 $\mathcal A_D:=M_3\big(\tfrac{\delta D}{\delta\phi}\big)^\star\otimes\big(\tfrac{\delta D}{\delta\phi}\big)^\star$（秩一、半正定）。

##### 正交 $N$
$N=\tfrac12\|s\|^2$，$s=\sqrt\epsilon(\nabla\phi\cdot\nabla\eta)$。 $\big(\tfrac{\delta N}{\delta\phi}\big)^\star=-\epsilon\nabla\!\cdot\!((\nabla\phi^\star\cdot\nabla\eta^\star)\nabla\eta^\star)$， $N^\star=\tfrac12\|s^\star\|^2$（目标值 $0$）

Gauss–Newton 后线性方程：
$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+M_4\big(\tfrac{\delta N}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta N}{\delta\phi}\big)^\star\Big)\,\delta^n\phi
=-M_4 N^\star\big(\tfrac{\delta N}{\delta\phi}\big)^\star
+M_4\big(\tfrac{\delta N}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta N}{\delta\phi}\big)^\star[\delta^{n-1}\phi]
+\tfrac1{2\tau}\,\delta^{n-1}\phi
\;}
$$

左端隐式算子记 $\mathcal A_N:=M_4\big(\tfrac{\delta N}{\delta\phi}\big)^\star\otimes\big(\tfrac{\delta N}{\delta\phi}\big)^\star$（秩一、半正定）。

##### 体积 $V$

$V=\int_\Omega\phi\,dx$ 为线性泛函，$\tfrac{\delta V}{\delta\phi}=\mathbf 1$（常值场），$\mathrm d^2V=0$，二阶项天然为零、无需丢弃。
$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+M_1\,\mathbf 1\!\otimes\!\mathbf 1\Big)\,\delta^n\phi
=-M_1(V^\star-v_d)\,\mathbf 1
+M_1\,\mathbf 1\!\otimes\!\mathbf 1[\delta^{n-1}\phi]
+\tfrac1{2\tau}\,\delta^{n-1}\phi
\;}
$$

左端隐式算子记 $\mathcal A_V:=M_1\,\mathbf 1\otimes\mathbf 1$（秩一、半正定）。

#### 总线性方程

##### Eyre

**目的**：补一个无条件正定的 $\sigma\mathcal I$，使左端对任意 $\tau$ 正定可逆（允许大步、保稳定）。

**操作**：方程加入 $\sigma(\delta^n\phi-\delta^{n-1}\phi)$（$\sigma>0$）：$\sigma\delta^n\phi$ 入左端（即 $\sigma\mathcal I$），$-\sigma\delta^{n-1}\phi$ 入右端。该项作用在二阶差分 $\delta^n\phi-\delta^{n-1}\phi=O(\tau^2)$ 上，不降精度；稳态时为零，不改不动点。

##### IEQ 线性方程

各项贡献相加（各项小方程中的 $\tfrac{3}{2\tau}\mathcal I$ 是同一时间项，汇总只计一次、不叠加），$\delta^n\phi$ 满足线性方程

$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+\sigma\mathcal I
+\mathcal A_E+\mathcal A_L+\mathcal A_A+\mathcal A_D+\mathcal A_N+\mathcal A_V\Big)\,\delta^n\phi=\mathcal R^n_\phi
\;}
$$

各隐式算子：

$$
\begin{aligned}
\mathcal A_E &= \mathcal G_q\mathcal D_q && \text{弯曲}\\
\mathcal A_L &= \mathcal G_r\big[\Phi^\star_\eta\,\mathcal D_r\big] && \text{线张力}\\
\mathcal A_A &= M_2\big(\tfrac{\delta A}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta A}{\delta\phi}\big)^\star && \text{面积罚，秩一}\\
\mathcal A_D &= M_3\big(\tfrac{\delta D}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta D}{\delta\phi}\big)^\star && \text{面积差罚，秩一}\\
\mathcal A_N &= M_4\big(\tfrac{\delta N}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta N}{\delta\phi}\big)^\star && \text{正交罚，秩一}\\
\mathcal A_V &= M_1\,\mathbf 1\!\otimes\!\mathbf 1 && \text{体积罚，秩一}
\end{aligned}
$$

**右端 $\mathcal R^n_\phi$。** 由各项回代后留在右端的已知量收集而成。逐项列出：

弯曲、线张力（来自式 (2)、(2′) 中的已知部分）：

$$
\mathcal R^n_{\phi,E}=-\mathcal G_q\big[\tfrac13(4q^n-q^{n-1})\big]+\tfrac13\mathcal G_q\mathcal D_q[\delta^{n-1}\phi]
$$

$$
\mathcal R^n_{\phi,L}=-\mathcal G_r\big[\Phi^\star_\eta\cdot\tfrac13(4r^n-r^{n-1})\big]+\tfrac13\mathcal G_r\big[\Phi^\star_\eta\,\mathcal D_r[\delta^{n-1}\phi]\big]
$$

罚项（每项 = 精确负梯度 + 主项二阶差分的已知半 $\delta^{n-1}\phi$ 部分），$\mathcal R^n_{\phi,P}=\mathcal R^n_{\phi,V}+\mathcal R^n_{\phi,A}+\mathcal R^n_{\phi,D}+\mathcal R^n_{\phi,N}$：

$$
\mathcal R^n_{\phi,V}=-M_1(V^\star-v_d)\,\mathbf 1+M_1\,\mathbf 1\!\otimes\!\mathbf 1[\delta^{n-1}\phi]
$$

$$
\mathcal R^n_{\phi,A}=-M_2(A^\star-a_0)\big(\tfrac{\delta A}{\delta\phi}\big)^\star+M_2\big(\tfrac{\delta A}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta A}{\delta\phi}\big)^\star[\delta^{n-1}\phi]
$$

$$
\mathcal R^n_{\phi,D}=-M_3(D^\star-a_d)\big(\tfrac{\delta D}{\delta\phi}\big)^\star+M_3\big(\tfrac{\delta D}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta D}{\delta\phi}\big)^\star[\delta^{n-1}\phi]
$$

$$
\mathcal R^n_{\phi,N}=-M_4 N^\star\big(\tfrac{\delta N}{\delta\phi}\big)^\star+M_4\big(\tfrac{\delta N}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta N}{\delta\phi}\big)^\star[\delta^{n-1}\phi]
$$

Eyre 稳定项与 BDF2 历史项：

$$
\mathcal R^n_{\phi,\sigma}=\sigma\,\delta^{n-1}\phi,\qquad
\mathcal R^n_{\phi,t}=\tfrac1{2\tau}\,\delta^{n-1}\phi
$$

总右端 $\mathcal R^n_\phi=\mathcal R^n_{\phi,E}+\mathcal R^n_{\phi,L}+\mathcal R^n_{\phi,P}+\mathcal R^n_{\phi,\sigma}+\mathcal R^n_{\phi,t}$，全部为旧层 $\{n,n-1\}$ 的已知量。其中罚项精确梯度密度（外推层）：

$$
\Big(\tfrac{\delta A}{\delta\phi}\Big)^\star=-\epsilon\Delta\phi^\star+\tfrac1\epsilon\big((\phi^\star)^3-\phi^\star\big)
$$

$$
\Big(\tfrac{\delta D}{\delta\phi}\Big)^\star=-\epsilon\nabla\!\cdot\!\big(\tanh(\eta^\star/\xi)\nabla\phi^\star\big)
+\tfrac1\epsilon\big((\phi^\star)^3-\phi^\star\big)\tanh(\eta^\star/\xi)
$$

$$
\Big(\tfrac{\delta N}{\delta\phi}\Big)^\star=-\epsilon\nabla\!\cdot\!\big((\nabla\phi^\star\cdot\nabla\eta^\star)\nabla\eta^\star\big)
$$

### §4.2 $\eta$-子步

固定 $\phi=\phi^{n+1}$（先更新 $\phi$ 再更新 $\eta$，故 $\eta$-子步中所有 $\phi$ 取新值 $\phi^{n+1}$，$\eta$ 相关系数取外推 $\eta^\star$，$\zeta^\star=(\eta^\star)^2-1$）。BDF2 增量式：

$$
\frac{3}{2\tau}\,\delta^n\eta=-\frac{\delta\mathcal E_M}{\delta\eta}\Big|^{n+1}+\frac1{2\tau}\,\delta^{n-1}\eta
$$

弯曲项与 $P$ 项的处理与 $\phi$-子步有明显差异，单独说明；线张力 $L$、面积差 $D$、正交 $N$ 与 §4.1 同构，直接列入总方程。

#### 弯曲项

$E=\tfrac12\|q\|^2$，$\phi$ 冻结使 $q$ 已知，$\eta$ 只在系数 $k(\eta)$ 中。由 §2.2：

$$
\frac{\delta E}{\delta\eta}=\frac{k'(\eta)}{2k(\eta)}q^2
$$

取 $n+1$ 层（$\eta\to\eta^\star$、$q\to q^{n+1}$ 均已知），整块为已知源项，直接入右端：

$$
\frac{3}{2\tau}\,\delta^n\eta=-\underbrace{\frac{k'(\eta^\star)}{2k(\eta^\star)}(q^{n+1})^2}_{\text{已知，入右端}}+\frac1{2\tau}\,\delta^{n-1}\eta+\cdots
$$

无未知 $\delta^n\eta$，故弯曲项不贡献左端隐式算子（无 $\mathcal B_E$）。

#### $P$ 项

$P=\|p\|^2$（无 $\tfrac12$），$p=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2$，罚项 $\tfrac{M_5}2P^2$ 对 $p$ 四次。演化与重构算子（§3.1）：

$$
\mathcal D_p[\psi]:=\xi\nabla\eta^\star\cdot\nabla\psi-\tfrac1\xi\eta^\star\zeta^\star\psi,\qquad
\mathcal G_p[h]:=-\xi\nabla\!\cdot\!(h\nabla\eta^\star)-\tfrac1\xi\eta^\star\zeta^\star h
$$

梯度 $\dfrac{\delta}{\delta\eta}\big[\tfrac{M_5}2P^2\big]=M_5P\,\dfrac{\delta P}{\delta\eta}$，其中 $\dfrac{\delta P}{\delta\eta}=2\mathcal G_p[p]$（$P$ 无 $\tfrac12$，故系数 $2$），即梯度 $=2M_5P\,\mathcal G_p[p]$。Newton 主项（乘积法则，标量因子求导再带出一个 $\delta P/\delta\eta=2\mathcal G_p[p]$）：

$$
2M_5\big(2\mathcal G_p[p],\psi\big)\mathcal G_p[p]=4M_5\,\big(\mathcal G_p[p]\otimes\mathcal G_p[p]\big)[\psi]
$$

取外推 $p^\star$、作用在二阶差分 $\delta^n\eta-\delta^{n-1}\eta$ 上，含 $\delta^n\eta$ 入左端：

$$
\boxed{\;\mathcal B_P=4M_5\,\mathcal G_p[p^\star]\otimes\mathcal G_p[p^\star]\;}
$$

（系数 $4M_5$ 中的 $4=2^2$ 源自 $P=\|p\|^2$ 不带 $\tfrac12$；对照正交项 $N=\tfrac12\|s\|^2$ 带 $\tfrac12$，其主项系数为 $M_4$，无 $4$。）精确负梯度 $-2M_5P^\star\mathcal G_p[p^\star]$ 与主项的 $\delta^{n-1}\eta$ 半入右端；二阶项含 $P^\star\ge0$ 与二阶变分，按 Gauss–Newton 丢弃。

#### 总线性方程

各项相加（$\tfrac{3}{2\tau}\mathcal I$ 只计一次），$\delta^n\eta$ 满足

$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+\sigma\mathcal I+\mathcal B_L+\mathcal B_D+\mathcal B_N+\mathcal B_P\Big)\,\delta^n\eta=\mathcal R^n_\eta
\;}
$$

各隐式算子（$\mathcal B_L$ 微分型；$\mathcal B_{D,N,P}$ 秩一）：

$$
\begin{aligned}
\mathcal B_L[\psi]&=-\delta\xi\,\nabla\!\cdot\!\big(W(\phi^{n+1})\nabla\psi\big)+\delta\tfrac1\xi W(\phi^{n+1})\,\zeta^\star\,\psi && \text{线张力}\\
\mathcal B_D&=M_3\,f^\star_D\!\otimes\!f^\star_D, & f^\star_D&=\tfrac1\xi W(\phi^{n+1})\,\mathrm{sech}^2(\eta^\star/\xi) && \text{面积差}\\
\mathcal B_N&=M_4\,\big(\mathcal G_{s\eta}[s^{n+1}]\big)\!\otimes\!\big(\mathcal G_{s\eta}[s^{n+1}]\big), & \mathcal G_{s\eta}[h]&=-\sqrt\epsilon\,\nabla\!\cdot\!(h\,\nabla\phi^{n+1}) && \text{正交}\\
\mathcal B_P&=4M_5\,\mathcal G_p[p^\star]\!\otimes\!\mathcal G_p[p^\star] && \text{剖面}
\end{aligned}
$$

其中 $\mathcal B_N$ 的 $s^{n+1}$ 取 $\phi=\phi^{n+1}$、$\eta=\eta^\star$。

右端：

$$
\mathcal R^n_\eta=-\frac{k'(\eta^\star)}{2k(\eta^\star)}(q^{n+1})^2
+\mathcal R^n_{\eta,L}+\mathcal R^n_{\eta,D}+\mathcal R^n_{\eta,N}+\mathcal R^n_{\eta,P}
+\big(\sigma+\tfrac1{2\tau}\big)\delta^{n-1}\eta
$$

其中：

$$
\mathcal R^n_{\eta,L}=-\Big(-\delta\xi\nabla\!\cdot\!\big(W(\phi^{n+1})\nabla\eta^\star\big)+\delta\tfrac1\xi W(\phi^{n+1})\,\eta^\star\zeta^\star\Big)+\mathcal B_L[\delta^{n-1}\eta]
$$

$$
\mathcal R^n_{\eta,D}=-M_3(D^\star-a_d)\tfrac1\xi W(\phi^{n+1})\,\mathrm{sech}^2(\eta^\star/\xi)+M_3\,f^\star_D\!\otimes\!f^\star_D[\delta^{n-1}\eta]
$$

$$
\mathcal R^n_{\eta,N}=M_4 N^\star\,\epsilon\nabla\!\cdot\!\big((\nabla\phi^{n+1}\cdot\nabla\eta^\star)\nabla\phi^{n+1}\big)+M_4\,\mathcal G_{s\eta}[s^{n+1}]\!\otimes\!\mathcal G_{s\eta}[s^{n+1}][\delta^{n-1}\eta]
$$

$$
\mathcal R^n_{\eta,P}=-2M_5 P^\star\,\mathcal G_p[p^\star]+4M_5\,\mathcal G_p[p^\star]\!\otimes\!\mathcal G_p[p^\star][\delta^{n-1}\eta]
$$

其中 $f^\star_D=\tfrac1\xi W(\phi^{n+1})\,\mathrm{sech}^2(\eta^\star/\xi)$，$\mathcal G_{s\eta}[h]=-\sqrt\epsilon\,\nabla\!\cdot\!(h\,\nabla\phi^{n+1})$。

$\mathcal B_L$ 对称，但零阶系数 $\delta\tfrac1\xi W(\phi^{n+1})\zeta^\star$ 在 $|\eta^\star|<1$（区域大部分）处为负，故 $\mathcal B_L$ **非半正定**；左端整体正定性需 $\sigma$ 足够大保证（阈值见 §5.2）

### §4.3 辅助变量的 BDF2 推进

辅助变量按 BDF2 三层公式推进，与场同为二阶。推进时机由各辅助变量依赖的场决定：$r$ 只含 $\phi$、$p$ 只含 $\eta$，各更新一次；$q$ 随 $\phi$ 更新一次；$s=\sqrt\epsilon(\nabla\phi\cdot\nabla\eta)$ 同时含 $\phi,\eta$，故在两个子步各更新一次。

#### 更新顺序

一个完整时间步：

```
φ-子步: 解 δⁿφ ⇒ φⁿ⁺¹
  └ 用 δⁿφ 更新: qⁿ⁺¹, rⁿ⁺¹, 以及 s 的 φ 贡献 (得中间值 s_(φ))
η-子步: 解 δⁿη ⇒ ηⁿ⁺¹
  └ 用 δⁿη 更新: pⁿ⁺¹, 以及 s 的 η 贡献 (由 s_(φ) 得 sⁿ⁺¹)
```

$s$ 必须两步都推进：只推进一次会缺失另一场变化的贡献，使正交量 $N=\tfrac12\|s\|^2$ 失准。

#### 递推表达式

通式：辅助变量 $f$ 与其所依赖场增量 $\delta^n(\cdot)$ 的 BDF2 推进为 $3f^{n+1}-4f^n+f^{n-1}=\tfrac{df}{dt}\big[3\delta^n(\cdot)-\delta^{n-1}(\cdot)\big]$，即 $f^{n+1}=\tfrac13\big(4f^n-f^{n-1}+\tfrac{df}{dt}[3\delta^n(\cdot)-\delta^{n-1}(\cdot)]\big)$，$\tfrac{df}{dt}$ 取 §3.1 演化算子（系数外推）。逐个写出：

**$q$（$\phi$-子步，用 $\delta^n\phi$）：**

$$
\boxed{\;
q^{n+1}=\tfrac13\Big(4q^n-q^{n-1}+\mathcal D_q\big[3\delta^n\phi-\delta^{n-1}\phi\big]\Big)
\;}
$$

**$r$（$\phi$-子步，用 $\delta^n\phi$）：**

$$
\boxed{\;
r^{n+1}=\tfrac13\Big(4r^n-r^{n-1}+\mathcal D_r\big[3\delta^n\phi-\delta^{n-1}\phi\big]\Big)
\;}
$$

**$p$（$\eta$-子步，用 $\delta^n\eta$）：**

$$
\boxed{\;
p^{n+1}=\tfrac13\Big(4p^n-p^{n-1}+\mathcal D_p\big[3\delta^n\eta-\delta^{n-1}\eta\big]\Big)
\;}
$$

**$s$（两步推进）。** 先在 $\phi$-子步用 $\delta^n\phi$ 推进 $\phi$ 贡献（$\mathcal D_{s\phi}[\psi]=\sqrt\epsilon\,\nabla\psi\cdot\nabla\eta^\star$）：

$$
\boxed{\;
s_{(\phi)}=\tfrac13\Big(4s^n-s^{n-1}+\mathcal D_{s\phi}\big[3\delta^n\phi-\delta^{n-1}\phi\big]\Big)
\;}
$$

再在 $\eta$-子步用 $\delta^n\eta$ 推进 $\eta$ 贡献（$\mathcal D_{s\eta}[\psi]=\sqrt\epsilon\,\nabla\phi^{n+1}\cdot\nabla\psi$，$\phi$ 取新值）：

$$
\boxed{\;
s^{n+1}=s_{(\phi)}+\tfrac13\mathcal D_{s\eta}\big[3\delta^n\eta-\delta^{n-1}\eta\big]
\;}
$$

---

## §5 谱空间求解

### §5.1 FFT 空间离散

$$
\mathcal L\,\delta^n\phi=\mathcal R^n_\phi,\qquad
\mathcal M\,\delta^n\eta=\mathcal R^n_\eta,
$$

线性即原则上可解。§5 要解决的不是可解性，而是如何在每个时间步**高效**解出。周期边界下，场按 Fourier 基展开：

$$
\phi(x)=\sum_{\mathbf k}\widehat\phi_{\mathbf k}\,e^{\,i\mathbf k\cdot x},\qquad
\mathbf k\in\tfrac{2\pi}{L}\mathbb Z^3
$$

微分作用在 $e^{i\mathbf k\cdot x}$ 上化为乘子：

$$
\nabla\,e^{\,i\mathbf k\cdot x}=i\mathbf k\,e^{\,i\mathbf k\cdot x},\qquad
\Delta\,e^{\,i\mathbf k\cdot x}=-|\mathbf k|^2\,e^{\,i\mathbf k\cdot x}
$$

举例：常系数方程 $(\tfrac3{2\tau}-\epsilon\Delta)\phi=B$，两边取 Fourier 系数，$\Delta$ 换成 $-|\mathbf k|^2$，每个波数得一个**标量**方程

$$
\Big(\tfrac3{2\tau}+\epsilon|\mathbf k|^2\Big)\widehat\phi_{\mathbf k}=\widehat B_{\mathbf k}
\quad\Longrightarrow\quad
\widehat\phi_{\mathbf k}=\frac{\widehat B_{\mathbf k}}{\tfrac3{2\tau}+\epsilon|\mathbf k|^2}
$$

**左端按可解性分两类。** $\mathcal L,\mathcal M$ 由 §4 诸项加和，逐项考察：

- **微分项**：时间项 $\tfrac3{2\tau}\mathcal I$、Eyre $\sigma\mathcal I$、弯曲 $\mathcal A_E$、线张力 $\mathcal A_L$（$\phi$-子步）与 $\mathcal B_L$（$\eta$-子步）。常系数部分在谱空间为乘子、可逐 $\mathbf k$ 除；变系数部分对应卷积、波数耦合，以空间平均取常系数骨架入左端、偏差外推滞后入右端（§5.2）。
- **秩一罚项**：$\mathcal A_{A,D,N,V}$、$\mathcal B_{D,N,P}$，形如 $u\otimes u$ 的低秩矩阵，采用 Woodbury 处理（§5.3）。

两类合成 §5.4 子步流程。

### §5.2 劈分近似求解

#### 弯曲能方程

弯曲项线性方程，记右端已知量为 $\mathcal R^n_{\phi,E}$：

$$
\Big(\tfrac3{2\tau}+\sigma+\mathcal G_q\mathcal D_q\Big)\delta^n\phi=\mathcal R^n_{\phi,E}
$$

弯曲算子 $\mathcal G_q\mathcal D_q$ 含外推冻结的变系数 $a(\eta^\star),g'(\phi^\star)$：

$$
\mathcal D_q[\psi]=a(\eta^\star)\big(\epsilon\Delta\psi+g'(\phi^\star)\psi\big),\qquad
\mathcal G_q[h]=\epsilon\Delta\big(a(\eta^\star)h\big)+g'(\phi^\star)a(\eta^\star)h
$$

复合后逐点形式：

$$
\mathcal G_q\mathcal D_q\,\delta^n\phi
=\epsilon\Delta\Big(a(\eta^\star)^2\big(\epsilon\Delta\,\delta^n\phi+g'(\phi^\star)\delta^n\phi\big)\Big)
+g'(\phi^\star)a(\eta^\star)^2\big(\epsilon\Delta\,\delta^n\phi+g'(\phi^\star)\delta^n\phi\big)
$$

取 FFT，$\Delta\to-|\mathbf k|^2$，逐点积化为卷积 $\widehat{fg}_{\mathbf k}=\sum_{\mathbf k'}\widehat f_{\mathbf k-\mathbf k'}\widehat g_{\mathbf k'}$。记 $u=\widehat{\delta^n\phi}$、$\widehat{a^2}=\widehat{a(\eta^\star)^2}$、$\widehat{g'}=\widehat{g'(\phi^\star)}$，从内层向外逐层代：

内括号 $\epsilon\Delta\,\delta^n\phi+g'(\phi^\star)\delta^n\phi$：

$$
-\epsilon|\mathbf k|^2 u_{\mathbf k}+\sum_{\mathbf k_1}\widehat{g'}_{\mathbf k-\mathbf k_1}u_{\mathbf k_1}
$$

乘 $a(\eta^\star)^2$：

$$
C_{\mathbf k}=\sum_{\mathbf k_2}\widehat{a^2}_{\mathbf k-\mathbf k_2}\Big(-\epsilon|\mathbf k_2|^2 u_{\mathbf k_2}+\sum_{\mathbf k_1}\widehat{g'}_{\mathbf k_2-\mathbf k_1}u_{\mathbf k_1}\Big)
$$

外层 $\epsilon\Delta(\cdots)+g'(\phi^\star)(\cdots)$ 作用在 $C_{\mathbf k}$ 上，整方程在 FFT 下合写为

$$
\boxed{\;
\underbrace{\Big(\tfrac3{2\tau}+\sigma\Big)u_{\mathbf k}}_{\text{常数项，乘子}}
\;\underbrace{-\,\epsilon|\mathbf k|^2 C_{\mathbf k}+\sum_{\mathbf k_3}\widehat{g'}_{\mathbf k-\mathbf k_3}C_{\mathbf k_3}}_{\mathcal G_q\mathcal D_q\,\delta^n\phi}
=\widehat{\mathcal R^n_{\phi,E}}_{\mathbf k}
\;}
$$

第一项是干净乘子 $(\tfrac3{2\tau}+\sigma)u_{\mathbf k}$。第二项（$\mathcal G_q\mathcal D_q$）中 $\widehat{a^2},\widehat{g'}$ 的卷积令 $\mathbf k_1,\mathbf k_2,\mathbf k_3$ 跑遍全空间，第 $\mathbf k$ 行把全体 $\{u_{\mathbf k'}\}$ 线性混在一起，提不出公因子 $u_{\mathbf k}$，故不能逐 $\mathbf k$ 除。这里提出劈分近似方案：对变系数算子 $\mathcal O=\mathcal G_q\mathcal D_q$，拆常系数骨架 + 余项，余项二阶外推滞后：

$$
\mathcal O[\phi^{n+1}]=\mathcal S[\phi^{n+1}]+(\mathcal O-\mathcal S)[\phi^{n+1}]
\approx\mathcal S[\phi^{n+1}]+(\mathcal O-\mathcal S)\big[\mathrm{Ext}[\phi^n,\phi^{n-1}]\big]
$$

骨架取空间平均 $\bar a^2=\overline{k(\eta^\star)}/\epsilon$、$\bar g'=\overline{g'(\phi^\star)}$，左端算子连续化简：

$$
\begin{aligned}
\tfrac3{2\tau}+\sigma+\mathcal G_q\mathcal D_q
&=\tfrac3{2\tau}+\sigma+\mathcal G_q\big[a(\eta^\star)(\epsilon\Delta+g'(\phi^\star))\big]\\
&=\tfrac3{2\tau}+\sigma+\epsilon\Delta\big(a(\eta^\star)^2(\epsilon\Delta+g'(\phi^\star))\big)+g'(\phi^\star)a(\eta^\star)^2(\epsilon\Delta+g'(\phi^\star))\\
&\xrightarrow{a\to\bar a,\,g'\to\bar g'}\;\tfrac3{2\tau}+\sigma+\bar a^2(\epsilon\Delta+\bar g')^2\\
&\xrightarrow{\;\rm FFT,\ \Delta\to-|\mathbf k|^2\;}\;\tfrac3{2\tau}+\sigma+\bar a^2(\epsilon|\mathbf k|^2-\bar g')^2
\end{aligned}
$$

骨架入左端、余项入右端，逐 $\mathbf k$ 除：

$$
\boxed{\;
\Big(\tfrac3{2\tau}+\sigma+\bar a^2(\epsilon|\mathbf k|^2-\bar g')^2\Big)\widehat{\delta^n\phi}_{\mathbf k}
=\widehat{\mathcal R^n_{\phi,E}}_{\mathbf k}-\widehat{\big(\mathcal G_q\mathcal D_q-\bar a^2(\epsilon\Delta+\bar g')^2\big)\delta^{n-1}\phi}_{\mathbf k}
\;}
$$

记

$$
\widehat{\mathcal L}_E(\mathbf k):=\tfrac3{2\tau}+\sigma+\bar a^2(\epsilon|\mathbf k|^2-\bar g')^2,\qquad
\widehat{\mathcal R}_{\mathbf k}:=\widehat{\mathcal R^n_{\phi,E}}_{\mathbf k}-\widehat{\big(\mathcal G_q\mathcal D_q-\bar a^2(\epsilon\Delta+\bar g')^2\big)\delta^{n-1}\phi}_{\mathbf k}
$$

则 $\widehat{\delta^n\phi}_{\mathbf k}=\widehat{\mathcal R}_{\mathbf k}/\widehat{\mathcal L}_E(\mathbf k)$。

#### 线张力方程

模仿弯曲项，直接给出劈分后的方程。

**$\phi$-子步**（$\mathcal A_L=\mathcal G_r[\Phi^\star_\eta\mathcal D_r]$）：
$$
\boxed{\;
\Big(\tfrac3{2\tau}+\sigma+b_L|\mathbf k|^2\Big)\widehat{\delta^n\phi}_{\mathbf k}
=\widehat{\mathcal R^n_{\phi,L}}_{\mathbf k}-\widehat{\big(\mathcal A_L-b_L(-\Delta)\big)\delta^{n-1}\phi}_{\mathbf k}
\;}
$$

$$
b_L=\epsilon^2\,\overline{\Phi^\star_\eta\,|\nabla\phi^\star|^2/(r^\star)^2}
$$

其中 $1/(r^\star)^2$ 由 §3 的 $C_0>0$ 保证 $r^\star\ge\sqrt{C_0}$，界面外平坦区 $W\to0$ 不致发散。

**$\eta$-子步**（$\mathcal B_L=-\delta\xi\nabla\!\cdot\!(W(\phi^{n+1})\nabla\cdot)+\delta\tfrac1\xi W(\phi^{n+1})\zeta^\star\cdot$）：

$$
\boxed{\;
\Big(\tfrac3{2\tau}+\sigma+\delta\xi\bar W|\mathbf k|^2+\delta\tfrac1\xi\bar W\bar\zeta\Big)\widehat{\delta^n\eta}_{\mathbf k}
=\widehat{\mathcal R^n_{\eta,L}}_{\mathbf k}-\widehat{\big(\mathcal B_L-(-\delta\xi\bar W\Delta+\delta\tfrac1\xi\bar W\bar\zeta)\big)\delta^{n-1}\eta}_{\mathbf k}
\;}
$$

$$
\bar W=\overline{W(\phi^{n+1})},\qquad\bar\zeta=\overline{(\eta^\star)^2-1}
$$

零阶项 $\delta\tfrac1\xi\bar W\bar\zeta$ 在 $\bar\zeta<0$（$|\eta^\star|<1$，区域大部）时为负，左端正性需 $\sigma>\delta\bar W|\bar\zeta|/\xi-\tfrac3{2\tau}$。

### §5.3 秩一罚项：Woodbury

罚项算子 $\sum_j M_j\,u_j^\star\otimes u_j^\star$（$u_j^\star\in\{(\tfrac{\delta A}{\delta\phi})^\star,(\tfrac{\delta D}{\delta\phi})^\star,(\tfrac{\delta N}{\delta\phi})^\star,\mathbf 1\}$）作用在 $\delta^n\phi$ 上：

$$
\Big(\sum_j M_j\,u_j^\star\otimes u_j^\star\Big)\delta^n\phi
$$

为低秩矩阵，以 Woodbury 公式精确求逆（$\eta$-子步同构，秩一项为 $\mathcal B_{D,N,P}$）。

### §5.4 求解流程（$\phi$-子步，$\eta$-子步同构）

两类合并，$\phi$-子步谱空间完整方程：

$$
\boxed{\;
\begin{aligned}
&\widehat{\mathcal L}(\mathbf k)\,\widehat{\delta^n\phi}_{\mathbf k}
+\Big[\Big(\sum_j M_j\,u_j^\star\otimes u_j^\star\Big)\delta^n\phi\Big]_{\mathbf k}
=\widehat{\mathcal R^n_\phi}_{\mathbf k}\\[4pt]
&\widehat{\mathcal L}(\mathbf k)=\tfrac3{2\tau}+\sigma+\bar a^2(\epsilon|\mathbf k|^2-\bar g')^2+b_L|\mathbf k|^2
\end{aligned}
\;}
$$

其中 $u_j^\star$ 取 $\{(\tfrac{\delta A}{\delta\phi})^\star,(\tfrac{\delta D}{\delta\phi})^\star,(\tfrac{\delta N}{\delta\phi})^\star,\mathbf 1\}$，系数 $M_j$ 依次为 $M_2,M_3,M_4,M_1$。

$\eta$-子步谱空间完整方程：

$$
\boxed{\;
\begin{aligned}
&\widehat{\mathcal M}(\mathbf k)\,\widehat{\delta^n\eta}_{\mathbf k}
+\Big[\big(M_3\,f^\star_D\otimes f^\star_D+M_4\,\mathcal G_{s\eta}[s^{n+1}]\otimes\mathcal G_{s\eta}[s^{n+1}]+4M_5\,\mathcal G_p[p^\star]\otimes\mathcal G_p[p^\star]\big)\delta^n\eta\Big]_{\mathbf k}
=\widehat{\mathcal R^n_\eta}_{\mathbf k}\\[4pt]
&\widehat{\mathcal M}(\mathbf k)=\tfrac3{2\tau}+\sigma+\delta\xi\bar W|\mathbf k|^2+\delta\tfrac1\xi\bar W\bar\zeta
\end{aligned}
\;}
$$

两子步左端均为“对角 $\widehat{\mathcal L}$ 或 $\widehat{\mathcal M}$ + 低秩 $\sum M_j u_j^\star\otimes u_j^\star$”：对角部分逐 $\mathbf k$ 除、秩一部分 Woodbury 修正，即精确解出。

```
1. 二阶外推: φ★=2φⁿ−φⁿ⁻¹, η★=2ηⁿ−ηⁿ⁻¹; q★,r★,s★,p★ 同
2. 实空间组装右端 R:
     精确负梯度 −(δE_M/δfield)★
     + BDF2 历史 (1/2τ)δⁿ⁻¹field + Eyre σ·δⁿ⁻¹field
     + 变系数余项 −(O−骨架)[δⁿ⁻¹field]   (弯曲/张力)
3. FFT(R)
4. 逐波数除骨架符号 L̂(k)   (η-子步除 M̂(k))
5. 对全部秩一项作 Woodbury 修正 (§5.3)
6. IFFT  ⇒ δⁿφ
7. φⁿ⁺¹ = φⁿ + δⁿφ
8. 按 §4.3 BDF2 推进 qⁿ⁺¹,rⁿ⁺¹,sⁿ⁺¹（需 qⁿ,qⁿ⁻¹ 两层）
   注: 此处 sⁿ⁺¹ 仅完成 φ 贡献(中间值), η 贡献待 η-子步补齐(§4.3)
```

变系数余项以二阶外推估值，滞后误差 $O(\tau^2)$；若 $\tau$ 较大使外推不足，可以骨架符号之逆 $\widehat{\mathcal L}^{-1}$（或 $\widehat{\mathcal M}^{-1}$）为预条件子，对含余项的完整方程作不动点或 CG 迭代细化，将余项收敛至精确。

---

## §6 罚因子 continuation

$M_i$ 取大则约束精确但系统刚性强、$\tau$ 受限。采用 continuation 策略：$M_i$ 从小值起，每完成一轮内层迭代（至稳态）后放大 $M_i\leftarrow\rho M_i$（$\rho\sim2$–$10$），并以当前解暖启动重启，直至约束残差 $|V-v_d|,|A-a_0|,|D-a_d|$ 与正则量 $N,P$ 均小于容差。

该策略为罚方法（quadratic penalty method）与 continuation / homotopy method 的标准组合，合理性参考 Nocedal & Wright, *Numerical Optimization*, 2nd ed., Springer 2006（Ch. 17 罚方法、Ch. 11 continuation）与 Allgower & Georg, *Numerical Continuation Methods*, Springer 1990；本模型罚因子与 Lagrange 乘子的对应关系参考 Wang & Du, *J. Math. Biol.* 56 (2008) 347–371, §2。

$\rho$、$M$ 初值、目标容差均为经验量，需实验标定（$\rho$ 大则级数短但每步刚性跳变大）；BDF2 版每次放大 $M$ 后时间历史 $\phi^{n-1}$ 失效，须按 §7 重做 startup。

---

## §7 完整算法

```
输入: φ⁰, η⁰; ε=ξ, δ, c₁=c₂=0; v_d, a₀, a_d;
      M₁..M₅; τ; σ; C₀; tol; k,c (k>|c|>0); ρ, M_max

初始化辅助变量（按 §3 定义直接代入 φ⁰,η⁰ 计算）:
  q⁰ = a(η⁰)(εΔφ⁰ + g(φ⁰))
  r⁰ = √(2W(φ⁰) + C₀)
  s⁰ = √ε (∇φ⁰·∇η⁰)
  p⁰ = (ξ/2)|∇η⁰|² − (1/4ξ)((η⁰)²−1)²
预计算: 波数 k, 谱符号 L̂(k), M̂(k)  （时间系数 3/2τ）

# ── Startup: BDF2 需两层历史，第一步无 φⁿ⁻¹ ──
用一阶格式（向后 Euler IEQ）走 1 步: (φ⁰,η⁰) → (φ¹,η¹)
  同时得 q¹,r¹,s¹,p¹
（或用更小子步的一阶格式走若干步以控 startup 误差，
  使其不污染整体 O(τ²)）

while 约束残差 > tol and M < M_max:        # continuation
  for n = 1,2,3,... :                            # 内层至稳态，BDF2
    ── φ-子步 (固定 η=η★) ──
      外推 φ★,η★,q★,r★,s★
      组装 §4.1 算子（3/2τ）与 R_φ（含 1/2τ·δⁿ⁻¹φ）
      §5.4 求解 ⇒ δ ⁿφ,  φⁿ⁺¹ = φⁿ + δ ⁿφ
      §4.3 BDF2 推进 qⁿ⁺¹,rⁿ⁺¹,sⁿ⁺¹   # sⁿ⁺¹ 此步仅 φ 贡献(中间值)
    ── η-子步 (固定 φ=φⁿ⁺¹) ──
      外推 η★,p★
      组装 §4.2 算子（3/2τ）与 R_η（含 1/2τ·δⁿ⁻¹η）
      §5.4 求解 ⇒ δ ⁿη,  ηⁿ⁺¹ = ηⁿ + δ ⁿη
      §4.3 BDF2 推进 pⁿ⁺¹,sⁿ⁺¹   # sⁿ⁺¹ 此步补齐 η 贡献, 至此完整
    if ‖δ ⁿφ‖ + ‖δ ⁿη‖ < tol : break             # 内层收敛
  M ← min(ρ·M, M_max)
  若 M 改变: 重做 startup（暖启动 + 一阶走 1 步重建历史）

输出: 稳态 (φ*, η*)
  Γ={φ*=0}, Γ⊥={η*=0}, γ₀={φ*=η*=0}
```

---

## §8 超参数

下表列出需事先选定、整个运行不变的超参数。代表常数 $\bar a^2,\bar g',b_L,\bar W,\bar\zeta$（§5.2）由当前场每步重算，不属超参数，不列入。

| 量 | 出处 | 取法 |
|---|---|---|
| $C_0>0$ | §3 $r$ 正则常数 | $1/r$ 下界 $\sqrt{C_0}$ |
| $\sigma>0$ | Eyre 稳定化（§4.1）| 起点 $\sigma\gtrsim\delta\bar W|\bar\zeta|/\xi$（正性阈含 $3/2\tau$）|
| $\tau$ | 时间步 | 实验定；BDF2 下精度 $O(\tau^2)$ |
| $M_i$ 初值, $\rho$, 容差 | §6 continuation | 实验定 |
| startup 格式 | §7 | 一阶格式走 1 步；或更小子步控 startup 误差 |

---
