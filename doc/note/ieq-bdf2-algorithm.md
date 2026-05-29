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

> 这里我们引入了四个辅助变量，可能在数值上增加误差，但是从能量证明上依然满足耗散。

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

### §3.1 辅助变量演化恒等式

由辅助变量定义对时间求导（链式法则），即得其演化恒等式。每式右端对场增量线性，系数因子取冻结值（在 BDF2 格式中取二阶外推 $\cdot^\star$，见 §4）。演化恒等式的结构如下：

$$
q^{n+1}=q^n+\mathcal C^n_q[\delta^n\phi],\qquad
\mathcal C^n_q[\psi]:=a(\eta^n)\big(\epsilon\Delta\psi+g'(\phi^n)\psi\big)
$$

$$
r^{n+1}=r^n+\mathcal C^n_r[\delta^n\phi],\qquad
\mathcal C^n_r[\psi]:=\tfrac1{r^n}\big(\epsilon\nabla\phi^n\cdot\nabla\psi+\tfrac1\epsilon\phi^n((\phi^n)^2-1)\psi\big)
$$

$$
s^{n+1}=s^n+\mathcal C^n_s[\delta^n\phi],\qquad
\mathcal C^n_s[\psi]:=\sqrt\epsilon\,\nabla\psi\cdot\nabla\eta^n
$$

$$
p^{n+1}=p^n+\mathcal D^n_p[\delta^n\eta],\qquad
\mathcal D^n_p[\psi]:=\xi\nabla\eta^n\cdot\nabla\psi-\tfrac1\xi\eta^n\zeta^n\psi,\quad\zeta^n:=(\eta^n)^2-1
$$

$$
\mathcal D^n_s[\psi]:=\sqrt\epsilon\,\nabla\phi^{n+1}\cdot\nabla\psi
$$

说明：$\mathcal C^n_s$ 用于 $\phi$-子步的 $s$ 演化（$\eta$ 冻结），$\mathcal D^n_s$ 用于 $\eta$-子步的 $s$ 演化（$\phi$ 冻结）；二者结构同为 $\sqrt\epsilon\,\nabla(\cdot)\cdot\nabla(\text{冻结场})$，仅冻结的场不同。所有算子 $\mathcal C,\mathcal D$ 对 $\psi$ 线性 —— 这是 IEQ 能将时间格式线性化的根本。

---

## §4 时间格式：BDF2 一步 $(\phi^{n-1},\phi^n)\to\phi^{n+1}$

### §4.0 BDF2 离散

梯度流 $\phi_t=-\delta\mathcal E_M/\delta\phi$，时间导数用三层 BDF2 公式：

$$
\phi_t\Big|^{n+1}\approx\frac{3\phi^{n+1}-4\phi^n+\phi^{n-1}}{2\tau}
$$

代入梯度流并增量化（$\phi^{n+1}=\phi^n+\delta^n\phi$，$3\phi^{n+1}-4\phi^n+\phi^{n-1}=3\delta^n\phi-\delta^{n-1}\phi$）：

$$
\frac{3}{2\tau}\,\delta^n\phi=-\frac{\delta\mathcal E_M}{\delta\phi}\Big|^{n+1}+\frac{1}{2\tau}\,\delta^{n-1}\phi
$$

右端变分导数取在 $n+1$ 层。下面 §4.1 将其逐项回代 §3.1 的辅助变量演化恒等式：含未知 $\delta^n\phi$ 的部分归并入左端、形成隐式算子，其余已知量留右端。解耦两子步：先解 $\phi$（固定 $\eta=\eta^\star$），后解 $\eta$（固定 $\phi=\phi^{n+1}$），$\eta^\star=\mathrm{Ext}[\eta^n,\eta^{n-1}]$。

### §4.1 $\phi$-子步

固定 $\eta=\eta^\star$。本节先以弯曲项为例完整推出其线性方程，其余项同构。

记号（均为已知量，除 $\phi^{n+1}$、$\delta^n\phi$ 外）：

$$
\eta^\star=2\eta^n-\eta^{n-1},\quad
\phi^\star=2\phi^n-\phi^{n-1},\quad
q^\star=2q^n-q^{n-1}
$$

两个线性微分算子（系数 $a,g'$ 取外推值，故为已知系数算子）：

$$
\mathcal C^\star_q[\psi]:=a(\eta^\star)\big(\epsilon\Delta\psi+g'(\phi^\star)\,\psi\big)
$$

$$
\mathcal G^\star_q[h]:=\epsilon\Delta\big(a(\eta^\star)\,h\big)+g'(\phi^\star)\,a(\eta^\star)\,h
$$

$\mathcal G^\star_q$ 即 §2.1 弯曲项变分导数 $\delta E/\delta\phi$ 中作用在辅助变量 $q$ 上的算子（直接读出，系数取外推值）。

**弯曲项。** 弯曲能 $E=\tfrac12\|q\|^2$，$\delta E/\delta\phi=\mathcal G_q[q]$。BDF2 离散梯度流的弯曲部分（取自 §4.0 末式）：

$$
\frac{3}{2\tau}\,\delta^n\phi=-\mathcal G^\star_q[q^{n+1}]+\frac1{2\tau}\,\delta^{n-1}\phi
\tag{1}
$$

$q^{n+1}$ 未知。$q$ 与 $\phi$ 同样按 BDF2 推进（§4.3 详述），其与场增量的关系为

$$
q^{n+1}=\tfrac13\big(4q^n-q^{n-1}\big)+\mathcal C^\star_q[\delta^n\phi]-\tfrac13\mathcal C^\star_q[\delta^{n-1}\phi]
\tag{2}
$$

右端仅 $\mathcal C^\star_q[\delta^n\phi]$ 含未知 $\delta^n\phi$，其余 $\tfrac13(4q^n-q^{n-1})$ 与 $\tfrac13\mathcal C^\star_q[\delta^{n-1}\phi]$ 全为旧层已知量。

(2) 代入 (1)，用 $\mathcal G^\star_q$ 线性展开：

$$
\frac{3}{2\tau}\,\delta^n\phi
=-\mathcal G^\star_q\mathcal C^\star_q[\delta^n\phi]
-\mathcal G^\star_q\big[\tfrac13(4q^n-q^{n-1})\big]
+\tfrac13\mathcal G^\star_q\mathcal C^\star_q[\delta^{n-1}\phi]
+\frac1{2\tau}\,\delta^{n-1}\phi
$$

含 $\delta^n\phi$ 的项移左、已知项留右：

$$
\Big(\tfrac{3}{2\tau}\mathcal I+\mathcal G^\star_q\mathcal C^\star_q\Big)\,\delta^n\phi
=-\mathcal G^\star_q\big[\tfrac13(4q^n-q^{n-1})\big]
+\tfrac13\mathcal G^\star_q\mathcal C^\star_q[\delta^{n-1}\phi]
+\tfrac1{2\tau}\,\delta^{n-1}\phi
$$

即弯曲项的线性方程，左端隐式算子 $\mathcal A^\star_E:=\mathcal G^\star_q\mathcal C^\star_q$。

**线张力项。** 线张力能 $L$ 经辅助变量 $r$ 表达（$W(\phi)=\tfrac12(r^2-C_0)$），$\delta L/\delta\phi=\mathcal G_r[\Phi_\eta\,r]$，其中 $\Phi^\star_\eta$ 为冻结权重（已知）。算子定义：

$$
\mathcal C^\star_r[\psi]:=\tfrac1{r^\star}\big(\epsilon\nabla\phi^\star\cdot\nabla\psi+\tfrac1\epsilon\phi^\star((\phi^\star)^2-1)\psi\big)
$$

$$
\mathcal G^\star_r[h]:=-\epsilon\,\nabla\!\cdot\!\big(\tfrac h{r^\star}\nabla\phi^\star\big)+\tfrac1{\epsilon r^\star}\phi^\star((\phi^\star)^2-1)\,h
$$

BDF2 离散梯度流的线张力部分：

$$
\frac{3}{2\tau}\,\delta^n\phi=-\mathcal G^\star_r\big[\Phi^\star_\eta\,r^{n+1}\big]+\frac1{2\tau}\,\delta^{n-1}\phi
\tag{1'}
$$

$r^{n+1}$ 未知，与 $\phi$ 同按 BDF2 推进（§4.3）：

$$
r^{n+1}=\tfrac13\big(4r^n-r^{n-1}\big)+\mathcal C^\star_r[\delta^n\phi]-\tfrac13\mathcal C^\star_r[\delta^{n-1}\phi]
\tag{2'}
$$

(2') 代入 (1')，用 $\mathcal G^\star_r$ 线性展开，含 $\delta^n\phi$ 的项移左、已知项留右：

$$
\Big(\tfrac{3}{2\tau}\mathcal I+\mathcal G^\star_r\big(\Phi^\star_\eta\,\mathcal C^\star_r\big)\Big)\,\delta^n\phi
=-\mathcal G^\star_r\big[\Phi^\star_\eta\cdot\tfrac13(4r^n-r^{n-1})\big]
+\tfrac13\mathcal G^\star_r\big(\Phi^\star_\eta\,\mathcal C^\star_r\big)[\delta^{n-1}\phi]
+\tfrac1{2\tau}\,\delta^{n-1}\phi
$$

左端隐式算子 $\mathcal A^\star_L:=\mathcal G^\star_r(\Phi^\star_\eta\,\mathcal C^\star_r)$。

**其余项。** 约束 $A,D$ 经 $r$、正交 $N$ 经 $s$，其辅助变量同按 §4.3 的 BDF2 推进，回代方式与弯曲、线张力相同，各自的隐式算子见下方算子表。Eyre 稳定化：在方程中加入 $\sigma(\delta^n\phi-\delta^{n-1}\phi)$（$\sigma>0$），其中 $\sigma\delta^n\phi$ 含未知量、归入左端，$-\sigma\delta^{n-1}\phi$ 已知、归入右端。稳定项作用在二阶差分 $\delta^n\phi-\delta^{n-1}\phi=\phi^{n+1}-2\phi^n+\phi^{n-1}=O(\tau^2)$ 上，与 BDF2 主格式同阶，不降低时间精度；稳态时该项为零，不改不动点。罚项的隐式离散需专门处理，见下。

**罚项的 Newton 线性化。** 罚项 $\tfrac{M_j}2(C_j-c_j)^2$（$C_j$ 为约束泛函 $V,A,D,N$，$c_j$ 为目标值）。其变分导数（一次求导，链式法则）：

$$
\frac{\delta}{\delta\phi}\Big[\tfrac{M_j}2(C_j-c_j)^2\Big]=M_j(C_j-c_j)\,\frac{\delta C_j}{\delta\phi}
$$

BDF2 离散梯度流为 $\tfrac{3\phi^{n+1}-4\phi^n+\phi^{n-1}}{2\tau}=-\delta\mathcal E_M/\delta\phi|^{n+1}$，右端梯度取在未知层 $n+1$。把上标 $|^{n+1}$ 落到罚项梯度上——式中 $\phi$ 全代 $\phi^{n+1}$：

$$
M_j(C_j-c_j)\,\frac{\delta C_j}{\delta\phi}\bigg|^{n+1}
=M_j\big(C_j^{n+1}-c_j\big)\,\Big(\frac{\delta C_j}{\delta\phi}\Big)^{n+1}
$$

其中 $C_j^{n+1}=C_j(\phi^{n+1})$ 与 $(\delta C_j/\delta\phi)^{n+1}$ 均含未知场 $\phi^{n+1}$，不能直接计算。

**与弯曲项的对照。** 弯曲项也面对同一处境：式 (1) 右端 $\mathcal G^\star_q[q^{n+1}]$ 含未知 $q^{n+1}$。但弯曲能的非线性已被辅助变量 $q$ 二次化吸收，$q$ 有一条**精确且线性**的 BDF2 推进式 (2)，代入即把未知量精确转为对场增量的线性依赖，无需任何近似。罚项的 $C_j^{n+1}$ 没有这样的辅助变量——罚项外层平方 $(C_j-c_j)^2$ 未被 IEQ 二次化，$C_j$ 对 $\phi$ 非线性，不存在现成的线性递推。故只能退而用 **Newton 线性化**：对 $n+1$ 层梯度作泰勒展开取一阶，将其对 $\phi^{n+1}$ 的非线性依赖近似成对增量的线性依赖。

具体地，对梯度 $M_j(C_j-c_j)\,\delta C_j/\delta\phi$ 作 Newton 线性化。它是两个含 $\phi$ 的因子 $M_j(C_j-c_j)$ 与 $\delta C_j/\delta\phi$ 之积，对其求方向导数（按乘积法则）得 Hessian 两项：

$$
\text{Hessian}[\,\cdot\,]
=\underbrace{M_j\big(\tfrac{\delta C_j}{\delta\phi},\,\cdot\,\big)\,\tfrac{\delta C_j}{\delta\phi}}_{\text{主项：秩一,半正定}}
+\underbrace{M_j(C_j-c_j)\,\mathrm d^2C_j[\,\cdot\,]}_{\text{二阶项}}
$$

**主项**为 $(\delta C_j/\delta\phi)\otimes(\delta C_j/\delta\phi)$ 型秩一算子，半正定。与 Eyre 稳定项同理，为保持 BDF2 二阶精度，主项作用在二阶差分 $\delta^n\phi-\delta^{n-1}\phi$ 上：$M_j(\delta C_j/\delta\phi,\delta^n\phi-\delta^{n-1}\phi)(\delta C_j/\delta\phi)$，其中含 $\delta^n\phi$ 的部分（隐式算子 $\mathcal A^\star_{A,D,N,V}$）移入左端，含 $\delta^{n-1}\phi$ 的部分已知、移入右端。**二阶项**含约束残差 $(C_j-c_j)$ 与二阶变分 $\mathrm d^2C_j$：$(C_j-c_j)$ 在收敛前可正可负，$\mathrm d^2C_j$ 有定号，两者相乘符号不定；若入左端会破坏 LHS 正定性。又因该项在稳态 $\delta^n\phi-\delta^{n-1}\phi=0$ 时为零，对不动点无贡献。故**整项丢弃**——不进 LHS（保正定），不进 RHS（不污染不动点）。此即 Gauss–Newton 做法：仅取 Hessian 的半正定主项。右端另放精确负梯度 $-M_j(C_j-c_j)\,\delta C_j/\delta\phi$。

*体积 $V$。* $V(\phi)=\int_\Omega\phi\,dx$ 为线性泛函，$\delta V/\delta\phi=\mathbf 1$（常值场），$\mathrm d^2V=0$。二阶项天然为零，无需丢弃；主项即 $\mathcal A^\star_V=M_1\,\mathbf 1\otimes\mathbf 1$。

*面积 $A$（逐步）。* $A=\int_\Omega W(\phi)\,dx$，$W(\phi)=\tfrac\epsilon2|\nabla\phi|^2+\tfrac1{4\epsilon}(\phi^2-1)^2$，$\phi$ 非线性出现，故 $\mathrm d^2A\ne0$。

一阶变分（$W$ 对 $\phi$ 求变分，梯度项分部积分）：

$$
\frac{\delta A}{\delta\phi}=-\epsilon\Delta\phi+\tfrac1\epsilon\phi(\phi^2-1)
$$

二阶变分（对上式再求方向导数；$-\epsilon\Delta$ 线性、导数为自身，$\tfrac1\epsilon\phi(\phi^2-1)=\tfrac1\epsilon(\phi^3-\phi)$ 求导得 $\tfrac1\epsilon(3\phi^2-1)$）：

$$
\mathrm d^2A[\psi]=-\epsilon\Delta\psi+\tfrac1\epsilon(3\phi^2-1)\,\psi
$$

代入 Hessian 通式（$j=2$，系数取外推层 $\phi^\star$），主项作用在二阶差分上：

$$
\text{主项}_A=M_2\big(\tfrac{\delta A}{\delta\phi}\big)^\star\!\!\otimes\!\big(\tfrac{\delta A}{\delta\phi}\big)^\star\,[\delta^n\phi-\delta^{n-1}\phi],
\qquad
\text{二阶项}_A=M_2(A^\star-a_0)\,\mathrm d^2A^\star[\,\cdot\,]
$$

主项中含 $\delta^n\phi$ 的部分给出 $\mathcal A^\star_A=M_2\,e^\star_A\otimes e^\star_A$（$e^\star_A=(\delta A/\delta\phi)^\star$，经 $r$ 即 $\mathcal G^\star_r[r^\star]$），含 $\delta^{n-1}\phi$ 的部分入右端；二阶项 $M_2(A^\star-a_0)\big(-\epsilon\Delta+\tfrac1\epsilon(3(\phi^\star)^2-1)\big)$ 含变号标量 $(A^\star-a_0)$、整项丢弃。$D,N$ 同理。

**回代结果。** 各项相加，$\delta^n\phi$ 满足线性方程

$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+\sigma\mathcal I
+\mathcal A^\star_E+\mathcal A^\star_L+\mathcal A^\star_A+\mathcal A^\star_D+\mathcal A^\star_N+\mathcal A^\star_V\Big)\,\delta^n\phi=\mathcal R^n_\phi
\;}
$$

各隐式算子：

$$
\begin{aligned}
\mathcal A^\star_E &= \mathcal G^\star_q\mathcal C^\star_q && \text{弯曲}\\
\mathcal A^\star_L &= \mathcal G^\star_r\big(\Phi_\eta^\star\,\mathcal C^\star_r\big) && \text{线张力}\\
\mathcal A^\star_A &= M_2\,e^\star_A\!\otimes\!e^\star_A, & e^\star_A&=\mathcal G^\star_r[r^\star] && \text{面积罚，秩一}\\
\mathcal A^\star_D &= M_3\,e^\star_D\!\otimes\!e^\star_D, & e^\star_D&=\mathcal G^\star_r[\tanh(\eta^\star/\xi)\,r^\star] && \text{面积差罚，秩一}\\
\mathcal A^\star_N &= M_4\,e^\star_N\!\otimes\!e^\star_N, & e^\star_N&=\mathcal G^\star_s[s^\star] && \text{正交罚，秩一}\\
\mathcal A^\star_V &= M_1\,\mathbf 1\!\otimes\!\mathbf 1 && \text{体积罚，秩一}
\end{aligned}
$$

其中正交项的外推算子 $\mathcal C^\star_s[\psi]=\sqrt\epsilon\,\nabla\psi\cdot\nabla\eta^\star$，$\mathcal G^\star_s$ 为 §2.1 正交项变分导数中作用在辅助变量 $s$ 上的算子：

$$
\mathcal G^\star_s[h]=-\sqrt\epsilon\,\nabla\!\cdot\!(h\,\nabla\eta^\star)
$$

$\mathcal A^\star_{A,D,N,V}$ 为秩一算子（$u\otimes u$ 型），对称半正定。$\mathcal A^\star_E=\mathcal G^\star_q\mathcal C^\star_q$、$\mathcal A^\star_L=\mathcal G^\star_r(\Phi^\star_\eta\,\mathcal C^\star_r)$ 为 $\mathcal G\mathcal C$ 型复合：$\mathcal G^\star_q$ 恰为 $\mathcal C^\star_q$ 的伴随（$\mathcal C^\star_q=M_a\circ(\epsilon\Delta+M_{g'})$ 中乘子 $M_a,M_{g'}$ 与 $\epsilon\Delta$ 在周期边界下均自伴，故 $(\mathcal C^\star_q)^*=(\epsilon\Delta+M_{g'})\circ M_a=\mathcal G^\star_q$，与 $a$ 是否变系数无关），故 $\mathcal A^\star_E=(\mathcal C^\star_q)^*\mathcal C^\star_q$ 对称半正定；$\mathcal A^\star_L$ 同理，且因 $\Phi^\star_\eta\ge0$ 亦对称半正定。左端整体的可逆性由时间项 $\tfrac3{2\tau}\mathcal I$ 与 $\sigma$ 保证，谱求解见 §5。

**右端 $\mathcal R^n_\phi$。** 由各项回代后留在右端的已知量收集而成。逐项列出：

弯曲、线张力（来自式 (2)、(2′) 中的已知部分）：

$$
\mathcal R^n_{\phi,E}=-\mathcal G^\star_q\big[\tfrac13(4q^n-q^{n-1})\big]+\tfrac13\mathcal G^\star_q\mathcal C^\star_q[\delta^{n-1}\phi]
$$

$$
\mathcal R^n_{\phi,L}=-\mathcal G^\star_r\big[\Phi^\star_\eta\cdot\tfrac13(4r^n-r^{n-1})\big]+\tfrac13\mathcal G^\star_r(\Phi^\star_\eta\mathcal C^\star_r)[\delta^{n-1}\phi]
$$

罚项（精确负梯度 + 主项二阶差分的已知半 $\delta^{n-1}\phi$ 部分；$j=1,\dots,4$ 对应 $V,A,D,N$）：

$$
\mathcal R^n_{\phi,P}=\sum_j\Big(-M_j(C_j^\star-c_j)\big(\tfrac{\delta C_j}{\delta\phi}\big)^\star
+M_j\big(\tfrac{\delta C_j}{\delta\phi}\big)^\star\!\otimes\!\big(\tfrac{\delta C_j}{\delta\phi}\big)^\star[\delta^{n-1}\phi]\Big)
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

解出 $\delta^n\phi$ 后：$\phi^{n+1}=\phi^n+\delta^n\phi$，并按 §4.3 更新辅助变量。

### §4.2 $\eta$-子步

固定 $\phi=\phi^{n+1}$。$\eta^{n+1}$（即 $\delta^n\eta$）满足

$$
\boxed{\;
\Big(\tfrac{3}{2\tau}\mathcal I+\sigma\mathcal I
+\mathcal B^\star_L+\mathcal B^\star_D+\mathcal B^\star_N+\mathcal B^\star_P\Big)\,\delta^n\eta=\mathcal R^n_\eta
\;}
$$

各隐式算子如下，冻结系数取二阶外推（$W(\phi^{n+1})$ 已是新层精确值，无需外推；$\eta$ 相关系数取 $\eta^\star$）：

$$
\begin{aligned}
\mathcal B^\star_L[\psi] &= -\delta\xi\,\nabla\!\cdot\!\big(W(\phi^{n+1})\nabla\psi\big)
+\delta\tfrac1\xi W(\phi^{n+1})\,\zeta^\star\,\psi && \zeta^\star=(\eta^\star)^2-1\\
\mathcal B^\star_D &= M_3\,f^\star_D\!\otimes\!f^\star_D, & f^\star_D&=\tfrac1\xi W(\phi^{n+1})\,\mathrm{sech}^2(\eta^\star/\xi)\\
\mathcal B^\star_N &= M_4\,e^\star_{N\eta}\!\otimes\!e^\star_{N\eta}, & e^\star_{N\eta}&=\mathcal G^\star_{s\eta}[s^{n+1}_{(\phi)}]\\
\mathcal B^\star_P &= 4M_5\,e^\star_P\!\otimes\!e^\star_P, & e^\star_P&=\mathcal G^\star_p[p^\star]
\end{aligned}
$$

$$
\mathcal G^\star_{s\eta}[h]=-\sqrt\epsilon\,\nabla\!\cdot\!(h\,\nabla\phi^{n+1}),\qquad
\mathcal G^\star_p[h]=-\xi\,\nabla\!\cdot\!(h\,\nabla\eta^\star)-\tfrac1\xi\eta^\star\zeta^\star\,h
$$

$\mathcal B^\star_{D},\mathcal B^\star_N,\mathcal B^\star_P$ 为秩一算子（$u\otimes u$ 型），对称半正定。$\mathcal B^\star_L$ 对称，但其零阶系数 $\delta\tfrac1\xi W(\phi^{n+1})\,\zeta^\star$ 在 $|\eta^\star|<1$（区域大部分）处为负，故 $\mathcal B^\star_L$ **非半正定**；左端整体 $\tfrac3{2\tau}\mathcal I+\sigma\mathcal I+\mathcal B^\star_L$ 的正定性需 $\sigma$ 足够大保证（阈值见 §5.2）。$\mathcal B^\star_L$ 的零阶项取 $\zeta^\star=(\eta^\star)^2-1$ 是 IEQ 式半隐线性化（系数冻结于外推层），非 Newton 雅可比 $3(\eta^\star)^2-1$；二者收敛时一致，此处为格式选择。

**右端 $\mathcal R^n_\eta$。** 各项分两类处理。

*弯曲项* —— $q^{n+1}$ 在 $\phi$-子步已解出，$-\tfrac{k'(\eta^\star)}{2k(\eta^\star)}(q^{n+1})^2$ 整项已知，直接入右端。

*线张力项* —— $L$ 的 $\eta$-依赖全在 $\Phi_\eta(\eta)$（含 $|\nabla\eta|^2$ 与 $(\eta^2-1)^2$），无对应辅助变量（$r$ 只二次化 $W(\phi)$，在 $\eta$-子步不推进）。故 $\mathcal B^\star_L$ 是直接微分算子，按 $\eta^{n+1}=\eta^\star+(\delta^n\eta-\delta^{n-1}\eta)$ 作二阶差分分解：$\mathcal B^\star_L[\eta^{n+1}]=\mathcal B^\star_L[\eta^\star]+\mathcal B^\star_L[\delta^n\eta]-\mathcal B^\star_L[\delta^{n-1}\eta]$。含 $\delta^n\eta$ 的 $\mathcal B^\star_L[\delta^n\eta]$ 入左端，其余入右端：

$$
\mathcal R^n_{\eta,L}=-\mathcal B^\star_L[\eta^\star]+\mathcal B^\star_L[\delta^{n-1}\eta]
$$

*正交、剖面项* —— 罚项 $\tfrac{M_4}2N^2,\tfrac{M_5}2P^2$ 对 $s,p$ 为四次（$N=\tfrac12\|s\|^2$），故按罚项 Newton 线性化处理，与 §4.1 的 $\mathcal A^\star_N$ 同类、非 $r$ 代入：$\mathcal B^\star_N,\mathcal B^\star_P$ 为秩一主项（其秩一向量 $e^\star_{N\eta}=\mathcal G^\star_{s\eta}[s^{n+1}]$、$e^\star_P=\mathcal G^\star_p[p^\star]$ 取已推进的辅助变量值，$s,p$ 推进见 §4.3），作用在二阶差分 $\delta^n\eta-\delta^{n-1}\eta$ 上，$\delta^{n-1}\eta$ 半与精确负梯度入右端，记 $\mathcal R^n_{\eta,N},\mathcal R^n_{\eta,P}$。

*面积差项* —— 经 $r$ 但 $r$ 在 $\eta$-子步不推进；其 $\eta$-依赖在 $\tanh(\eta/\xi)$ 中，$\mathcal B^\star_D$ 为罚项 Newton 线性化的秩一主项，作用在二阶差分 $\delta^n\eta-\delta^{n-1}\eta$ 上，$\delta^{n-1}\eta$ 半入右端，记 $\mathcal R^n_{\eta,D}$。

*Eyre 与 BDF2 历史* —— $\sigma(\delta^n\eta-\delta^{n-1}\eta)$ 的 $-\sigma\delta^{n-1}\eta$、BDF2 历史 $\tfrac1{2\tau}\delta^{n-1}\eta$ 入右端。

总右端：

$$
\mathcal R^n_\eta=-\tfrac{k'(\eta^\star)}{2k(\eta^\star)}(q^{n+1})^2
+\mathcal R^n_{\eta,L}+\mathcal R^n_{\eta,D}+\mathcal R^n_{\eta,N}+\mathcal R^n_{\eta,P}
+\big(\sigma+\tfrac1{2\tau}\big)\delta^{n-1}\eta
$$

其中各项的旧层精确梯度密度（外推层）记 $\mathcal F^\star_X$（$X\in\{L,D,N,P\}$，即 §2.2 变分导数对应项在外推层取值）：

$$
\begin{aligned}
\mathcal F^\star_L&=-\delta\xi\,\nabla\!\cdot\!\big(W(\phi^{n+1})\nabla\eta^\star\big)+\delta\tfrac1\xi W(\phi^{n+1})\,\eta^\star((\eta^\star)^2-1)\quad\big(=\mathcal B^\star_L[\eta^\star]\big)\\
\mathcal F^\star_D&=M_3(D^\star-a_d)\,\tfrac1\xi W(\phi^{n+1})\,\mathrm{sech}^2(\eta^\star/\xi)\\
\mathcal F^\star_N&=-M_4 N^\star\,\epsilon\,\nabla\!\cdot\!\big((\nabla\phi^{n+1}\cdot\nabla\eta^\star)\nabla\phi^{n+1}\big)\\
\mathcal F^\star_P&=M_5 P^\star\big(-2\xi\nabla\!\cdot\!(\Pi(\eta^\star)\nabla\eta^\star)-\tfrac2\xi\Pi(\eta^\star)\,\eta^\star((\eta^\star)^2-1)\big)
\end{aligned}
$$

各 $\mathcal R^n_{\eta,X}$ 由"精确负梯度密度 $-\mathcal F^\star_X$"与"该项隐式算子作用在已知增量 $\delta^{n-1}\eta$ 上"组成，结构统一：

$$
\mathcal R^n_{\eta,X}=-\mathcal F^\star_X+\mathcal B^\star_X[\delta^{n-1}\eta],\qquad X\in\{L,D,N,P\}
$$

其中 $\mathcal B^\star_L$ 为线张力微分算子（此式即上方 $\mathcal R^n_{\eta,L}$，因 $\mathcal F^\star_L=\mathcal B^\star_L[\eta^\star]$），$\mathcal B^\star_{D,N,P}$ 为 §4.2 算子表中的秩一算子。弯曲项 $-\tfrac{k'(\eta^\star)}{2k(\eta^\star)}(q^{n+1})^2$ 用新层 $q^{n+1}$、不属此外推密度，单列于总右端。

解出 $\delta^n\eta$ 后：$\eta^{n+1}=\eta^n+\delta^n\eta$，按 §4.3 更新辅助变量。

### §4.3 辅助变量的 BDF2 推进

辅助变量按 BDF2 三层公式推进，与场保持同阶：

$$
q^{n+1}=\tfrac13\big(4q^n-q^{n-1}\big)+\tfrac{2\tau}3\,\mathcal C^\star_q\Big[\tfrac{\delta^n\phi}{\tau}\Big]_{\text{BDF2}}
$$

实用等价形式（由 §3.1 演化恒等式 $q_t=\mathcal C_q[\phi_t]$ 配 BDF2 离散得）：

$$
3q^{n+1}-4q^n+q^{n-1}=\mathcal C^\star_q\big[\,3\delta^n\phi-\delta^{n-1}\phi\,\big]
$$

即

$$
\boxed{\;
q^{n+1}=\tfrac13\Big(4q^n-q^{n-1}+\mathcal C^\star_q\big[3\delta^n\phi-\delta^{n-1}\phi\big]\Big)
\;}
$$

$r,s,p$ 同构（把 $\mathcal C^\star_q$ 换成 $\mathcal C^\star_r,\mathcal C^\star_s,\mathcal D^\star_p$，$\delta^n\phi$ 换成对应场增量）。这样辅助变量推进与场推进同为 BDF2 二阶。

> BDF2 推进需保存两层辅助变量旧值 $q^n,q^{n-1}$（$r,s,p$ 同）。

---

## §5 谱空间求解

§4 的隐式算子含变系数：$\phi$-子步中 $\mathcal A^\star_E$ 含 $a(\eta^\star)^2$，$\mathcal A^\star_L$ 含 $\Phi^\star_\eta$、$1/r^\star$、$\nabla\phi^\star$；$\eta$-子步中 $\mathcal B^\star_L$ 含 $W(\phi^{n+1})$。这些系数随空间位置变化。Fourier 谱方法的高效性在于：常系数微分算子（如 $\Delta$）经 FFT 变为各波数 $\mathbf k$ 上的乘子（$\Delta\to-|\mathbf k|^2$），不同波数互不耦合，故可逐波数求逆。但变系数与场逐点相乘，在 Fourier 空间对应不同波数间的卷积——波数被耦合在一起，不能再逐波数独立求逆。下面的劈分策略即把变系数算子拆成"波数可分离的常系数部分（隐式求解）+ 变系数偏差（显式滞后）"。

### §5.2 劈分：常系数骨架（隐式）+ 余项（显式）

采用 $(\mathcal O-\mathcal S)$ 残差策略（stabilized semi-implicit / IMEX splitting，标准方法，见 Chen & Shen, *Comput. Phys. Commun.* 108 (1998) 147；Eyre, MRS Proc. 529 (1998)）。对每个含变系数的微分算子 $\mathcal O$，将其劈成常系数骨架 $\mathcal S$（Fourier 谱对角，隐式）与余项 $\mathcal O-\mathcal S$（含全部变系数偏差，显式）。

**余项取二阶外推** $\mathrm{Ext}[\phi^n,\phi^{n-1}]$：

$$
\mathcal O[\phi^{n+1}]=\mathcal S[\phi^{n+1}]+(\mathcal O-\mathcal S)[\phi^{n+1}]
\approx\mathcal S[\phi^{n+1}]+(\mathcal O-\mathcal S)\big[\mathrm{Ext}[\phi^n,\phi^{n-1}]\big]
$$

骨架 $\mathcal S[\phi^{n+1}]$ 进 LHS（对角可逆），余项 $-(\mathcal O-\mathcal S)[\mathrm{Ext}[\phi^n,\phi^{n-1}]]$ 滞后入右端。余项以二阶外推估 $n+1$ 层值，使滞后误差为 $O(\tau^2)$，与 BDF2 主格式同阶。

只有微分型算子进骨架（$\phi$-子步的 $\mathcal A^\star_E,\mathcal A^\star_L$，$\eta$-子步的 $\mathcal B^\star_L$）；秩一算子（$\mathcal A^\star_{A,D,N,V}$、$\mathcal B^\star_{D,N,P}$）不进骨架，全部走 Woodbury（§5.3）。

骨架谱符号如下，LHS 时间系数为 $\tfrac3{2\tau}$：

$$
\widehat{\mathcal L}(\mathbf k)=\tfrac{3}{2\tau}+\sigma
+\bar a^2\big(\epsilon|\mathbf k|^2-\bar g'\big)^2+b_L|\mathbf k|^2
$$

$$
\widehat{\mathcal M}(\mathbf k)=\tfrac{3}{2\tau}+\sigma
+\delta\xi\bar W|\mathbf k|^2+\delta\tfrac1\xi\bar W\bar\zeta
$$

其中 $\widehat{\mathcal L}$ 为 $\phi$-子步骨架符号、$\widehat{\mathcal M}$ 为 $\eta$-子步骨架符号。代表常数由变系数取**空间平均**得到：

$$
\bar a^2=\overline{k(\eta^\star)}/\epsilon,\quad
b_L=\epsilon^2\,\mathrm{rep}\big(\Phi^\star_\eta|\nabla\phi^\star|^2/(r^\star)^2\big),\quad
\bar W=\overline{W(\phi^{n+1})},\quad\bar\zeta=\overline{(\eta^\star)^2-1}
$$

$\bar g'$ 为 $g'(\phi^\star)$ 的空间平均。代表常数的底层场量在 BDF2 版取二阶外推值 $\phi^\star,\eta^\star$。"取空间平均"这一规则固定，规则算出的数值随当前场每步变化，每步重算 —— 它不是超参数。

正性：$\widehat{\mathcal L}>0$ 对所有 $\mathbf k$ 无条件成立（三项均非负，$\tfrac3{2\tau}+\sigma>0$）；$\widehat{\mathcal M}>0$ 要求 $\sigma>\delta\bar W|\bar\zeta|/\xi-\tfrac3{2\tau}$（零阶项 $\delta\tfrac1\xi\bar W\bar\zeta$ 在 $\bar\zeta<0$ 时为负，由 $\sigma$ 与时间项压住）。

### §5.3 秩一项：Woodbury

体积、面积、面积差、正交罚项的隐式算子是**秩一**的（形如 $u\otimes u$）。§4 的左端整体形如"对角骨架 $\mathcal D$ + 若干秩一修正"，其中 $\mathcal D$ 的符号为 $\widehat{\mathcal L}$（$\phi$-子步）或 $\widehat{\mathcal M}$（$\eta$-子步）。该结构可用 Woodbury 公式**精确**求逆（直接法，非迭代、无近似误差）。

LHS 写成 $\mathcal D+\sum_j\alpha_j u_j u_j^\top$，$\mathcal D$ 对角，$u_j$ 为秩一向量，$\alpha_j\ge0$；记 $U=[u_1,\dots,u_m]$、$\alpha=\mathrm{diag}(\alpha_j)$：

$$
(\mathcal D+U\alpha U^\top)^{-1}\mathcal R
=\mathcal D^{-1}\mathcal R-\mathcal D^{-1}U\big(\alpha^{-1}+U^\top\mathcal D^{-1}U\big)^{-1}U^\top\mathcal D^{-1}\mathcal R
$$

$\mathcal D^{-1}$ 即逐波数除（FFT 域对角），$(\alpha^{-1}+U^\top\mathcal D^{-1}U)$ 为 $m\times m$ 小稠密矩阵（$m\le4$，约束个数）。每子步代价：$m+1$ 次 FFT-除 + 一个 $m\times m$ 稠密小解。BDF2 不改变此结构，亦不增加 Woodbury 代价。

### §5.4 求解流程（$\phi$-子步，$\eta$-子步同构）

```
1. 二阶外推: φ★=2φⁿ−φⁿ⁻¹, η★=2ηⁿ−ηⁿ⁻¹, q★,r★,s★,p★ 同
2. 实空间组装右端 R = −(δE_M/δfield)★ + (1/2τ)δⁿ⁻¹field
                       （含外推变系数余项 −(O−S)[Ext]）
3. FFT(R)
4. 逐波数除 L̂(k)   ⇒ D⁻¹R     （η-子步除 M̂(k)，时间系数 3/2τ）
5. 对全部秩一项作 Woodbury 修正
6. IFFT  ⇒ δ ⁿφ
7. φⁿ⁺¹ = φⁿ + δ ⁿφ
8. 按 §4.3 BDF2 推进 qⁿ⁺¹,rⁿ⁺¹,sⁿ⁺¹（需 qⁿ,qⁿ⁻¹ 两层）
```

变系数余项以二阶外推估值，滞后误差为 $O(\tau^2)$；若 $\tau$ 较大使外推不足，可改以 $\widehat{\mathcal L}^{-1}$（或 $\widehat{\mathcal M}^{-1}$）为预条件子，对含余项的完整方程作不动点迭代或 CG 迭代细化，将余项收敛至精确。

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

while 约束残差 > outer_tol and M < M_max:        # continuation
  for n = 1,2,3,... :                            # 内层至稳态，BDF2
    ── φ-子步 (固定 η=η★) ──
      外推 φ★,η★,q★,r★,s★
      组装 §4.1 算子（3/2τ）与 R_φ（含 1/2τ·δⁿ⁻¹φ）
      §5.4 求解 ⇒ δ ⁿφ,  φⁿ⁺¹ = φⁿ + δ ⁿφ
      §4.3 BDF2 推进 qⁿ⁺¹,rⁿ⁺¹,sⁿ⁺¹
    ── η-子步 (固定 φ=φⁿ⁺¹) ──
      外推 η★,p★
      组装 §4.2 算子（3/2τ）与 R_η（含 1/2τ·δⁿ⁻¹η）
      §5.4 求解 ⇒ δ ⁿη,  ηⁿ⁺¹ = ηⁿ + δ ⁿη
      §4.3 BDF2 推进 pⁿ⁺¹,sⁿ⁺¹
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
