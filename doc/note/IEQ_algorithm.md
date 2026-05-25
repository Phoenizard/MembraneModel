# 经典 IEQ 算法 —— 两组分囊泡相场模型

> 目标：求梯度流稳态 $(\phi^*,\eta^*)=\arg\min\mathcal E_M$。
> 时间离散：一阶解耦 IEQ。空间离散：Fourier 谱（FFT），周期边界，$64^3$。
> 简化：$c_1=c_2=0\Rightarrow c_0(\eta)\equiv0$，$\epsilon=\xi$。

---

## §0 记号

$$
g(\phi)=\tfrac1\epsilon\phi(1-\phi^2),\qquad
g'(\phi)=\tfrac1\epsilon(1-3\phi^2),\qquad
W(\phi)=\tfrac\epsilon2|\nabla\phi|^2+\tfrac1{4\epsilon}(\phi^2-1)^2
$$

$$
k(\eta)=k+c\,\tanh(\eta/\xi),\qquad
a(\eta)=\sqrt{k(\eta)/\epsilon},\qquad
k'(\eta)=\tfrac c\xi\,\mathrm{sech}^2(\eta/\xi)
$$

$L^2$ 内积 $(f,g)=\int_\Omega fg\,dx$，$\|f\|^2=(f,f)$。场增量 $\delta^n\phi:=\phi^{n+1}-\phi^n$，$\delta^n\eta:=\eta^{n+1}-\eta^n$。

---

## §1 梯度流与能量

$$
\phi_t=-\frac{\delta\mathcal E_M}{\delta\phi},\qquad
\eta_t=-\frac{\delta\mathcal E_M}{\delta\eta}
$$

$$
\mathcal E_M=\underbrace{E}_{\text{弯曲}}+\underbrace{L}_{\text{线张力}}
+\tfrac{M_1}2(V-v_d)^2+\tfrac{M_2}2(A-a_0)^2+\tfrac{M_3}2(D-a_d)^2
+\tfrac{M_4}2N^2+\tfrac{M_5}2P^2
$$

各泛函见 `IEQ_scheme_vesicle.md` §1。算法只依赖下面两个变分导数（§2）与 IEQ 二次化结构（§3）。

---

## §2 变分导数（算法入口）

经三步法（方向导数 → 分部积分 → 读系数）对泛函直接求得。完整推导见 `IEQ_scheme_vesicle.md` §4，此处只列最终结果。

### §2.1 对 $\phi$

冻结系数 $\Phi_\eta:=\delta\big(\tfrac\xi2|\nabla\eta|^2+\tfrac1{4\xi}(\eta^2-1)^2\big)$。

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

### §2.2 对 $\eta$

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

其中 $\Pi(\eta)=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2$。

### §2.3 经辅助变量改写

§2.1、§2.2 为变分导数的**原始形式**（仅依赖场，不含辅助变量）—— 部署版直接用这一形式算精确梯度 $G$。经典 IEQ 时间格式（§4）则用 §3 的辅助变量改写：弯曲项 $\delta E/\delta\phi$ 改写为 $\mathcal C^{n*}_q[q]$ 型，$\delta E/\delta\eta$ 改写为 $\tfrac{k'(\eta)}{2k(\eta)}q^2$ 型，线张力/约束项经 $r$、正交项经 $s$、剖面项经 $p$ 改写。改写后非线性被吸收进辅助变量，格式对未知场线性。

---

## §3 IEQ 二次化：辅助变量

对每个非线性势引辅助变量，使能量成为其二次型。

$$
\begin{array}{lll}
\text{弯曲} & q:=a(\eta)\big(\epsilon\Delta\phi+g(\phi)\big) & E=\tfrac12\|q\|^2\\[4pt]
\text{线张力/面积} & r:=\sqrt{2W(\phi)+C_0} & W(\phi)=\tfrac12(r^2-C_0)\\[4pt]
\text{正交项} & s:=\sqrt\epsilon\,(\nabla\phi\cdot\nabla\eta) & N=\tfrac12\|s\|^2\\[4pt]
\eta\ \text{剖面} & p:=\tfrac\xi2|\nabla\eta|^2-\tfrac1{4\xi}(\eta^2-1)^2 & P=\|p\|^2
\end{array}
$$

要求 $k(\eta)>0$（取 $k>|c|$），$C_0>0$（避免 $1/r$ 奇异）。

$A,D$ 经 $r$ 共用：$A=\tfrac12\int(r^2-C_0)dx$，$D=\tfrac12\int\tanh(\eta/\xi)(r^2-C_0)dx$。

### §3.1 辅助变量演化恒等式

由定义对时间求导（链式法则），系数因子取旧层（D2）：

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

注：$\mathcal C^n_s$ 用于 $\phi$-子步的 $s$ 演化（$\eta=\eta^n$ 冻结）；$\mathcal D^n_s$ 用于 $\eta$-子步的 $s$ 演化（$\phi=\phi^{n+1}$ 冻结）。二者结构同为 $\sqrt\epsilon\,\nabla(\cdot)\cdot\nabla(\text{冻结场})$，仅冻结的场不同。

所有 $\mathcal C,\mathcal D$ 对 $\psi$ 线性 —— 这是 IEQ 能线性化的根。

---

## §4 时间格式：一步 $(\phi^n,\eta^n)\to(\phi^{n+1},\eta^{n+1})$

解耦两子步（D3）：先 $\phi$（固定 $\eta=\eta^n$），后 $\eta$（固定 $\phi=\phi^{n+1}$）。

格式约定：
- **D1** 辅助变量 $n+1$ 层隐式，由 §3.1 演化式回代消元。
- **D4** 罚项 $\tfrac{M_j}2(C_j-c_j)^2$ 外层标量作 Newton 线性化，仅半正定 Hessian 主项进 LHS，可变号二阶项显式入右端。
- **D5** 双井项 Eyre 凸劈分 + 稳定化常数 $\sigma>0$。

### §4.1 $\phi$-子步

$L^2$ 梯度流一阶离散 + Eyre 稳定化，回代 §3.1 辅助变量后，$\phi^{n+1}$ 满足线性方程

$$
\boxed{\;
\Big(\tfrac1\tau\mathcal I+\sigma\mathcal I
+\mathcal A^n_E+\mathcal A^n_L+\mathcal A^n_A+\mathcal A^n_D+\mathcal A^n_N+\mathcal A^n_V\Big)\,\delta^n\phi=\mathcal R^n_\phi
\;}
$$

各隐式算子（均对称半正定）：

$$
\begin{aligned}
\mathcal A^n_E &= (\mathcal C^n_q)^*\mathcal C^n_q && \text{弯曲，}\mathcal C^*\mathcal C\text{ 型}\\
\mathcal A^n_L &= (\mathcal C^n_r)^*\big(\Phi_\eta\,\mathcal C^n_r\big) && \text{线张力，}\Phi_\eta>0\\
\mathcal A^n_A &= M_2\,e^n_A\!\otimes\!e^n_A, & e^n_A&=(\mathcal C^n_r)^*[r^n] && \text{面积罚，秩一}\\
\mathcal A^n_D &= M_3\,e^n_D\!\otimes\!e^n_D, & e^n_D&=(\mathcal C^n_r)^*[\tanh(\eta^n/\xi)\,r^n] && \text{面积差罚，秩一}\\
\mathcal A^n_N &= M_4\,e^n_N\!\otimes\!e^n_N, & e^n_N&=(\mathcal C^n_s)^*[s^n] && \text{正交罚，秩一}\\
\mathcal A^n_V &= M_1\,\mathbf 1\!\otimes\!\mathbf 1 && \text{体积罚，秩一}
\end{aligned}
$$

伴随算子（分部积分，周期边界，D6）：

$$
(\mathcal C^n_q)^*=\mathcal C^n_q\quad(\text{自伴}),\qquad
(\mathcal C^n_s)^*[h]=-\sqrt\epsilon\,\nabla\!\cdot\!(h\,\nabla\eta^n)
$$

$$
(\mathcal C^n_r)^*[h]=-\epsilon\,\nabla\!\cdot\!\big(\tfrac h{r^n}\nabla\phi^n\big)+\tfrac1{\epsilon r^n}\phi^n((\phi^n)^2-1)\,h
$$

右端 $\mathcal R^n_\phi$ 收集全部显式项：

$$
\begin{aligned}
\mathcal R^n_\phi=\;&\big(\tfrac1\tau+\sigma\big)\phi^n
-(\mathcal C^n_q)^*[q^n]
-(\mathcal C^n_r)^*[\Phi_\eta\,r^n]\\
&+M_1 v_d\\
&-\mathcal T^n_A-\mathcal T^n_D-\mathcal T^n_N\\
&-(\mathcal O-\mathcal S)[\phi^n]\quad(\text{变系数余项，}\mathcal O=\mathcal A^n_E+\mathcal A^n_L\text{ 的非骨架部分，见 §5.2})
\end{aligned}
$$

其中 $\mathcal T^n_A,\mathcal T^n_D,\mathcal T^n_N$ 为 Newton 线性化产生的二阶显式项（D4）。罚项 $\tfrac{M_j}2(C_j-c_j)^2$ 的 Hessian 为 $M_j(\mathrm dC_j)\otimes(\mathrm dC_j)+M_j(C_j-c_j)\mathrm d^2C_j$；第一项（半正定秩一）进 LHS，第二项 $M_j(C_j^n-c_j)\mathrm d^2C_j$ 取旧层显式入右端，即 $\mathcal T^n$。逐项写出（$\mathrm d^2C_j$ 作用在 $\phi^n$ 上）：

**面积** $A=\int W(\phi)dx$，$\mathrm d^2A$ 的密度为 $-\epsilon\Delta+\tfrac1\epsilon(3\phi^2-1)$：

$$
\mathcal T^n_A=M_2(A^n-a_0)\Big(-\epsilon\Delta\phi^n+\tfrac1\epsilon\big(3(\phi^n)^2-1\big)\phi^n\Big)
$$

**面积差** $D=\int\tanh(\eta^n/\xi)W(\phi)dx$，$\tanh(\eta^n/\xi)$ 为冻结权重：

$$
\mathcal T^n_D=M_3(D^n-a_d)\Big(-\epsilon\nabla\!\cdot\!\big(\tanh(\eta^n/\xi)\nabla\phi^n\big)
+\tfrac1\epsilon\big(3(\phi^n)^2-1\big)\phi^n\tanh(\eta^n/\xi)\Big)
$$

**正交** $N=\tfrac\epsilon2\int|\nabla\phi\cdot\nabla\eta|^2dx$，$\mathrm d^2N$ 的密度为 $-\epsilon\nabla\!\cdot\!\big((\nabla(\cdot)\cdot\nabla\eta^n)\nabla\eta^n\big)$：

$$
\mathcal T^n_N=M_4 N^n\Big(-\epsilon\,\nabla\!\cdot\!\big((\nabla\phi^n\cdot\nabla\eta^n)\nabla\eta^n\big)\Big)
$$

体积 $V$ 为线性泛函，$\mathrm d^2V=0$，无二阶显式项。

> 注：$\mathcal T^n$ 是精确项，非近似——Newton 线性化只是把 Hessian 分块（半正定主项进 LHS、二阶项进右端），稳态 $\delta^n\phi\to0$ 时不动点不变。唯一的近似是变系数余项 $(\mathcal O-\mathcal S)[\phi^n]$ 的旧层滞后（$O(\tau)$，见 §5.2）。

解出 $\delta^n\phi$ 后：$\phi^{n+1}=\phi^n+\delta^n\phi$，并由 §3.1 显式更新 $q^{n+1},r^{n+1},s^{n+1}$。

### §4.2 $\eta$-子步

固定 $\phi=\phi^{n+1}$。$q^{n+1},r^{n+1}$ 已定值，弯曲项 §2.2 第一行为纯已知量入右端。$\eta^{n+1}$ 满足

$$
\boxed{\;
\Big(\tfrac1\tau\mathcal I+\sigma\mathcal I
+\mathcal B^n_L+\mathcal B^n_D+\mathcal B^n_N+\mathcal B^n_P\Big)\,\delta^n\eta=\mathcal R^n_\eta
\;}
$$

$$
\begin{aligned}
\mathcal B^n_L[\psi] &= -\delta\xi\,\nabla\!\cdot\!\big(W(\phi^{n+1})\nabla\psi\big)
+\delta\tfrac1\xi W(\phi^{n+1})\,\zeta^n\,\psi && \text{线张力，扩散 + 零阶}\\
\mathcal B^n_D &= M_3\,f^n_D\!\otimes\!f^n_D, & f^n_D&=\tfrac1\xi W(\phi^{n+1})\,\mathrm{sech}^2(\eta^n/\xi) && \text{秩一}\\
\mathcal B^n_N &= M_4\,e^n_{N\eta}\!\otimes\!e^n_{N\eta}, & e^n_{N\eta}&=(\mathcal D^n_s)^*[s^{n+1}_{(\phi)}] && \text{秩一}\\
\mathcal B^n_P &= 4M_5\,e^n_P\!\otimes\!e^n_P, & e^n_P&=(\mathcal D^n_p)^*[p^n] && \text{秩一}
\end{aligned}
$$

$$
(\mathcal D^n_s)^*[h]=-\sqrt\epsilon\,\nabla\!\cdot\!(h\,\nabla\phi^{n+1}),\qquad
(\mathcal D^n_p)^*[h]=-\xi\,\nabla\!\cdot\!(h\,\nabla\eta^n)-\tfrac1\xi\eta^n\zeta^n\,h
$$

$s^{n+1}_{(\phi)}$ 为 $\phi$-子步结束时的 $s$ 值。$\mathcal B^n_L$ 零阶项 $\delta\tfrac1\xi W\zeta^n$ 在 $\zeta^n<0$ 时为负，由 $\sigma$ 压住正性（§5.3）。

右端：

$$
\begin{aligned}
\mathcal R^n_\eta=\;&\big(\tfrac1\tau+\sigma\big)\eta^n
-\tfrac{k'(\eta^n)}{2k(\eta^n)}(q^{n+1})^2\\
&-\mathcal T^n_{D\eta}-\mathcal T^n_{N\eta}-\mathcal T^n_{P\eta}\\
&-(\mathcal O-\mathcal S)[\eta^n]\quad(\text{变系数余项，}\mathcal O=\mathcal B^n_L\text{ 的非骨架部分，见 §5.2})
\end{aligned}
$$

二阶显式项（$\mathrm d^2C_j$ 作用在 $\eta^n$ 上，$W(\phi^{n+1})$ 为冻结权重）：

**面积差** $D=\int\tanh(\eta/\xi)W(\phi^{n+1})dx$，$\mathrm d^2D$ 密度 $-\tfrac2{\xi^2}\tanh(\eta/\xi)\,\mathrm{sech}^2(\eta/\xi)\,W(\phi^{n+1})$：

$$
\mathcal T^n_{D\eta}=M_3(D^n-a_d)\Big(-\tfrac2{\xi^2}\tanh(\eta^n/\xi)\,\mathrm{sech}^2(\eta^n/\xi)\,W(\phi^{n+1})\Big)
$$

**正交** $N=\tfrac\epsilon2\int|\nabla\phi^{n+1}\cdot\nabla\eta|^2dx$，$\mathrm d^2N$ 密度 $-\epsilon\nabla\!\cdot\!\big((\nabla(\cdot)\cdot\nabla\phi^{n+1})\nabla\phi^{n+1}\big)$：

$$
\mathcal T^n_{N\eta}=M_4 N^n\Big(-\epsilon\,\nabla\!\cdot\!\big((\nabla\phi^{n+1}\cdot\nabla\eta^n)\nabla\phi^{n+1}\big)\Big)
$$

**剖面正则** $P=\|p\|^2$。$\tfrac{M_5}2P^2$ 的 Hessian 半正定主项 $4M_5\,e^n_P\otimes e^n_P$ 已进 LHS（$\mathcal B^n_P$）；其余含 $P^n$ 的项（来自 $\mathrm d^2P$ 及 $\mathrm d(\|p\|^2)$ 的非主部）取旧层显式：

$$
\mathcal T^n_{P\eta}=M_5\big(\text{含 }P^n,p^n\text{ 的二阶显式项}\big)
$$

> $\mathcal T^n_{P\eta}$ 因 $P=\|p\|^2$ 经 $p$ 两层复合、展开冗长，实现时按"Hessian 减去已入 LHS 的秩一主项"取差即可，无需手写闭式。其余 $\mathcal T^n$ 同 §4.1 注：均为精确项，唯一近似仍是变系数余项的旧层滞后。

解出 $\delta^n\eta$ 后：$\eta^{n+1}=\eta^n+\delta^n\eta$，由 §3.1 更新 $p^{n+1},s^{n+1}$。

---

## §5 谱空间求解

### §5.1 困难：变系数非对角

§4 的隐式算子含变系数（$a(\eta^n)^2,\Phi_\eta,W(\phi^{n+1}),1/r^n,\nabla\phi^n$）。变系数乘法在 Fourier 空间为卷积、非对角，不能逐波数求逆。

### §5.2 劈分：常系数骨架（隐式）+ 余项（显式）

用 $(\mathcal O-\mathcal S)$ 残差策略（stabilized semi-implicit / IMEX splitting，标准方法，见 Chen & Shen, *Comput. Phys. Commun.* 108 (1998) 147；Eyre, MRS Proc. 529 (1998)）。微分算子 $\mathcal O$ 劈成常系数骨架 $\mathcal S$（谱对角）+ 余项：

$$
\mathcal O[\phi^{n+1}]=\mathcal S[\phi^{n+1}]+(\mathcal O-\mathcal S)[\phi^{n+1}]
\approx\mathcal S[\phi^{n+1}]+(\mathcal O-\mathcal S)[\phi^n]
$$

余项 $-(\mathcal O-\mathcal S)[\phi^n]$ 滞后旧层入右端，引入 $O(\tau)$ 误差。

只有微分型算子进骨架（$\mathcal A^n_E,\mathcal A^n_L$ 和 $\mathcal B^n_L$）；秩一算子 $\mathcal A^n_{A,D,N,V},\mathcal B^n_{D,N,P}$ 全走 Woodbury（§5.3）。

骨架谱符号中的代表常数 $\bar a^2,\bar g',b_L,\bar W,\bar\zeta$ 由变系数取**空间平均**得到，每步重算 —— 它不是超参数（"取平均"这一规则固定，规则算出的数值随当前场每步变化）：

$$
\widehat{\mathcal L}(\mathbf k)=\tfrac1\tau+\sigma
+\bar a^2\big(\epsilon|\mathbf k|^2-\bar g'\big)^2+b_L|\mathbf k|^2
$$

$$
\widehat{\mathcal M}(\mathbf k)=\tfrac1\tau+\sigma
+\delta\xi\bar W|\mathbf k|^2+\delta\tfrac1\xi\bar W\bar\zeta
$$

$$
\bar a^2=\overline{k(\eta^n)}/\epsilon,\quad
b_L=\epsilon^2\,\mathrm{rep}\big(\Phi_\eta|\nabla\phi^n|^2/(r^n)^2\big),\quad
\bar W=\overline{W(\phi^{n+1})},\quad\bar\zeta=\overline{(\eta^n)^2-1}
$$

正性：$\widehat{\mathcal L}>0$ 无条件；$\widehat{\mathcal M}>0$ 要求 $\sigma>\delta\bar W|\bar\zeta|/\xi-1/\tau$。

### §5.3 秩一项：Woodbury

体积、面积、面积差、正交罚项的隐式算子是**秩一**的。§4 的左端整体形如"对角骨架 $\mathcal D$ + 若干秩一修正"，用 Woodbury 公式**精确**求逆（直接法，非迭代、无近似误差）。

LHS 写成 $\mathcal D+\sum_j\alpha_j u_j u_j^\top$，$\mathcal D$ 对角（符号 $\widehat{\mathcal L}$ 或 $\widehat{\mathcal M}$），$u_j$ 秩一向量，$\alpha_j\ge0$：

$$
(\mathcal D+U\alpha U^\top)^{-1}\mathcal R
=\mathcal D^{-1}\mathcal R-\mathcal D^{-1}U\big(\alpha^{-1}+U^\top\mathcal D^{-1}U\big)^{-1}U^\top\mathcal D^{-1}\mathcal R
$$

每子步代价：$m+1$ 次 FFT-除（$m\le4$）+ 一个 $m\times m$ 稠密小解。

### §5.4 求解流程（$\phi$-子步，$\eta$-子步同构）

```
1. 实空间组装右端 R（含变系数余项、Newton 二阶显式项、Eyre 显式项）
2. FFT(R)
3. 逐波数除 L̂(k)   ⇒ D⁻¹R     （η-子步除 M̂(k)）
4. 对全部秩一项作 Woodbury 修正
5. IFFT  ⇒ δ ⁿφ
6. φⁿ⁺¹ = φⁿ + δ ⁿφ
7. 由 §3.1 更新 qⁿ⁺¹, rⁿ⁺¹, sⁿ⁺¹
```

变系数余项滞后引入 $O(\tau)$ 误差；$\tau$ 受限时改以 $\widehat{\mathcal L}^{-1}$ 为预条件子作不动点 / CG 迭代细化。

---

## §6 罚因子 continuation

$M_i$ 大则约束准但系统刚。从小值起，每内层收敛一次 $M_i\leftarrow\rho M_i$（$\rho\sim2$–$10$）并以当前解重启，直至约束残差 $|V-v_d|,|A-a_0|,|D-a_d|$ 与 $N,P$ 均 $<$ 容差。该策略是 **penalty method（罚方法）** 与 **continuation / homotopy method** 的标准组合，并非本模型特有发明。各层依据：

- **罚方法**：用 $\tfrac{M_i}2(C_i-c_i)^2$ 把约束并入能量，$M_i\to\infty$ 时罚问题的解收敛到真约束问题的解。标准优化教材，如 Nocedal & Wright, *Numerical Optimization*, 2nd ed., Springer 2006, Ch. 17（quadratic penalty method）。$M_i$ 增大时系统病态/刚性加剧亦在该章讨论。
- **本模型中的罚方法**：Wang & Du, *J. Math. Biol.* 56 (2008) 347–371，§2 公式 (15)；其中定义 $\Lambda_i=M_i(C_i-c_i)$，并验证 $M_i\to\infty$ 时 $\Lambda_i\to\lambda_i$（真 Lagrange 乘子）——见该文 Table 1。
- **continuation / 逐步增大 $M$ 并暖启动**：即"沿参数逐步移动、每步用上一步的解作初值"，是 continuation method 的标准思想（Nocedal & Wright 同书 Ch. 11 引言对 homotopy/continuation 的描述；Allgower & Georg, *Numerical Continuation Methods*, Springer 1990）。本算法把"参数"取为罚因子 $M$。

几何增长 $M_i\leftarrow\rho M_i$ 中的 $\rho$、$M$ 初值、目标容差均为**经验量**（`IEQ_scheme_vesicle.md` §5.8 列为待标定）。权衡：$\rho$ 小 → 每步刚性增幅小、暖启动易、但级数长、总步数多；$\rho$ 大 → 级数短、但每步刚性跳变大、内层迭代增多。最优 $\rho$ 依问题而定，需标定。

---

## §7 完整算法

```
输入: φ⁰, η⁰; ε=ξ, δ, c₁=c₂=0; v_d, a₀, a_d;
      M₁..M₅; τ; σ; C₀; tol; k,c (k>|c|>0); ρ, M_max

初始化辅助变量（代定义）:
  q⁰ = a(η⁰)(εΔφ⁰ + g(φ⁰))
  r⁰ = √(2W(φ⁰) + C₀)
  s⁰ = √ε (∇φ⁰·∇η⁰)
  p⁰ = (ξ/2)|∇η⁰|² − (1/4ξ)((η⁰)²−1)²
预计算: 波数 k, 谱符号 L̂(k), M̂(k)

while 约束残差 > outer_tol and M < M_max:        # continuation
  for n = 0,1,2,... :                            # 内层至稳态
    ── φ-子步 (固定 η=ηⁿ) ──
      组装 §4.1 算子与 R_φ
      §5.4 求解 ⇒ δ ⁿφ,  φⁿ⁺¹ = φⁿ + δ ⁿφ
      更新 qⁿ⁺¹, rⁿ⁺¹, sⁿ⁺¹                     (§3.1)
    ── η-子步 (固定 φ=φⁿ⁺¹) ──
      组装 §4.2 算子与 R_η
      §5.4 求解 ⇒ δ ⁿη,  ηⁿ⁺¹ = ηⁿ + δ ⁿη
      更新 pⁿ⁺¹, sⁿ⁺¹                            (§3.1)
    if ‖δ ⁿφ‖ + ‖δ ⁿη‖ < tol : break             # 内层收敛
  M ← min(ρ·M, M_max)

输出: 稳态 (φ*, η*)
  Γ={φ*=0}, Γ⊥={η*=0}, γ₀={φ*=η*=0}
```

---

## §8 超参数

| 量 | 出处 | 取法 |
|---|---|---|
| $C_0>0$ | §3 $r$ 正则常数 | $1/r^n$ 下界 $\sqrt{C_0}$ |
| $\sigma>0$ | D5 Eyre | 起点 $\sigma\gtrsim\delta\bar W|\bar\zeta|/\xi$ |
| $\tau$ | 时间步 | 实验定可用范围 |
| $M_i$ 初值, $\rho$, 容差 | §6 continuation | 实验定 |

