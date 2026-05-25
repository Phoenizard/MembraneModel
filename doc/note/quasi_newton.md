# 拟牛顿求解器 —— 算法与离散化

> 把梯度流 $\phi_t=-\delta E_M/\delta\phi,\ \eta_t=-\delta E_M/\delta\eta$ 积分到稳态的**预条件修正 Newton**格式。
> 梯度（右端）$\delta E_M/\delta\phi,\ \delta E_M/\delta\eta$ 见 [`model_variance.md`](model_variance.md)。
> 代码：[`solver/quasi_newton.py`](../../solver/quasi_newton.py)、[`solver/woodbury.py`](../../solver/woodbury.py)、[`discretization/grid.py`](../../discretization/grid.py)。
>
> 非经典 IEQ：**不引辅助变量**。不动点恒为 $\delta E_M/\delta\phi=\delta E_M/\delta\eta=0$（即原 $E_M$ 临界点），与下面预条件子的取法无关。

## 1. 递推：半隐式梯度流 → 拟牛顿步

记一般场 $w\in\{\phi,\eta\}$（子步内另一场冻结）。对梯度流做半隐式 Euler，并把梯度线性化 $\delta E_M/\delta w\approx G(w^n)+\mathcal H\,\delta w$，其中 $G$ 为精确梯度、$\mathcal H$ 为 Hessian 近似：

$$
\frac{w^{n+1}-w^n}{\tau}=-\frac{\delta\mathcal E_M}{\delta w}
\;\Longrightarrow\;
\Big(\tfrac1\tau\mathcal I+\mathcal H\Big)\,\delta w=-G
\;\Longrightarrow\;
\boxed{\ w^{n+1}=w^n-(P+H)^{-1}G\ }
$$

其中 $P+H$ 扮演 $\tfrac1\tau\mathcal I+\mathcal H$：$P$ 是弯曲 + 线张力 Hessian 的常系数谱骨架（含 Eyre），$H$ 是罚项 Hessian 的秩一 SPD 主部。$G$ 为当前场上现算的精确梯度（含全部罚项力，对应 `dEM_dphi`、`dEM_deta`）。

**不动点**：$\delta w=0 \Longleftrightarrow G=0$，与近似 $\mathcal H\approx P+H$ 无关。**下降方向**：$P+H$ 对称正定，故 $\langle G,-(P+H)^{-1}G\rangle<0$。

子步解耦：先 $\phi$ 子步（固定 $\eta^n$），后 $\eta$ 子步（固定 $\phi^{n+1}$）。

## 2. 离散化（本算法专属）

### 2.1 空间：Fourier 谱

周期盒 $\Omega=[-\pi,\pi]^3$，$N^3$ 网格。微分算子对角化：$\nabla\to i\mathbf k$、$\Delta\to-|\mathbf k|^2$。积分与内积用网格体元 $dV=h^3$。

代码 `grid.py`：`grad, lap, div, integral, inner, apply_symbol, solve_symbol`。

### 2.2 预条件子 P（常系数对角骨架）

把变系数算子冻结成空间平均常数，得对角符号。$\phi$ 子步：

$$
\widehat{\mathcal L}(\mathbf k)=\Big(\tfrac1\tau+\sigma\Big)+\bar a^2\big(\epsilon|\mathbf k|^2-\bar g'\big)^2+b_L|\mathbf k|^2,
\qquad \bar a^2=\overline{k(\eta)}/\epsilon,\ \ \bar g'=\overline{g'(\phi)},\ \ b_L=\epsilon\,\overline{\Phi_\eta}.
$$

$\eta$ 子步：

$$
\widehat{\mathcal M}(\mathbf k)=\Big(\tfrac1\tau+\sigma\Big)+\delta\xi\,\overline{W}\,|\mathbf k|^2.
$$

Eyre 常数 $\sigma$ 在此作阻尼：增大 $P$，缩小步长，增稳。代码 `quasi_newton.py`：`_phi_precond`、`_eta_precond`。

### 2.3 秩一罚项 H + Woodbury

约束罚项 Hessian 的半正定主部为 $\sum_j M_j(\delta C_j)\otimes(\delta C_j)$。取方向 $u_j=\delta C_j/\delta w$（见 [`model_variance.md`](model_variance.md) §2.3），$\alpha_j=M_j$：

$$
H=\sum_j \alpha_j\,u_j u_j^{\!\top},\qquad
(P+H)^{-1}b=\mathcal D^{-1}b-\mathcal D^{-1}U\big(\alpha^{-1}+U^{\!\top}\mathcal D^{-1}U\big)^{-1}U^{\!\top}\mathcal D^{-1}b .
$$

其中 $\mathcal D$ 为对角算子（$\widehat{\mathcal L}$ 或 $\widehat{\mathcal M}$）。$\phi$ 子步取 $u_j\in\{\delta V,\delta A,\delta D,\delta N\}/\delta\phi$、$\alpha=(M_1,M_2,M_3,M_4)$；$\eta$ 子步取 $\{\delta D,\delta N,\delta P\}/\delta\eta$、$\alpha=(M_3,M_4,M_5)$；$M_j=0$ 的项自动剔除。每子步 $m+1$ 次 FFT 求逆加一个 $m\times m$ 小稠密解。

代码 `woodbury.py`：`solve`；`quasi_newton.py`：`phi_step`、`eta_step` 的 `us/alphas`。

## 3. 外层控制

- **continuation**：$M_1=M_2=M_3$ 从 `M_start` 起，每内层收敛一次乘 $\rho$，至约束残差小于 `outer_tol` 或到 `M_max`；$M_4=M_5$（正则）固定。
- **能量平台收敛**：每 `log_every` 步采样 $E_M$，连续 `patience` 次不再创新低（相对 `e_rtol`）即停；$|d|$ 全程近常数，不用作判据。

代码 `quasi_newton.py`：`continuation`、`steady_solve`。

## 4. 流程图

```
输入: φ⁰=sphere(v_d), η⁰=tanh(z/√2ε)          # initial.py；无辅助变量
M ← M_start
repeat (continuation):
  令 M1=M2=M3=M
  for step = 1..max_inner (steady_solve):
    # φ 子步（固定 η）
    G = δE_M/δφ(φ,η)                           # dEM_dphi（精确梯度）
    P = L̂(k; 冻结均值);  H = Σ M_j u_j⊗u_j       # u_j = δC_j/δφ
    φ ← φ − Woodbury(P, G, {u_j, M_j})         # = φ − (P+H)⁻¹ G
    # η 子步（固定 φ=新值）
    G = δE_M/δη(φ,η)                           # dEM_deta
    P = M̂(k);  H = Σ M_j u_j⊗u_j
    η ← η − Woodbury(P, G, {u_j, M_j})
    每 log_every 步: 采样 E_M → 能量平台判收敛
  if 残差 < outer_tol or M ≥ M_max: break
  M ← min(M·ρ, M_max)
输出: 稳态 (φ*, η*)
```

## 5. 性质与脚注

- **稳态保真**：不动点等于原 $E_M$ 临界点，不受 $P,H$ 近似影响；正确性只依赖被 FD 验证的梯度。
- **条件稳定**（非无条件能量稳定）：需 $\sigma\gtrsim\delta\bar W|\bar\zeta|/\xi$（保 $\widehat{\mathcal M}>0$，$\bar\zeta$ 为 $(\eta^2-1)$ 的均值）且 $\tau$ 须标定（exp1：N64 上 $\tau\ge 5\times10^{-4}$ 即 NaN，取 $\tau=2\times10^{-4}$、$\sigma=2$）。
- 预条件子细节（仅影响收敛速度、不影响稳态）：$b_L=\epsilon\cdot\mathrm{mean}(\Phi_\eta)$（各向同性代表值）；$\widehat{\mathcal M}$ 只保留扩散主部，未含零阶项 $\delta\tfrac1\xi\bar W\bar\zeta$。
