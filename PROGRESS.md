# PROGRESS — MembraneModel

> 项目进度日志。按时间倒序，最新在最上。每条记录的是**已发生**的事实，不是 todo。
> 维护约定见 `CLAUDE.md`。

---

## 2026-05-25 — 笔记重组 + 厘清"代码非 IEQ"

- **确认 `solver/ieq.py`（commit `1250c06`，full `1250c061b276bd5d526cee3e4988063d4dd95c1a`）并非经典 IEQ**：不存储/演化辅助变量 `q,r,s,p`，实为对 `E_M` 的预条件修正 Newton / 梯度下降（不动点 = 原 `E_M` 真临界点）。据此重命名 `ieq.py→quasi_newton.py`、`run_IEQ.py→run_solver.py`，并更新 docstring/引用。
- **笔记按"算法无关 / 算法专属"分离**：新建 `doc/note/model_variance.md`（能量 + 变分导数，所有算法共享）与 `doc/note/quasi_newton.md`（当前拟牛顿格式 + 其离散化）。旧笔记 `IEQ_scheme_vesicle.md`、`IEQ_deployed_vs_classic.md` 暂留待手动清理。
- **整理经典 IEQ 算法笔记 `doc/note/IEQ_algorithm.md`**：完整推导（变分导数 / 二次化 / 辅助变量演化 / 时间格式 / 谱+Woodbury / continuation），经数学校核修正（伴随 `C_q*`、右端取旧层精确负梯度、去除错误的 Newton 二阶显式项）。**下一步候选**：将经典 IEQ 部署为代码，与现有拟牛顿求解器并列对比。

---

## 2026-05-25 — Initial Test：代码可行性 + 模型合理性

- 测试代码可行性与模型合理性；目标是跑通 IEQ 算法下的模型解（N64）。具体参数见 [doc/exp/exp1/exp1.md](doc/exp/exp1/exp1.md)。

---

## 2026-05-22 — 项目启动

- 仓库初始化：搭建 `CLAUDE.md`（项目总纲 + 符号约定 + 协作约定）与本进度日志
- 同步搭建 Claude 项目记忆（user / project / reference / feedback）
- 已收集材料：
  - 主参考论文：Wang & Du 2008, *J. Comput. Phys.*（`paper/2007-Qiang-Du-multi-component lipid membranes.pdf`）
  - Sec.3 实验清单中文整理：`section3_experiments_zh.md`
- 当前阶段：**文献阅读 + 选题**，具体研究课题未定
- 直接目标：复现论文 Sec.3 全部实验
- 下一步候选：精读 Sec.1–2（模型推导：能量泛函 (15)、梯度流 (16)、惩罚/正则项设计）后再决定首个复现目标（候选：Fig.1 球→葫芦的定性演示，或 Fig.2 收敛性测试）

---

<!--
条目模板：

## YYYY-MM-DD — 一句话标题

- 做了什么 / 发生了什么
- 关键发现 / 决策（含 why）
- 下一步候选（不是承诺）
-->
