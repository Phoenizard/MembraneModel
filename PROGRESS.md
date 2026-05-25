# PROGRESS — MembraneModel

> 项目进度日志。按时间倒序，最新在最上。每条记录的是**已发生**的事实，不是 todo。
> 维护约定见 `CLAUDE.md`。

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
