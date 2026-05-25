# MembraneModel

二组分脂质囊泡的相场模型,用 **IEQ**(不变能量二次化)一阶解耦格式 + 谱方法(FFT)求解,复现 Wang & Du 2008 (*J. Comput. Phys.*) Sec.3。

两个相场:`phi` 刻画膜内/外(界面层=膜),`eta` 刻画膜上两组分。能量 `E_M(phi,eta)` = 弯曲 + 线张力 + 约束惩罚(体积/面积/面积差)+ 正则项。

## 环境

普通 conda 环境,不打包。首次配置:

```bash
conda create -n membrane python=3.12
conda activate membrane
pip install numpy scipy matplotlib pytest scikit-image
conda env config vars set PYTHONPATH=$(pwd) -n membrane   # 让 import 找到根目录模块
conda activate membrane                                   # 重新激活使其生效
```

之后每次开终端 `conda activate membrane` 即可。

## 目录

```
.
├─ discretization/         谱网格:FFT 求导 / 积分 / 内积、谱符号求解
├─ model/
│  ├─ fields.py            逐点基元 g, g', W, m
│  └─ energy.py            能量 E_M 与变分导数 δE/δφ、δE/δη
├─ solver/
│  ├─ quasi_newton.py      拟牛顿步进 / continuation
│  └─ woodbury.py          秩-1 修正线性求解
├─ params.py               参数定义(Params dataclass)
├─ initial.py              初值:球形 φ⁰ + 上下两相 η⁰
├─ results.py              结果落盘:.npy 场 + 同名 .txt 参数表
├─ viz.py                  出图函数库
├─ run_solver.py           计算驱动(命令行启动求解)
├─ visualize.py            出图驱动
├─ tests/                  梯度有限差分校验 + 谱算子精度校验
├─ experiments/exp1/out/   产物:.npy / .txt / .png
└─ doc/                    论文 / 推导笔记 / 实验规格
```

库内核(`model` / `solver` / `discretization`)与驱动(`run_solver` / `visualize`)并列在根目录。

## 跑一次实验

```bash
# smoke:32^3 小算例,前台,实时进度
python -u run_solver.py --quick --name smoke --log 20

# 正式算例:后台 + 写日志文件
python -u run_solver.py --N 32 --eps 0.1964 --k 1.0 --delta 10 \
    --name exp1-1-N32 --log 50 > experiments/exp1/out/exp1-1-N32.run.log 2>&1 &
tail -f experiments/exp1/out/exp1-1-N32.run.log   # 查看进度
```

`-u` 无缓冲、`--log N` 每 N 步打印,二者保证进度可实时查看。产物写入 `experiments/exp1/out/`:`{name}_phi.npy`、`{name}_eta.npy`、`{name}.txt`。

出图验收:

```bash
python visualize.py --name exp1-1-N32     # -> {name}_summary.png 等
```

### 进度行字段

`E_M` 总能量(收敛应平稳);`V-v_d / A-a_0 / D-a_d` 三约束残差(应→0);`|d|` 每步场变化量(应降到 `inner_tol`)。`E_M` 震荡且 `|d|` 卡住 = 未收敛。

### 常用 flag

| flag | 默认 | 作用 |
|---|---|---|
| `--N` | 64 | 网格分辨率 |
| `--eps` | 0.1964 | 界面宽度 |
| `--k` / `--c` | 1.0 / 0.0 | 弯曲刚度 / 两相刚度差 |
| `--delta` | 10.0 | 线张力 |
| `--tau` / `--sigma` | 1e-3 / 2.0 | 时间步 / 稳定化 |
| `--max_inner` | 2500 | 每个惩罚档最多迭代步 |
| `--name` | exp1-1-N{N} | 产物 basename |
| `--log` | 0 | 每几步打印(0=不打印) |

`v_d/a_0/a_d`(几何目标)与 continuation 的 `M_start/M_max/rho` 目前写死在 `run_solver.py`,需改这些再编辑脚本。

## 测试

```bash
python -m pytest -q     # 梯度有限差分校验 + 谱算子精度校验
```

## 文档

`doc/note/IEQ_scheme_vesicle.md` 为完整数值方案推导(能量→变分导数→时间格式→谱离散),配合代码对照阅读。
