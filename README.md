# Curvature-Distance-Trajectory-SQP

## 项目简介 (Description)
Global time-optimal racing trajectory generation via sparse Sequential Quadratic Programming (OSQP solver).
本项目是对《Global minimum time trajectory planning considering curvature and distance for track racing》论文算法的完整 Python 复现，并采用现代高效的 OSQP 稀疏二次规划求解器实现了核心的 SQP 迭代。

## 项目背景 (Project Background)
论文提出了一种基于离线高精度赛道地图的**全局最小时间赛车轨迹规划方法**。主要创新点如下：
- 在目标函数中同时考虑了**曲率因子**和**距离因子**。
- 提供了一种新的基于二次规划 (QP) 形式的高效求解范式。
- 对不同的汽车动力学表现，通过对权重 $\epsilon$ 进行平衡，可以计算得到不同车辆特征下的全局最优时间轨迹。

本代码复现了其核心的三次样条曲率近似矩阵、前向有限差分距离矩阵，以及由于变量 $\alpha$ 位于 0 和 1 之间的多重非线性 SQP（序列二次规划）迭代流程，并完整地将 MATLAB 的 `quadprog` 逻辑替换为了高效且现代的 `OSQP` 稀疏二次规划求解器。

## 文件说明 (File Descriptions)

- `reproduce_paper.py`: 这是算法的核心代码。我们使用了 `numpy`, `scipy.sparse` 与 `pandas` 等库完成了论文从公式导出到矩阵构建的全部过程。
- `track.csv`: 赛道的离线边界数据（包含 left_border, right_border 数据列）。
- `trajectory_comparison.png`: 脚本运行后输出的赛车线可视化对比图，可直观看到纯最小曲率模型和综合平衡模型下赛车起始终点附近的切弯与走线的策略差异。
- `README.md`: 项目的说明与使用文档。

## 环境与依赖项 (Dependencies)

本项目要求运行在 Python 环境下，推荐使用 Python 3.8+。
算法运行依赖以下科学计算与作图包：

```bash
pip install numpy pandas scipy osqp matplotlib
```
_注：OSQP 是非常强大的专门用来求解大规模稀疏二次规划问题的开源求解器，计算效率极高。_

## 数学与代码的映射关系 (Code & Math Mapping)

你可以直接阅读 `reproduce_paper.py` 中的每一行，我们在核心函数上方标注了对应的物理意义和公式。基本流程与论文中第3节的理论紧密相扣：

1. **赛车线点的符号表示 (Section 3.1)**:
   赛车线由 $\alpha$ 参数控制，实际坐标 $r_i = p_i + \alpha_i v_i$ 位于内外边界线之间。

2. **计算距离因子 $J_s$ (Section 3.2)**:
   涉及代码函数 `calculate_distance_factor` 与 `build_difference_matrix`。我们推导并展开后，得到了二次矩阵方程所需的一阶和二阶项 $H_s, f_s$。

3. **计算曲率因子 $J_k$ (Section 3.3)**:
   这里最关键的是使用三次样条平滑差分。代码函数 `calculate_derivative_matrices` 通过参考线先构建出加权矩阵 $T_{xx}, T_{yy}, T_{xy}$ 以及二阶导数映射矩阵 $M$。随后代入 `calculate_curvature_factor` 中构建曲率对应的二次函数目标矩阵 $H_k, f_k$。

4. **序列二次规划迭代 (Section 3.4 & 3.5)**:
   由 `optimize_trajectory` 函数呈现出。合并总目标矩阵 $P = H_k + \epsilon H_s$ 并在安全边界约束下（考虑车宽尺寸限制 $\alpha$ 为 $[0.05, 0.95]$）交由 OSQP 求解。将每次求解出来的新 $\alpha$ 再代入去重新计算基于新参考线的 $M, T$ 计算下一代轨迹，迭代至结果收敛。

## 运行方式 (Usage)

直接运行 Python 脚本：
```bash
python reproduce_paper.py
```

运行后会在控制台打印每次 SQP 迭代过程中变量 $\alpha$ 的变化差异（这证明算法正在进行平滑收敛），并分别输出：
1. **最小曲率轨迹 ($\epsilon = 0$)**: 展现纯粹关注转长弯但不计较行驶路程的数学特性。
2. **平衡权重轨迹 ($\epsilon = 10$)**: 展现为了减少路程距离，算法愿意在局部接受稍微急促变向（转急弯）的情况。

执行完毕后会自动在当前目录生成并保存 `trajectory_comparison.png` 对比图景。

## 分析与讨论 (Discussion)
从保存的可视化轨迹图 `trajectory_comparison.png` 中可以看到：
- 在同样的内外边界限制（黑色实线）内，**紫色/红色虚线 (最小曲率)** 极大程度地利用了道路的全部宽度，从最外侧入弯后强力切入最内侧弯心出弯，此举平滑了整车行驶的 G 力。
- **蓝色点划线 (距离平衡)** 虽然也切弯，但更倾向走赛道的直线几何最短距离。这是由于权重系数 $\epsilon=10$ 造成的优化妥协。
- 这些结果完美符合作者在论文第四节（图7）中对于曲率、距离权重倾向（Vehicle performance 车辆极限）的理论陈述与计算机仿真表现。
