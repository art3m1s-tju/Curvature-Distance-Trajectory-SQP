# Curvature-Distance-Trajectory-SQP

## 项目简介 (Description)
Global time-optimal racing trajectory generation via sparse Sequential Quadratic Programming (OSQP solver).
本项目是对《Global minimum time trajectory planning considering curvature and distance for track racing》论文算法的完整 Python 复现，并采用现代高效的 OSQP 稀疏二次规划求解器实现了核心的 SQP 迭代。

## 项目背景 (Project Background)
论文## 最新改进与功能 (Latest Improvements)

本项目最近进行了重大更新，以更精准地复现论文描述的物理约束和数值稳定性：

### 1. 严格复现物理占用约束 (Formula 23 & 24)
- **动态车身宽度计算 (w_v)**：实现了论文公式 (23)，根据车辆的长宽 (`l_v`, `b_v`) 以及当前轨迹的实时航向角，动态计算车辆在赛道横截面上的投影宽度。
- **边界法向量投影约束**：实现了论文公式 (24)，通过计算赛道内、外边界的单位法向量 ($n_I, n_O$)，将车辆宽度约束投影到从内边界指向外边界的方向向量 $v$ 上，从而精确定义每个决策点的 $\alpha$ 允许范围。
- **论文公式纠错**：在复现过程中，我们发现并修正了原论文公式 (24) 中的一个严重数学笔误（原公式分子多乘了一个 $|v_i|$ 导致量纲错误），修正后轨迹不再被困在赛道中心，而是能够完美实现切弯。

### 2. 数值稳定性优化 (SQP Relaxation)
- **松弛步长控制 (Damping)**：为了解决纯曲率优化 ($\epsilon=0$) 时可能出现的振荡不收敛问题，在 SQP 迭代中引入了松弛因子 $\gamma=0.5$。每一轮的更新量只采用计算结果的一半，有效抑制了非线性项引起的跳变。
- **高维稀疏矩阵处理**：利用 `scipy.sparse` 对所有 Hessian 和 Jacobi 矩阵进行稀疏化存储，确保即使在 3000+ 控制点的大型赛道上也能实现秒级求解。

### 3. 可视化升级
- **多维度细节图**：输出的 `trajectory_comparison.png` 升级为 24x20 高清大图，包含：
  - **全赛道概览 (Full Track Overview)**
  - **关键弯道放大细节 (Corner Zoom-in)**
  - **Alpha 决策变量分布 (Alpha Distribution)**：直观展示赛车在不同路段相对于赛道中心的偏离程度。

## 数学与代码的映射关系 (Code & Math Mapping)

1. **车辆物理尺寸 (Section 3.6)**:
   涉及函数 `calculate_wv_per_point`。考虑了车辆对角线长度及其在横向截面的投影。

2. **精确 alpha 边界 (Section 3.7)**:
   涉及函数 `calculate_boundary_normals` 和 `optimize_trajectory` 内部循环。计算了由公式 (24) 定义的非线性变化上下界。

... (其他映射关系同前)

## 运行方式 (Usage)

1. 确保已安装依赖：`pip install numpy pandas scipy osqp matplotlib`
2. 运行脚本：`python reproduce_paper.py`
3. 检查生成的 `trajectory_comparison.png`

## 分析与讨论 (Discussion)
通过最新的高清对比图可以看到：
- **最小曲率轨迹 (Red Line)**：表现出极端的切弯倾向。由于公式 (24) 的精确约束，车辆会在不发生碰撞的前提下，尽早进入弯心并尽可能晚地贴合外侧，以保持最大的弯道半径。
- **平衡轨迹 (Blue Line)**：权衡了路程和曲率。在长直道段更倾向于走几何中心，而在急弯处则会采取适度的切弯策略。
- **Alpha 分布**：可以观察到在直道处 $\alpha$ 趋向 0.5，而在左转弯处 $\alpha$ 迅速变化以贴合内侧边界，证明了非线性 SQP 的有效性。

