# Global Time-Optimal Racing Trajectory Planning via SQP

## 1. 项目简介 (Overview)

本项目是对顶级赛车轨迹规划算法《Global minimum time trajectory planning considering curvature and distance for track racing》的深度 Python 复现。

核心采用 **序列二次规划 (Sequential Quadratic Programming, SQP)** 框架，结合 **OSQP (Operator Splitting Quadratic Program)** 这一现代、高效的稀疏二次规划求解器，实现了针对大规模赛道地图（3000+ 控制点）的秒级最优轨迹生成。

## 2. 核心数学原理 (Mathematical Foundations)

本项目不仅实现了基本的样条曲率优化，还完整还原了论文中复杂的物理占用约束。

### 2.1 目标函数构成 (Cost Function)
优化目标由两部分组成：
$$J = J_k + \epsilon J_s$$
- **$J_k$ (Curvature Factor)**: 最小化路径曲率。通过三次样条 (Cubic Splines) 近似二阶导数矩阵 $M$，从而将曲率惩罚转化为关于决策变量 $\alpha$ 的二次型 $1/2 \alpha^T H_k \alpha + f_k^T \alpha$。
- **$J_s$ (Distance Factor)**: 最小化行驶路程。通过一阶差分矩阵 $A$ 计算相邻点间距离。
- **$\epsilon$ (Weight)**: 平衡因子。$\epsilon=0$ 为极致切弯（最小曲率线），$\epsilon > 1000$ 为极致路程节省（几何中心线）。

### 2.2 物理占用约束与公式纠错 (Exact Physical Constraints)

本实现严格执行了论文第 3.6 和 3.7 节的复杂几何约束：

#### 2.2.1 动态有效车宽 (Formula 23)
考虑车辆在弯道中的侧向投影，有效宽度 $w_v$ 为：
$$w_v = \frac{\sqrt{l_v^2 + b_v^2}}{2} |\cos(\theta_{vd} - \theta_{veh})|$$
代码通过实时的轨迹切线斜率动态计算 `angle_vehicle`，从而确保在每一个迭代步，车身占用的赛道面积都是精确计算的。

#### 2.2.2 投影约束边界 (Formula 24)
为了将三维车辆约束投影到二维 $\alpha$ 比例系数上，我们通过计算边界法向量 $n_I, n_O$：
$$\frac{w_v + w_s}{n_{Ii} \cdot v_i} \le a_i \le 1 - \frac{w_v + w_s}{n_{Oi} \cdot v_i}$$

> [!IMPORTANT]
> **公式修正说明**：原论文公式 (24) 分子中包含 $|v_i|$ 乘子。在复现过程中我们通过量纲分析证实这是一个**印刷错误**。按原公式会导致 $\alpha$ 约束远超 0-1 范围，导致车辆被死锁在中心。本项目已移除该错误乘子，完美恢复了“外-内-外”走线。

### 2.3 SQP 稳定性优化 (Damping)
为了解决非线性耦合带来的迭代振荡，我们在每轮迭代引入了松弛因子 $\gamma = 0.5$：
$$\alpha_{ref}^{(k+1)} = (1-\gamma)\alpha_{ref}^{(k)} + \gamma\alpha_{new}$$

## 3. 环境准备 (Installation)

本项目依赖经典的科学计算栈 python 3.8+：

```bash
# 克隆仓库
git clone https://github.com/art3m1s-tju/Curvature-Distance-Trajectory-SQP.git
cd Curvature-Distance-Trajectory-SQP

# 安装依赖
pip install numpy pandas scipy matplotlib osqp
```

## 4. 运行指南 (Getting Started)

### 4.1 快速运行
直接执行主脚本，它会自动加载 `track.csv` 并分别计算两种权重下的轨迹：
```bash
python reproduce_paper.py
```

### 4.2 数据格式说明
如果是使用自定义赛道，请确保 `track.csv` 包含以下列：
- `left_border_x`, `left_border_y`: 内边界坐标
- `right_border_x`, `right_border_y`: 外边界坐标

## 5. 结果可视化 (Visualization)

运行后将生成高清大图 `trajectory_comparison.png`，包含四个子板块：
1. **Full Track Overview**: 全赛道宏观走线对比。
2. **Corner Details (Top/Bottom)**: 自动缩放至关键弯道，清晰查看红线（最小曲率）如何通过 Formula 24 的精确限制贴合弯心。
3. **Alpha Distribution**: 决策变量 $\alpha$ 随索引的变化曲线。红线越靠近 0，代表赛车越贴合左侧边界。

## 6. 项目结构 (Repository Structure)

- `reproduce_paper.py`: 核心算法实现（含详尽中文注释）。
- `debug_bounds.py`: 辅助调试脚本，用于计算并打印每一处 Formula 24 的原始数值。
- `track.csv`: 示例赛道地图。
- `trajectory_comparison.png`: 示例输出图像。

---

**Author**: art3m1s-tju  
**License**: MIT
