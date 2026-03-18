# Global Time-Optimal Racing Trajectory Planning via SQP

## 1. 项目简介 (Overview)

本项目是对顶级赛车轨迹规划算法《Global minimum time trajectory planning considering curvature and distance for track racing》的深度 Python 复现。本项目不仅实现了论文所述的算法，还针对原论文中存在的**三处核心数学公式错误**进行了修正，并采用 **OSQP** 求解器实现了高效的序列二次规划 (SQP) 迭代。

## 2. 核心数学修正与复现细节 (Mathematical Corrections)

在复现过程中，我们通过量纲分析和严密的数学推导，发现了原论文中三处导致程序无法正常运行或逻辑错误的笔误。以下是详细的对比与修正说明：

### 2.1 目标函数梯度项修正 (Formula 21.2)

**原论文问题**：
在计算曲率因子的线性项 $f_{\kappa}$ 时，原文给出的公式（见式 21.2）将所有的矩阵乘法项都应用在了参考点 $p$ 上。

$$
f_{\kappa} = 2(B^{-1}Cp_x)^T T_{xx} (B^{-1}Cp_x) + \dots
$$

**修正逻辑**：
根据目标函数 $E = \|M(p + V\alpha)\|_T^2$ 对变量 $\alpha$ 求偏导的链式法则，线性项梯度必须保留方向矩阵 $V$。
- **错误点**：如果全用 $p$，结果将是一个**标量**而非**向量**，导致优化器无法获得每个点的梯度方向。
- **项目修正**：在代码中我们将左侧项替换为方向向量 $V^T$，确保 $f_{\kappa}$ 是一个 $N \times 1$ 的梯度向量。

### 2.2 有效车宽投影修正 (Formula 23)

**原论文问题**：
论文在计算动态车宽 $w_v$ 时，公式内部出现了**纯数值与弧度相减**的单位错误：

$$
w_v = \frac{\sqrt{l_v^2+b_v^2}}{2} \cos\left( \frac{v_i \cdot d_i}{|v_i||d_i|} - \arctan\frac{b_v}{l_v} \right)
$$

**修正逻辑**：
- **错误点**： $\frac{v_{i} \cdot d_{i}}{|v_{i}||d_{i}|}$ 得到的是两个向量夹角的**余弦值**（范围 $[-1, 1]$），而 $\arctan\frac{b_v}{l_v}$ 是一个**角度值**。物理上无法直接相减。
- **项目修正**：我们在代码中为前项添加了 `arccos` 函数，将其转换为角度后进行运算：

$$
w_v = \frac{\sqrt{l_v^2+b_v^2}}{2} \cos\left( \arccos\left(\frac{v_i \cdot d_i}{|v_i||d_i|}\right) - \arctan\frac{b_v}{l_v} \right)
$$

### 2.3 边界决策变量 $\alpha$ 约束修正 (Formula 24)

**原论文问题**：
这是原论文中最致命的**量纲（物理单位）错误**。原文给出的 $\alpha$ 边界为：

$$
\frac{(w_v + w_s)|v_i|}{n_{Ii} \cdot v_i} \le \alpha_i \le 1 - \dots
$$

**修正逻辑**：
- **定义检查**：根据论文定义， $r_i = p_i + \alpha_i v_i$ ，其中 $\alpha_i$ 是一个 $[0, 1]$ 之间的**无量纲比例系数**。
- **错误点**：原公式分子 $(w_v + w_s)|v_i|$ 的单位是 $meters^2$，分母 $n \cdot v$ 单位是 $meters$，结果单位是 $meters$。要求一个百分比 $\alpha$ 大于一个长度（如 2.5 米）在数学上是不成立的。
- **项目修正**：移除多余的 $|v_i|$，使约束回归无量纲形式：

$$
\alpha_{min} = \frac{w_v + w_s}{n_{Ii} \cdot v_i}
$$

---

## 3. 环境准备 (Installation)

本项目运行在 Python 环境下，推荐使用 Python 3.8+。

```bash
# 1. 克隆项目
git clone https://github.com/art3m1s-tju/Curvature-Distance-Trajectory-SQP.git
cd Curvature-Distance-Trajectory-SQP

# 2. 安装科学计算依赖
pip install numpy pandas scipy matplotlib osqp
```

## 4. 运行指南 (Usage)

### 4.1 脚本执行
直接运行主脚本，程序将加载 `track.csv` 数据并执行 50 轮 SQP 迭代。
```bash
python reproduce_paper.py
```

### 4.2 输出结果说明
- **控制台输出**：将实时打印每一轮迭代的收敛状态（$\alpha$ 的最大变化量）。
- **图片结果**：运行结束后将在当前目录下生成 `trajectory_comparison.png`。该图包含：
  - **Full Track Overview**：展示整体赛道与两条轨迹。
  - **Corner Detail**：放大弯道细节，观察 Formula 24 修正后的精准切弯。
  - **Alpha Distribution**：决策变量的分布直方图。

### 4.3 核心参数调节
在 `reproduce_paper.py` 的 `__main__` 部分可以调节以下关键参数：
- `epsilon`: 平衡权重。设为 $0$ 时为纯最小曲率线；增大（如 $1000$）则倾向于最短路径。
- `ws`: 安全冗余距离（米）。增加此值会让轨迹离墙更远。
- `gamma`: SQP 松弛因子。默认 $0.5$，若发现不收敛可以适当调小（如 $0.2$）。

## 5. 项目结构 (Structure)

- `reproduce_paper.py`: 核心复现逻辑。
- `track.csv`: 赛道数据（包含左、右边界点坐标）。
- `trajectory_comparison.png`: 示例运行结果截图。

---

**Author**: art3m1s-tju  
**License**: MIT
