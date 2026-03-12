# Curvature-Distance-Trajectory-SQP (MATLAB & CarSim 部署指南)

该目录存放的是专为 MATLAB 和 CarSim 联合仿真设计的轨迹优化程序版本，旨在与《Global minimum time trajectory planning considering curvature and distance for track racing》论文高度对齐。

## 代码文件 (Contents)
- `trajectory_optimization.m`: 主运行脚本程序。

## 系统要求 (Prerequisites)
1. 安装 **MATLAB** (开发及测试支持 R2019b 及以上版本均可，代码纯净无小众外部依赖)。
2. 需要 MATLAB 开设并包含 **Optimization Toolbox** 库（用来调用内置的 `quadprog` 二次规划算子）。
3. 安装 **CarSim** （如 2019 / 2021 等各种基于路面的车辆仿真版均可）用以承接数据并运行联合推演。

## 代码执行逻辑与原理解读 (How it works)
根据论文介绍：
1. **数据读入**：代码会通过 `readtable` 将根目录生成的 `track.csv` 中的内外边界坐标读入矩阵 p 和 q，并计算法向量向差 `v`。
2. **生成边界约束 [lb, ub]**：根据配置好的车身宽 `wv=1.0` 以及防撞边距安全尺度 `ws=0.5` 计算获得自变量 $\alpha$ 所拥有的合法漂移域。
3. **公式映射**：
   - 提取对角距离矩阵 $H_s, f_s$。
   - 配置曲率 $k_i$ 中所包含的三次样条中心差分常矩阵 $M$ 及伴随随向角计算得来的矩阵 $T_{xx}, T_{yy}$ （论文公式 10-20）。
   - 将距离权重因子 $\epsilon$ 融入合并构成全局大型凸优化目标 $H = H_k + \epsilon * H_s$。
4. **SQP循环收敛 (Sequential Quadratic Programming)**：通过 for 迭代调用 `quadprog` ，并运用前一步的解答再次修订中心平滑逼近阵 M、T，将曲率非线性强行用二次逼近求解直至不再变化 ($\Delta \alpha < 1e^{-4}$)。

## 如何连接 CarSim 仿真 (How to use with CarSim)
在 `trajectory_optimization.m` 剧本在 MATLAB 成功运行（Converged）之后，将在当前目录吐出一个名为 **`optimal_trajectory_for_carsim.csv`** 的纯数据矩阵表。

**导入步骤**：
1. 打开 **CarSim** 主界面环境。
2. 转至 **Path / Road / Reference Line** 参数定义区。
3. 将赛道参考线的设置输入模式调为从外部 `X-Y` 平面坐标系文件导入 (Import X-Y Coordinates from External CSV/Text 文件)。
4. 指向本文件夹下导出的 `optimal_trajectory_for_carsim.csv` 文件建立仿真路径基石。
5. 去配置论文中涉及的“基于修改版自带超级跑车 Exotic Racecar”对应的汽车质量等（详见论文图表3）。
6. 配置 `Driver Model` （预瞄 0.5s，不偏离目标，带一定限制加速度 $1.0g - 1.8g$）。最终在 3D 录像动画中检阅赛车完美贴合切弯轨迹冲刺的极限竞速圈！
