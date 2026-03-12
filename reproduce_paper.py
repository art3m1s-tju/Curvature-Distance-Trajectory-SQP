import numpy as np
import pandas as pd
import scipy.sparse as sp
import osqp
import matplotlib.pyplot as plt

def load_track_data(filepath='track.csv'):
    """
    加载赛道边界数据:
    读取给定的边界坐标，并将赛道宽度较窄处或者需要修正的地方进行预处理。
    返回内边界坐标p, 外边界坐标q以及差值向量v。
    """
    # 调用 pandas 库的 read_csv 函数读取指定路径的 CSV 文件数据，存入 DataFrame 对象 df 中
    df = pd.read_csv(filepath)
    # 提取 df 中列名为 'left_border_x' 和 'left_border_y' 的数据，并转换为 numpy 数组，赋值给内边界坐标向量 p (对应论文公式中的内边界 p_i)
    p = df[['left_border_x', 'left_border_y']].values
    # 提取 df 中列名为 'right_border_x' 和 'right_border_y' 的数据，并转换为 numpy 数组，赋值给外边界坐标向量 q (对应论文公式中的外边界 q_i)
    q = df[['right_border_x', 'right_border_y']].values
    # 计算从内边界指向外边界的差值向量 v (如论文公式 2.2 所示，v_i = q_i - p_i)
    v = q - p
    # 返回提取出的内边界 p、外边界 q 以及方向向量 v 给调用函数处
    return p, q, v

def build_difference_matrix(N):
    """
    构建一阶差分矩阵 A (闭环)。
    用于计算相邻点之间的坐标差值。
    """
    # 创建一个长度为 N，元素全为 -1 的一维 numpy 数组，作为矩阵的主对角线 (表示 -r_i)
    diag = np.ones(N) * (-1)
    # 创建一个长度为 N-1，元素全为 1 的一维 numpy 数组，作为矩阵的右上副对角线 (表示 r_{i+1})
    off_diag = np.ones(N-1)
    # 利用 scipy.sparse.diags 函数生成稀疏对角矩阵 A，主对角线偏移量为 0，副对角线偏移量为 1
    # 结果通过 .tolil() 转化为 List of Lists (LIL) 格式，方便修改单个元素
    A = sp.diags([diag, off_diag], [0, 1], shape=(N, N)).tolil()
    # 为了实现赛道闭环，修改矩阵最后一行的第一个元素为 1，代表在最后一点时减去自身加上第一个点，即 r_1 - r_N
    A[N-1, 0] = 1 
    # 将生成的 LIL 格式稀疏矩阵转化为 Compressed Sparse Column (CSC) 格式，以便之后提高矩阵乘法效率并返回
    return A.tocsc()

def calculate_distance_factor(A, p, v):
    """
    计算距离因子的 Hessian 矩阵 Hs 和一次项 fs:
    这里是论文中的公式 (7.1) 和 (7.2) 核心构建逻辑。
    """
    # 获取赛道参考点的总数量 N
    N = len(p)
    # 从内边界坐标矩阵 p 中单独提取出所有的 x 坐标
    px = p[:, 0]
    # 从内边界坐标矩阵 p 中单独提取出所有的 y 坐标
    py = p[:, 1]
    
    # 提取赛道宽度向量 v 的 x 分量，并构建一个 N x N 的对角稀疏矩阵 Vx
    Vx = sp.diags(v[:, 0])
    # 提取赛道宽度向量 v 的 y 分量，并构建一个 N x N 的对角稀疏矩阵 Vy
    Vy = sp.diags(v[:, 1])
    
    # 预计算差分矩阵 A 的转置与其自身的乘积 (A^T * A)，这在计算二阶和一阶项时都需要用到
    ATA = A.T @ A
    
    # 计算 x 方向距离目标函数的二次项系数矩阵 Hs_x = Vx^T * A^T * A * Vx
    Hs_x = Vx.T @ ATA @ Vx
    # 计算 x 方向距离目标函数的一次项系数向量 fs_x = Vx^T * A^T * A * px
    fs_x = Vx.T @ ATA @ px
    
    # 计算 y 方向距离目标函数的二次项系数矩阵 Hs_y = Vy^T * A^T * A * Vy
    Hs_y = Vy.T @ ATA @ Vy
    # 计算 y 方向距离目标函数的一次项系数向量 fs_y = Vy^T * A^T * A * py
    fs_y = Vy.T @ ATA @ py
    
    # 将 x 方向和 y 方向的二次项相加，并乘以 2 以匹配标准二次规划形式的系数 (1/2 x^T H x)
    Hs = 2 * (Hs_x + Hs_y)
    # 将 x 方向和 y 方向的一次项相加，并乘以 2
    fs = 2 * (fs_x + fs_y)
    
    # 返回构建好的距离因子 Hessian 矩阵 Hs 和线性一次项向量 fs
    return Hs, fs

def calculate_derivative_matrices(r):
    """
    基于参考线 r，计算曲率公式（9）和（10）中的权重矩阵，
    并构造三次样条（等距简化版）的二阶导数映射矩阵 M。
    """
    # 获取赛道参考点的总数量 N
    N = len(r)
    # 提取参考轨迹坐标的 x 分量
    rx = r[:, 0]
    # 提取参考轨迹坐标的 y 分量
    ry = r[:, 1]
    
    # 计算相邻点在 x 方向的差值，并使用 numpy.roll 实现前向闭环位移 (模拟 r_{i+1} - r_i)
    rx_prime = np.roll(rx, -1) - rx
    # 计算相邻点在 y 方向的差值，并使用 numpy.roll 实现前向闭环位移 (模拟 r_{i+1} - r_i)
    ry_prime = np.roll(ry, -1) - ry
    # 根据勾股定理计算相邻点之间的直线距离 ds (可以近似为弧长 differential s)
    ds = np.sqrt(rx_prime**2 + ry_prime**2)
    
    # 避免除以 0 的情况，将 x 方向一阶导数归一化为关于弧长的导数 (dx/ds)
    rx_prime = rx_prime / (ds + 1e-6)
    # 避免除以 0 的情况，将 y 方向一阶导数归一化为关于弧长的导数 (dy/ds)
    ry_prime = ry_prime / (ds + 1e-6)
    
    # 计算论文公式 (10) 中 T矩阵共同的分母部分项 ((x')^2 + (y')^2)^{3/2}
    denominator = (rx_prime**2 + ry_prime**2)**1.5 + 1e-8
    
    # 计算公式(10)中权重矩阵 Txx 的对角线元素
    Txx_diag = (ry_prime**2) / denominator
    # 计算公式(10)中权重矩阵 Tyy 的对角线元素
    Tyy_diag = (rx_prime**2) / denominator
    # 计算公式(10)中权重矩阵 Txy 的对角线元素
    Txy_diag = -(rx_prime * ry_prime) / denominator
    
    # 将 Txx_diag 主对角线元素转换生成为 N x N 维度稀疏对角矩阵 Txx
    Txx = sp.diags(Txx_diag)
    # 将 Tyy_diag 主对角线元素转换生成为 N x N 维度稀疏对角矩阵 Tyy
    Tyy = sp.diags(Tyy_diag)
    # 将 Txy_diag 主对角线元素转换生成为 N x N 维度稀疏对角矩阵 Txy
    Txy = sp.diags(Txy_diag)
    
    # 构建拉普拉斯算子形(中心差分)的二阶导数矩阵 M 用于代替论文中复杂的三次样条 C 和 B 矩阵。
    # 设置常系数 1, -2, 1 在主对角线和上下副对角线处。
    M = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).tolil()
    # 为了实现赛道首尾完美闭合，修改首行末列对应连接节点为 1
    M[0, N-1] = 1
    # 为了实现赛道首尾完美闭合，修改末行首列对应连接节点为 1
    M[N-1, 0] = 1
    
    # 将前面构建的单位弧长 M 矩阵除以实际物理弧长平方 (ds^2)，转换为真正的关于 s 二阶导数运算矩阵，这部分对应公式 (13) 和 (16.2), (17) 三次样条约束的近似结果
    ds_diag = sp.diags(1.0 / (ds**2 + 1e-6))
    # 完成对角弧长平方矩阵与差分 M 矩阵的结合计算，并将结果转换成 CSC 格式返回
    M = ds_diag @ M.tocsc()
    
    # 将四个生成的曲率核心矩阵抛回给主程序使用
    return Txx, Tyy, Txy, M

def calculate_curvature_factor(M, Txx, Tyy, Txy, p, v):
    """
    计算曲率因子的 Hessian 矩阵 Hk 和一次项 fk 
    （论文公式 20）。
    """
    # 从内边界点集 p 中分离出所有 x 坐标，用于后续的一次项计算
    px = p[:, 0]
    # 从内边界点集 p 中分离出所有 y 坐标，用于后续的一次项计算
    py = p[:, 1]
    
    # 将指向外边界的方向向量 v 的 x 方向分量抽取并展开为稀疏对角矩阵
    Vx = sp.diags(v[:, 0])
    # 将指向外边界的方向向量 v 的 y 方向分量抽取并展开为稀疏对角矩阵
    Vy = sp.diags(v[:, 1])
    
    # 预计算用于组合曲率代价函数 M^T * Txx * M 这个由中心差分和一阶导权重组成的矩阵单元
    MT_Txx_M = M.T @ Txx @ M
    # 预计算用于组合曲率代价函数 M^T * Tyy * M 的核心对称矩阵
    MT_Tyy_M = M.T @ Tyy @ M
    # 预计算交叉项的曲率代价关联矩阵 M^T * Txy * M
    MT_Txy_M = M.T @ Txy @ M
    
    # Hessian 计算 (对应公式 20 中矩阵 H_k 的各组成部分)
    # 计算曲率矩阵第一主成分项(x与x作用项): Vx^T * (M^T Txx M) * Vx
    term1 = Vx.T @ MT_Txx_M @ Vx
    # 计算曲率矩阵第二主成分项(y与y作用项): Vy^T * (M^T Tyy M) * Vy
    term2 = Vy.T @ MT_Tyy_M @ Vy
    # 计算曲率矩阵交叉成分项1(x与y作用项): Vx^T * (M^T Txy M) * Vy
    term3 = Vx.T @ MT_Txy_M @ Vy
    # 计算曲率矩阵交叉成分项2(y与x作用项): Vy^T * (M^T Txy M) * Vx，由对称性此与上一步实质相似
    term4 = Vy.T @ MT_Txy_M @ Vx
    # 合并所有的曲率矩阵平方项，并按二次规划二次型要求乘以常数倍因子 2 形成最终的曲率 Hessian 矩阵
    Hk = 2 * (term1 + term2 + term3 + term4)
    
    # First-order term 计算 (对应公式 20 中向量 f_k 的各维组成)
    # 计算曲率公式关于 x 对一次边界位置的导数对应的一次项 fk1
    fk1 = Vx.T @ MT_Txx_M @ px
    # 计算曲率公式关于 y 交叉坐标作用产生的一次项 fk2
    fk2 = Vy.T @ MT_Tyy_M @ py
    # 计算由 Txy 相乘引入的对于 x 操作受 y 边界位置影响产生的一次项 fk3
    fk3 = Vx.T @ MT_Txy_M @ py
    # 计算由 Txy 相乘引入的对于 y 操作受 x 边界位置影响产生的一次项 fk4
    fk4 = Vy.T @ MT_Txy_M @ px
    # 合并上述分别计算的一次项向量元素并乘2构成完整的一次目标向量 fk
    fk = 2 * (fk1 + fk2 + fk3 + fk4)
    
    # 将构作好的二阶信息矩阵与一阶信息向量向外部函数返回
    return Hk, fk

def optimize_trajectory(p, v, wv, ws, epsilon=0, max_iter=6):
    """
    使用 OSQP 进行 SQP 序列二次规划迭代，求解最优轨迹 $\alpha$。
    """
    # 提取共有多少个离散路径控制点以决定整体规划问题的矩阵维度 N
    N = len(p)
    # 调用内置辅助函数生成一个 N * N 的闭环有限前向差分网络矩阵 A_diff
    A_diff = build_difference_matrix(N)
    
    # 直接在循环外求解距离目标，因为其系数矩阵是常数且不论迭代如何改变，它仅受几何形状控制
    Hs, fs = calculate_distance_factor(A_diff, p, v)
    
    # 初始化最优推断轨迹。我们这里假设赛车最初打算沿着整条赛道的绝对中心（即 0.5 宽）行驶。
    alpha_ref = np.ones(N) * 0.5
    
    # 对每一点计算路面总有效地理垂直宽度（向量 v 的二范数长度）
    road_widths = np.linalg.norm(v, axis=1)
    
    # 使用计算物理车身宽度占用(wv)与希望的安全冗余间隔(ws)，推算出车体不该触犯边界的最小合法 \alpha 参数阈值
    alpha_min = (wv + ws) / road_widths
    # 推算出另一侧最接近外部边界的合法最大 \alpha 阈值约束
    alpha_max = 1.0 - (wv + ws) / road_widths
    
    # 这里加两行保护代码确保就算存在异常极窄路段规划域也不为负，\alpha_min 被限制在不离内边太离谱 (0.05到0.45之间)
    alpha_min = np.clip(alpha_min, 0.05, 0.45)
    # \alpha_max 被强制拉回到不比0.55小且不超0.95，给数值优化留足可行域
    alpha_max = np.clip(alpha_max, 0.55, 0.95)
    
    # 由于求解器OSQP需要不等式边界阵对应所有的决策变量，我们的全变量约束就是标准单位对角稀疏矩阵（即每个 \alpha 限制自己）
    A_constr = sp.eye(N).tocsc()
    
    # 生成极其微小的扰动对角项加在主对角线上用于避免计算时由于纯弯道情况引入完全非满秩目标海森矩阵导致的奇异报错
    reg = 1e-6 * sp.eye(N).tocsc()
    
    # 外层 SQP (Sequential Quadratic Programming) 的主序列优化循环，最大跑 max_iter 次
    for iteration in range(max_iter):
        # 基于从上一轮求得出的（或者第一轮设定的中心点 0.5）\alpha 权重还原出当前真实的几何坐标参照线
        r_current = p + v * alpha_ref[:, np.newaxis]
        
        # 将刚刚还原的最参考坐标系送进去算出其相较于世界系各导数权重并以此产生下一次曲率运算所必须利用常数列
        Txx, Tyy, Txy, M = calculate_derivative_matrices(r_current)
        # 用前一步的临时系数 T 系列等结果结合赛道地形建立当次步进真正所需的关于 \alpha 决策变量的 Hk 矩阵
        Hk, fk = calculate_curvature_factor(M, Txx, Tyy, Txy, p, v)
        
        # 按照论文思想利用权重 \epsilon 把刚得出的带非线性假设的曲率 Hk阵 和 一直不变的距离常数 Hs 阵直接线形求和合并为一个唯一的二次惩罚结构 P 并注入微量的抗奇异稳定因子
        P = sp.csc_matrix(Hk + epsilon * Hs + reg)
        # 一次项直接作对应目标合并，构造全局线性衰减因子 q
        q = fk + epsilon * fs
        
        # 生成一个新的 OSQP 求解器封装对象用来解决本次构造的带约束QP凸优化命题
        prob = osqp.OSQP()
        # 把要最小化的方程常数P, q，我们自己设定的约束上下确界代入解析内核进行工作初始化。我们将最大迭代设定为较高以期稳定解决长赛道微小扰动容差收敛(eps=1e-5默认)
        prob.setup(P=P, q=q, A=A_constr, l=alpha_min, u=alpha_max, 
                   verbose=False, max_iter=20000)
        
        # 开始运行内部大规模方程推手并记录其操作日志结果给 res 承装变量
        res = prob.solve()
        
        # 如果优化内核由于步长或溢出抛出未能完成 solved 阶段求解就打印告警提示
        if res.info.status != 'solved':
            print(f"迭代 {iteration+1}: 求解器状态 '{res.info.status}'")
            # 出于健壮性，当无有效解阵(None)被吐出那么就立马拉停当前循环退出，反之勉强吃下近似解进行下一次抢救
            if res.x is None:
                break
        
        # 提取在约束下完成更新的新阶段每个离散点分配权重参数，存在新对象 \alpha_new 中
        alpha_new = res.x
        # 取本轮寻优结果的各个点与最初输入作为参照始点的权重点求最大绝对误差距离作为此收敛精度量纲指标 diff
        diff = np.max(np.abs(alpha_new - alpha_ref))
        # 于命令行记录本次迭代后发生的赛车轨迹线侧移收敛情况
        print(f"迭代 {iteration+1}/{max_iter} 完毕，最大 alpha 变化量: {diff:.6f}")
        
        # 通过验证本轮推演，我们强制将作为求解目标的原始估猜替换为您刚才找到更优秀的这一组合让其步入下个分析轮回
        alpha_ref = alpha_new
        
        # 倘若检测最大决策变量值相对前一次修改没能跳出极小误差区间容忍，判定曲线算法大体稳定抵达最速，主动结束。
        if diff < 1e-3:
            print("算法已收敛。")
            break
            
    # 全局 SQP 计算或者被强行切断后，将我们锁死在手里的 \alpha 参数乘以赛道地形得到整条最后真实的规划物理跑法散点序列 r_optimal
    r_optimal = p + v * alpha_ref[:, np.newaxis]
    # 返回最佳求解出的物理平滑路线和背后藏着的规划内向系数供制图使用
    return r_optimal, alpha_ref

if __name__ == '__main__':
    # 1. 车辆与安全边界参数
    wv = 1.0  
    ws = 0.5  
    
    print("加载并解析轨迹数据...")
    p, q_bound, v = load_track_data('track.csv')
    
    # 2. 计算纯最小曲率轨迹 (epsilon = 0)
    print("\n--- 计算最小曲率轨迹 (epsilon = 0) ---")
    r_opt_k, alpha_k = optimize_trajectory(p, v, wv, ws, epsilon=0.0, max_iter=5)
    
    # 3. 计算平衡轨迹 (epsilon = 10)
    print("\n--- 计算平衡轨迹 (epsilon = 10.0) ---")
    r_opt_s, alpha_s = optimize_trajectory(p, v, wv, ws, epsilon=10.0, max_iter=5)

    # 4. 绘图
    plt.figure(figsize=(10, 8))
    idx_start = 0
    idx_end = 800
    
    plt.plot(p[:, 0][idx_start:idx_end], p[:, 1][idx_start:idx_end], 'k-', label='Inner Border')
    plt.plot(q_bound[:, 0][idx_start:idx_end], q_bound[:, 1][idx_start:idx_end], 'k-', label='Outer Border')
    
    plt.plot(r_opt_k[:, 0][idx_start:idx_end], r_opt_k[:, 1][idx_start:idx_end], 'r--', linewidth=2, label='Min Curvature ($\epsilon$=0)')
    plt.plot(r_opt_s[:, 0][idx_start:idx_end], r_opt_s[:, 1][idx_start:idx_end], 'b-.', linewidth=2, label='Balanced ($\epsilon$=10)')
    
    plt.title("Trajectory Optimization Comparison")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.savefig('trajectory_comparison.png')
    print("\n对比图已保存为 'trajectory_comparison.png'")
