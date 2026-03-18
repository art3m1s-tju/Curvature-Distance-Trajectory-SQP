import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import osqp
import matplotlib.pyplot as plt
import scipy.linalg as la

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
    
    # 计算相邻点在 x 方向的一阶导数，求的是平均每一个格子 x 值的变化量 (中心差分, (r_{i+1} - r_{i-1}) / 2)
    rx_prime = (np.roll(rx, -1) - np.roll(rx, 1)) / 2.0
    # 计算相邻点在 y 方向的一阶导数，求的是平均每一个格子 y 值的变化量 (中心差分)
    ry_prime = (np.roll(ry, -1) - np.roll(ry, 1)) / 2.0
    # 根据勾股定理计算相邻点之间的直线距离 ds，代表一个单位步长下对应的单位弧长 (可以近似为弧长 differential s)
    ds = np.sqrt(rx_prime**2 + ry_prime**2)
    
    # 避免除以 0 的情况，将 x 方向一阶导数归一化为关于弧长的导数 (dx/ds)
    # 除以 ds 求的是沿着轨迹每往前走一米 x 和 y 坐标的变化量
    # + 1e-8 的原因是避免分母为 0
    rx_prime = rx_prime / (ds + 1e-8)
    # 避免除以 0 的情况，将 y 方向一阶导数归一化为关于弧长的导数 (dy/ds)
    ry_prime = ry_prime / (ds + 1e-8)
    
    # 按照图片中的公式 (12.1), (12.2), (12.3):
    # 分母应该是 (x_i'^2 + y_i'^2)^3
    # 在代码中，因为 rx_prime 和 ry_prime 本来就是差分 (且在这个部分重新计算且没有事先除以 ds)，
    # 我们可以重新计算未归一化的导数，或者直接按照之前算好的未归一化导数计算。
    
    # 重新获取未归一化的导数
    rx_prime_unnorm = (np.roll(rx, -1) - np.roll(rx, 1)) / 2.0
    ry_prime_unnorm = (np.roll(ry, -1) - np.roll(ry, 1)) / 2.0
    
    # 分母: (x'^2 + y'^2)^3
    denominator = (rx_prime_unnorm**2 + ry_prime_unnorm**2)**3 + 1e-8
    
    # 根据公式计算对角线元素
    Txx_diag = (ry_prime_unnorm**2) / denominator
    Tyy_diag = (rx_prime_unnorm**2) / denominator
    Txy_diag = (-2 * rx_prime_unnorm * ry_prime_unnorm) / denominator
    
    # 将 Txx_diag 主对角线元素转换生成为 N x N 维度稀疏对角矩阵 Txx
    Txx = sp.diags(Txx_diag)
    # 将 Tyy_diag 主对角线元素转换生成为 N x N 维度稀疏对角矩阵 Tyy
    Tyy = sp.diags(Tyy_diag)
    # 将 Txy_diag 主对角线元素转换生成为 N x N 维度稀疏对角矩阵 Txy
    Txy = sp.diags(Txy_diag)
    
    # 将生成的三个曲率核心矩阵抛回给主程序使用
    return Txx, Tyy, Txy

def calculate_M_matrix(N):
    """
    构造三次样条（等距简化版）的二阶导数映射矩阵 M (M = B^{-1}C)
    由于 B 和 C 对于固定 N 的问题是常数矩阵，只需在循环外计算一次即可，极大提升性能。
    """
    # 构造 B 矩阵的对角线元素
    B = sp.diags([1, 4, 1], [-1, 0, 1], shape=(N, N), dtype=float).tolil()
    
    # 构造 C 矩阵的对角线元素
    C = sp.diags([6, -12, 6], [-1, 0, 1], shape=(N, N), dtype=float).tolil()
    
    # 处理赛道闭环边界条件
    B[0, N-1] = 1
    B[N-1, 0] = 1
    
    C[0, N-1] = 6
    C[N-1, 0] = 6
    
    # 采用密集矩阵求逆以提升大矩阵乘法效率并转回稀疏传入
    C_dense = C.toarray()
    B_dense = B.toarray()

    B_inv = la.inv(B_dense)
    M_dense = B_inv @ C_dense
    
    # 截断极小值以恢复矩阵的带状稀疏性，如果不截断，随着 B^-1 变致密，Hk 也会成密集矩阵，OSQP 的稀疏 LDL 分解将耗时极大崩溃
    M_dense[np.abs(M_dense) < 1e-4] = 0.0
    M = sp.csc_matrix(M_dense)
    
    return M

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
    
    # 严格按照论文公式 (21.1) 重构 Hk
    # 第一项 (x分量): (B^-1 C v_x)^T T_xx (B^-1 C v_x)
    term1 = Vx.T @ MT_Txx_M @ Vx
    # 第二项 (交叉项): 这里必需补全对称结构，论文公式（15）其实是二次型，所以 A 需要 A + A^T
    term_cross = Vx.T @ MT_Txy_M @ Vy
    term_cross_sym = term_cross + term_cross.T
    # 第三项 (y分量): (B^-1 C v_y)^T T_yy (B^-1 C v_y)
    term2 = Vy.T @ MT_Tyy_M @ Vy
    
    # 合并构成公式 21.1 的 Hk (补全了漏掉的一半交叉偏导矩阵)
    Hk = 2 * (term1 + term2) + term_cross_sym
    
    # 论文公式 (21.2) 除把 p 写错以外，它还漏了另外一半交叉项的梯度传递！
    # 对 α 求偏导一次项:
    # J_x 部分: 2 * (MV_x)^T T_xx (Mp_x)
    # J_y 部分: 2 * (MV_y)^T T_yy (Mp_y)
    # J_xy 单叉项: (MV_x)^T T_xy (Mp_y) + (MV_y)^T T_xy (Mp_x)
    
    fk1 = Vx.T @ MT_Txx_M @ px
    fk_cross1 = Vy.T @ MT_Txy_M @ px
    fk_cross2 = Vx.T @ MT_Txy_M @ py
    fk2 = Vy.T @ MT_Tyy_M @ py
    
    # 合并构成正确的 fk（N×1 向量，包含了双向交叉项投影）
    fk = 2 * (fk1 + fk2) + fk_cross1 + fk_cross2
    
    return Hk, fk

def calculate_boundary_normals(bound_pts, v):
    """
    计算边界点集的单位法向量 (对应论文图5中的 n_{Ii} 和 n_{Oi})
    
    根据论文图5，无论是内边界还是外边界的法向量，都应当"指向赛道内部"。
    这意味着它与从内边界指向外边界的横向向量 v_i 总是呈现大致同向或锐角的夹角。
    数学上，只需要保证法向量与 v_i 的点积为正 (n·v > 0) 即可实现"指向赛道内部"。
    
    参数:
    - bound_pts: 边界点坐标集合 N x 2
    - v: 从内边界点横向指向外边界点的向量 N x 2
    
    返回:
    - normals: 符合物理意义的赛道向内单位法向量 N x 2
    """
    # 1. 使用中心差分计算边界曲线的单位切向量 tangent
    # 中心差分求导
    dp = (np.roll(bound_pts, -1, axis=0) - np.roll(bound_pts, 1, axis=0)) / 2.0
    # 计算向量长度，keepdims会保持二维矩阵的形状(N,1)，方便后续广播运算
    dp_norm = np.linalg.norm(dp, axis=1, keepdims=True)
    # 计算单位切向量
    tangent = dp / (dp_norm + 1e-8)
    
    # 2. 从切向量派生出两个互相方向相反的候选法向量
    # 候选法向量1：逆时针旋转 90 度
    n1 = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    # 候选法向量2：顺时针旋转 90 度
    n2 = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    
    # 3. 选择与跨越向量 v 点积为正的那一个方向，确保它总是指向赛道内侧
    dot1 = np.sum(n1 * v, axis=1)
    
    normals = np.where(dot1[:, np.newaxis] > 0, n1, n2)
    return normals

def calculate_wv_per_point(r, v, l_v, b_v):
    """
    根据论文公式 (23) 计算每个离散点所需的车辆实体有效半宽 w_v。
    
    w_v 表示的是：将车辆（长方形 l_v * b_v）置于当前轨迹点 r_i 上并指向航向角时，
    车辆占据在横向向量 v_i (跨越赛道方向)上的最大宽度投影。
    简而言之：此时此刻要想不让车角蹭到边界，当前车心必须至少离边界多远。
    
    参数:
    - r: 当前参考轨迹点的坐标 [N, 2]
    - v: 内部边界指向外部边界的横向向量 [N, 2]
    - l_v: 车辆物理总长度 (如 4.7m)
    - b_v: 车辆物理总宽度 (如 2.0m)
    """
    # d_i 表示当前轨迹上车辆的实际运动航向向量 (d_i = r_{i+1} - r_i)
    d = np.roll(r, -1, axis=0) - r
    d_norm = np.linalg.norm(d, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    
    # 避免因两点重合而除以 0 导致报错
    d_norm_safe = np.where(d_norm < 1e-8, 1e-8, d_norm)
    v_norm_safe = np.where(v_norm < 1e-8, 1e-8, v_norm)
    
    # 求解赛道横截向量 v_i 与当前行驶航向 d_i 之间的夹角 (v_i · d_i)
    dot_vd = np.sum(v * d, axis=1)
    # 根据点积公式 cos(theta) = (v·d) / (|v|*|d|)
    cos_val = dot_vd / (v_norm_safe * d_norm_safe)
    # 截断由于浮点数精度误差造成的边界溢出（如 1.00000001），防止 arccos 出现 nan
    cos_val = np.clip(cos_val, -1.0, 1.0)
    
    # 求解真实的航向夹角差
    # 论文原文在公式(23)中直接写了 cos( (v_i·d_i)/(|v_i||d_i|) - arctan(...) )，
    # 这是一个数学笔误，余弦值不能与角度相减。我们增加 arccos 将其还原为真正的角度。
    angle_vd = np.arccos(cos_val)
    # 车辆的长宽构成的自身外角大小 (即车辆对角线与中心轴的夹角)
    angle_vehicle = np.arctan2(b_v, l_v)
    
    # 论文公式 (23)： w_v = (sqrt(l_v^2 + b_v^2) / 2) * |cos(θ_vd - θ_veh)|
    # 计算车辆中心到其对角（最外扩点）的直线对角长度
    diagonal = np.sqrt(l_v**2 + b_v**2) / 2.0
    # 将此长度投影至垂直于赛道边界横向的方向上
    wv = diagonal * np.abs(np.cos(angle_vd - angle_vehicle))
    
    return wv

def optimize_trajectory(p, v, l_v, b_v, ws, epsilon=0, max_iter=6, gamma_normal=0.5, gamma_inaccurate=0.1):
    """
    使用 OSQP 进行 SQP 序列二次规划迭代，求解最优轨迹 $\alpha$。
    """
    # 提取共有多少个离散路径控制点以决定整体规划问题的矩阵维度 N
    N = len(p)
    import scipy.sparse.linalg as spla
    
    # 提前计算纯几何导数算子 M (只计算一次)
    M = calculate_M_matrix(N)
    
    # 调用内置辅助函数生成一个 N * N 的闭环有限前向差分网络矩阵 A_diff
    A_diff = build_difference_matrix(N)
    
    # 直接在循环外求解距离目标，因为其系数矩阵是常数且不论迭代如何改变，没受几何形状控制
    Hs, fs = calculate_distance_factor(A_diff, p, v)
    
    # 根据用户反馈：初始参考线切回“内边界首轮”，即 alpha=0
    alpha_ref = np.zeros(N)
    
    # === 边界预处理 ===
    # v_norm: |v_i|
    # n_I: n_{Ii} 内边界向内法向量
    # n_O: n_{Oi} 外边界向内法向量 (q = p + v 为外边界坐标)
    v_norm = np.linalg.norm(v, axis=1)
    q_bound = p + v
    n_I = calculate_boundary_normals(p, v)
    n_O = calculate_boundary_normals(q_bound, v)
    
    # 论文公式 (24) 分母中要求的点积投影因子 (n_{Ii} · v_i) 和 (n_{Oi} · v_i)
    v_dot_nI = np.sum(n_I * v, axis=1)
    v_dot_nO = np.sum(n_O * v, axis=1)
    
    # 防止分母处于奇异导致的除零报错
    v_dot_nI = np.maximum(v_dot_nI, 1e-4)
    v_dot_nO = np.maximum(v_dot_nO, 1e-4)
    
    # 由于求解器OSQP需要不等式边界阵对应所有的决策变量，我们的全变量约束就是标准单位对角稀疏矩阵（即每个 \alpha 限制自己）
    A_constr = sp.eye(N).tocsc()
    
    # 生成极其微小的扰动对角项加在主对角线上用于避免计算时由于纯弯道情况引入完全非满秩目标海森矩阵导致的奇异报错
    reg = 1e-6 * sp.eye(N).tocsc()
    
    # 外层 SQP (Sequential Quadratic Programming) 的主序列优化循环，最大跑 max_iter 次
    for iteration in range(max_iter):
        # 基于从上一轮求得出的（或者第一轮设定的中心点 0.5）\alpha 权重还原出当前真实的几何坐标参照线
        r_current = p + v * alpha_ref[:, np.newaxis]
        
        # 基于本轮得到的当前迭代轨道中心线，计算车辆走这条线时在各点霸占的车身实体投影宽度 w_v
        # 这是动态的，因为一旦车在这拐弯的角度发生了改变，车身在横向截面上的所需面积也会随之改变
        wv_array = calculate_wv_per_point(r_current, v, l_v, b_v)
        
        # 严格按论文公式 (24) 计算每个决策点可以自由规划的上下界域
        # 注意：这里我们修正了原论文公式 (24) 中的一个严重数学笔误！
        # 原论文写的是 (w_v + w_{s,i}) * |v_i| / (n_I · v_i)。
        # 但 n_I · v_i 的本身量纲就是长度，分子如果再乘 |v_i| 会导致整体量纲变为长度，
        # 而 alpha_i 是一个无量纲的比例因子 (0~1)。乘上 |v_i| 会直接导致算出的上下界大于 1 甚至大于 2。
        # 上下界全部失效后，受后面的 clip 保护，车辆被死死锁在 0.45~0.55 的赛道绝对中心。
        # 正确的区别是将多余的 v_norm 乘子去掉：
        alpha_min = (wv_array + ws) / v_dot_nI
        
        # 右侧不等式（上界）：要求向左靠时同样不能蹭墙
        alpha_max = 1.0 - (wv_array + ws) / v_dot_nO
        
        # 不再硬裁剪 0.05-0.95，而是直接确保 min 不超过 max，防止交叉死锁
        alpha_max = np.maximum(alpha_max, alpha_min + 1e-3)
        
        # 将刚刚还原的最参考坐标系送进去算出其相较于世界系各导数权重并以此产生下一次曲率运算所必须利用常数列
        Txx, Tyy, Txy = calculate_derivative_matrices(r_current)
        # 用前一步的临时系数 T 系列等结果结合赛道地形建立当次步进真正所需的关于 \alpha 决策变量的 Hk 矩阵
        Hk, fk = calculate_curvature_factor(M, Txx, Tyy, Txy, p, v)
        
        # 强制 Hk 对称化并加上轻量级正则化以保证求解器所要求的半正定性(PSD)
        # 如果依然非正定（虽然物理模型在附近是凸的），OSQP自己也有小幅度的 internal offset.
        # 这点大幅提升了复现速度
        Hk = (Hk + Hk.T) / 2.0
        Hk = Hk + sp.diags(np.ones(N) * 1e-4)

        # 同样轻量化检查 Hs
        if iteration == 0:
            Hs = (Hs + Hs.T) / 2.0
            Hs = Hs + sp.diags(np.ones(N) * 1e-4)

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
        
        # 提取在约束下完成更新的新阶段每个离散点分配权重参数
        alpha_new = res.x
        
        # 如果优化内核未能完成 solved 阶段求解就打印告警提示
        if res.info.status == 'solved':
            gamma_adaptive = gamma_normal # 正常解的保守步长
        elif res.info.status == 'solved inaccurate':
            print(f"迭代 {iteration+1:2d}: 求解器状态 '{res.info.status}'，启用自适应收缩步长保护。")
            gamma_adaptive = gamma_inaccurate # 极大降低本次步长权重抑制误差
        else:
            print(f"迭代 {iteration+1:2d}: 求解器状态异常 '{res.info.status}'，直接终止迭代以免污染轨迹。")
            break
            
        if alpha_new is None:
            break
            
        diff = np.max(np.abs(alpha_new - alpha_ref))
        
        # 计算全局自动监控指标 (基于 alpha_new 计算代价)
        J_k_val = 0.5 * alpha_new.T @ Hk @ alpha_new + fk.T @ alpha_new
        J_s_val = 0.5 * alpha_new.T @ Hs @ alpha_new + fs.T @ alpha_new
        J_total = J_k_val + epsilon * J_s_val
        
        # 于命令行记录本次迭代后发生的赛车轨迹线侧移收敛情况
        print(f"迭代 {iteration+1:2d}/{max_iter} | max|Δα|: {diff:.6f} | J_k: {J_k_val:.2e} | J_s: {J_s_val:.2e} | J_total: {J_total:.2e}")
        
        # SQP 松弛步长控制（Relaxation / Damping）
        # 如果直接采用 alpha_ref = alpha_new，纯曲率优化 (epsilon=0) 时优化器每轮会激进地
        # 将 alpha 推到边界极值，下一轮重新线性化后又回弹，导致严重振荡无法收敛。
        # 引入松弛因子 gamma ∈ (0, 1]，每轮只走部分步长：
        alpha_ref = (1 - gamma_adaptive) * alpha_ref + gamma_adaptive * alpha_new
        
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
    l_v = 4.7  # 车辆长度
    b_v = 2.0  # 车辆宽度
    ws = 0.5   # 额外安全距离
    
    print("加载并解析轨迹数据...")
    p, q_bound, v = load_track_data('track.csv')
    
    # 2. 从主函数暴露 SQP 的收敛控制因子以便用户测试
    gamma_normal = 0.5        # 大部分处于稳定计算状态下的正常牛顿步向界阻尼
    gamma_inaccurate = 0.1    # 探底触碰了极端硬曲率导致 OSQP 报告 inaccurate 时的回撤微小阻尼
    
    # 3. 计算纯最小曲率轨迹 (epsilon = 0)
    print("\n--- 计算最小曲率轨迹 (epsilon = 0) ---")
    r_opt_k, alpha_k = optimize_trajectory(
        p, v, l_v, b_v, ws, epsilon=0.0, max_iter=50,
        gamma_normal=gamma_normal, gamma_inaccurate=gamma_inaccurate
    )
    
    # 3. 计算平衡轨迹 (epsilon = 1000.0)
    print("\n--- 计算平衡轨迹 (epsilon = 1000.0) ---")
    r_opt_s, alpha_s = optimize_trajectory(
        p, v, l_v, b_v, ws, epsilon=1000.0, max_iter=50,
        gamma_normal=gamma_normal, gamma_inaccurate=gamma_inaccurate
    )

    # 4. 绘图
    # --- 主图：全赛道概览 + 弯道放大 + alpha 分布 ---
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    
    # 闭环连线：将第一个点拼接到最后，使得画图时首尾相连
    p_plot = np.vstack((p, p[0]))
    q_plot = np.vstack((q_bound, q_bound[0]))
    r_k_plot = np.vstack((r_opt_k, r_opt_k[0]))
    r_s_plot = np.vstack((r_opt_s, r_opt_s[0]))
    
    # 左上: 全赛道概览
    ax1 = axes[0, 0]
    ax1.plot(p_plot[:, 0], p_plot[:, 1], 'k-', linewidth=1.5, label='Left border (p)')
    ax1.plot(q_plot[:, 0], q_plot[:, 1], 'k-', linewidth=1.5, alpha=0.6, label='Right border (q)')
    ax1.plot(r_k_plot[:, 0], r_k_plot[:, 1], 'r-', linewidth=2.5, label='Min Curvature (eps=0)')
    ax1.plot(r_s_plot[:, 0], r_s_plot[:, 1], 'b--', linewidth=2.5, label='Balanced (eps=1000)')
    ax1.set_title("Full Track Overview", fontsize=16)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend(fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 右上: 放大弯道1 (左上角区域)
    ax2 = axes[0, 1]
    ax2.plot(p_plot[:, 0], p_plot[:, 1], 'k-', linewidth=1.5)
    ax2.plot(q_plot[:, 0], q_plot[:, 1], 'k-', linewidth=1.5, alpha=0.6)
    ax2.plot(r_k_plot[:, 0], r_k_plot[:, 1], 'r-', linewidth=3, label='Min Curvature')
    ax2.plot(r_s_plot[:, 0], r_s_plot[:, 1], 'b--', linewidth=3, label='Balanced')
    ax2.set_xlim(-650, -350)
    ax2.set_ylim(100, 350)
    ax2.set_title("Corner Detail (Top-Left)", fontsize=16)
    ax2.legend(fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 左下: 放大弯道2 (右下角区域)
    ax3 = axes[1, 0]
    ax3.plot(p_plot[:, 0], p_plot[:, 1], 'k-', linewidth=1.5)
    ax3.plot(q_plot[:, 0], q_plot[:, 1], 'k-', linewidth=1.5, alpha=0.6)
    ax3.plot(r_k_plot[:, 0], r_k_plot[:, 1], 'r-', linewidth=3, label='Min Curvature')
    ax3.plot(r_s_plot[:, 0], r_s_plot[:, 1], 'b--', linewidth=3, label='Balanced')
    ax3.set_xlim(100, 600)
    ax3.set_ylim(-500, -150)
    ax3.set_title("Corner Detail (Bottom-Right)", fontsize=16)
    ax3.legend(fontsize=12)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 右下: alpha 分布对比
    ax4 = axes[1, 1]
    indices = np.arange(len(alpha_k))
    ax4.plot(indices, alpha_k, 'r-', linewidth=1.5, label='Min Curvature alpha', alpha=0.8)
    ax4.plot(indices, alpha_s, 'b-', linewidth=1.5, label='Balanced alpha', alpha=0.8)
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Track center (alpha=0.5)')
    ax4.set_title("Alpha Distribution", fontsize=16)
    ax4.set_xlabel("Control Point Index")
    ax4.set_ylabel("Alpha Value")
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比图已保存为 'trajectory_comparison.png'")
    
    # 打印 alpha 统计信息
    print(f"\n最小曲率 alpha 统计: min={alpha_k.min():.4f}, max={alpha_k.max():.4f}, mean={alpha_k.mean():.4f}")
    print(f"平衡轨迹 alpha 统计: min={alpha_s.min():.4f}, max={alpha_s.max():.4f}, mean={alpha_s.mean():.4f}")

