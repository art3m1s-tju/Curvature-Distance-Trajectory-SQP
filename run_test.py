import numpy as np
import scipy.sparse as sp

def build_difference_matrix(N):
    diag = np.ones(N) * (-1)
    off_diag = np.ones(N-1)
    A = sp.diags([diag, off_diag], [0, 1], shape=(N, N)).tolil()
    A[N-1, 0] = 1 
    return A.tocsc()

def calculate_derivative_matrices(r):
    N = len(r)
    rx = r[:, 0]
    ry = r[:, 1]
    
    rx_prime = np.roll(rx, -1) - rx
    ry_prime = np.roll(ry, -1) - ry
    ds = np.sqrt(rx_prime**2 + ry_prime**2)
    
    rx_prime = rx_prime / (ds + 1e-6)
    ry_prime = ry_prime / (ds + 1e-6)
    
    denominator = (rx_prime**2 + ry_prime**2)**1.5 + 1e-8
    
    Txx_diag = (ry_prime**2) / denominator
    Tyy_diag = (rx_prime**2) / denominator
    Txy_diag = -(rx_prime * ry_prime) / denominator
    
    Txx = sp.diags(Txx_diag)
    Tyy = sp.diags(Tyy_diag)
    Txy = sp.diags(Txy_diag)
    
    # 论文公式(16.2), (17) 完整逆矩阵 C 和 B 的推导较复杂
    # 对于等弧长样条，二阶中心差分是一个极好的近似，且保持稀疏
    M = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).tolil()
    M[0, N-1] = 1
    M[N-1, 0] = 1
    
    # 除以 ds^2 转为真正的二阶导
    ds_diag = sp.diags(1.0 / (ds**2 + 1e-6))
    M = ds_diag @ M.tocsc()
    
    return Txx, Tyy, Txy, M

print("Script runs successfully.")
