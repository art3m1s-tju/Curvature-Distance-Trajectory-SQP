import numpy as np
import scipy.sparse as sp
import osqp

# Mock problem to check OSQP settings
P = sp.csc_matrix([[4, 1], [1, 2]])
q = np.array([1, 1])
A = sp.csc_matrix([[1, 1], [1, 0], [0, 1]])
l = np.array([1, 0, 0])
u = np.array([1, 0.7, 0.7])

prob = osqp.OSQP()
# Setup requires arguments as keyword arguments in OSQP 0.6.3 vs positional
prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
res = prob.solve()
print(res.x)
