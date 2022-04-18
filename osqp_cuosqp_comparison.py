import os
import osqp
import cuosqp
import osqp_benchmarks
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import sparse
from scipy.sparse import csr_matrix
import pandas
import cvxpy

def print_csr_matrix(path,matrix):
    name_ls={'m':matrix.shape[0],'n':matrix.shape[1],'nnz':matrix.nnz,'value':matrix.data.tolist(),'row pointer':matrix.indptr.tolist(),'column index':matrix.indices.tolist()}
    with open(f'{path}.txt', 'w') as f:
        for key, value in name_ls.items(): 
            if isinstance(value,int):
                print(value , file = f)
            else:
                print(*value , file = f)

def print_vector(path,vector):
    with open(f'{path}.txt', 'w') as f:
        print(len(vector) , file = f)
        print(*vector , file = f)

def print_res(path,res):
    name_ls={'iter':res.info.iter,'status':res.info.status,'obj_val':res.info.obj_val,'x':res.x.tolist(),'y':res.y.tolist(),'pri_res':res.info.pri_res,'dua_res':res.info.dua_res,'solve_time':res.info.solve_time,'rho_estimate':res.info.rho_estimate,'rho_updates':res.info.rho_updates}
    with open(f'{path}.txt', 'w') as f:
        for key, value in name_ls.items(): 
            f.write('%s: ' % key)
            if isinstance(value,int) or isinstance(value,float) or isinstance(value,str):
                print(value , file = f)
            else:
                print(*value , file = f)

# Generate problem data
sp.random.seed(1)
OSQP_solve_time,cuOSQP_solve_time=[],[]
prob_size=[]
# for m in [1,2,5,10,100,1000]:
#     for n in [1,2,5,10,100,1000,5000]:
for m in np.logspace(1,3,num=20,dtype=int):
    n = m*10
    Ad = sparse.random(m, n, density=0.5, format='csr')
    x_true = np.random.randn(n) / np.sqrt(n)
    ind95 = (np.random.rand(m) < 0.95).astype(float)
    b = Ad.dot(x_true) + np.multiply(0.5*np.random.randn(m), ind95) \
        + np.multiply(10.*np.random.rand(m), 1. - ind95)

    # OSQP data
    Im = sparse.eye(m)
    P = sparse.block_diag([sparse.csc_matrix((n, n)), 2*Im,
                        sparse.csc_matrix((2*m, 2*m))],
                        format='csr')
    q = np.append(np.zeros(m+n), 2*np.ones(2*m))
    A = sparse.bmat([[Ad,   -Im,   -Im,   Im],
                    [None,  None,  Im,   None],
                    [None,  None,  None, Im]], format='csr')
    l = np.hstack([b, np.zeros(2*m)])
    u = np.hstack([b, np.inf*np.ones(2*m)])

    dir = os.getcwd()
    path = os.path.join(dir, f'instance-{A.shape[0]}x{A.shape[1]}')
    os.mkdir(path)
    print_csr_matrix(os.path.join(path,'P'),P)
    print_csr_matrix(os.path.join(path,'A'),A)
    print_vector(os.path.join(path,'q'),q)
    print_vector(os.path.join(path,'l'),l)
    print_vector(os.path.join(path,'u'),u)

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u)

    # Solve problem
    res = prob.solve()
    print_res(os.path.join(path,'osqp_sol'),res)
    # Create an cuOSQP object
    prob2 = cuosqp.OSQP()
    prob2.setup(P, q, A, l, u)
    res2 = prob2.solve()
    prob_size.append((A.shape[0],A.shape[1]))
    cuOSQP_solve_time.append(res2.info.solve_time)
    OSQP_solve_time.append(res.info.solve_time)
    print_res(os.path.join(path,'cuosqp_sol'),res2)


OSQP_solve_time=np.array(OSQP_solve_time)
cuOSQP_solve_time=np.array(cuOSQP_solve_time)
ratio=OSQP_solve_time/cuOSQP_solve_time
prob_size1=np.array([np.prod(i) for i in prob_size])
sort_idx=np.argsort(prob_size1)
ratio=ratio[sort_idx]
prob_size1=prob_size[sort_idx]
# Plot ratio over problem size, log scale
mpl.rcParams['font.size']=15
fig, ax=plt.subplots(figsize = (10, 4))
ax.plot(prob_size1, ratio ,'ro')
plt.axhline(y=1, color='k', linestyle='-')
ax.set_xlabel('Problem size')
ax.set_ylabel('Ratio')
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('Ratio.png',bbox_inches='tight',pad_inches=0.2)
######
fig, ax=plt.subplots(figsize = (10, 4))
ax.plot(prob_size1, OSQP_solve_time ,'ro',label='OSQP')
ax.plot(prob_size1, cuOSQP_solve_time ,'bs',label='cuOSQP')
ax.set_xlabel('Problem size')
ax.set_ylabel('Solving time')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('Solve_time.png',bbox_inches='tight',pad_inches=0.2)
