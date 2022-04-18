import os
import solvers.gurobi as gb
import problem_classes.random_qp as rqp
import problem_classes.lasso as lasso
import problem_classes.svm as svm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import sparse
from scipy.sparse import csr_matrix
import pandas
import cvxpy

def print_csr_matrix(path,mat):
    matrix=csr_matrix(mat)
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
    name_ls={'status':res.status,'obj_val':res.obj_val,'niter':res.niter,'solve time':res.run_time,'x':res.x.tolist(),'y':res.y.tolist()}
    with open(f'{path}.txt', 'w') as f:
        for key, value in name_ls.items(): 
            f.write('%s: ' % key)
            if isinstance(value,int) or isinstance(value,float) or isinstance(value,str):
                print(value , file = f)
            else:
                print(*value , file = f)

for n_size in [1,2,5,10,20,50,80,100]:
    classes_name = ['Random','Lasso','SVM']
    classes = [rqp.RandomQPExample(n=n_size),lasso.LassoExample(n=n_size),svm.SVMExample(n=n_size)]
    results = []
    for (i,example) in enumerate(classes):
        A,P,q,l,u=[],[],[],[],[]
        if classes_name[i] == "Random":
            example = rqp.RandomQPExample(n=n_size*5)
            A = example.A
            P = example.P
            q = example.q
            l = example.l
            u = example.u
        else:
            A = example.qp_problem["A"]
            P = example.qp_problem["P"]
            q = example.qp_problem["q"]
            l = example.qp_problem["l"]
            u = example.qp_problem["u"]

        # Save results
        dir = os.path.join(os.getcwd(),'instances')
        if not os.path.isdir(dir):
            os.mkdir(dir)
        path = os.path.join(dir, f'{classes_name[i]}')
        if not os.path.isdir(path):
            os.mkdir(path)
        ipath = os.path.join(path, f'instance-{A.shape[0]}x{A.shape[1]}')
        if not os.path.isdir(ipath):
            os.mkdir(ipath)
        print_csr_matrix(os.path.join(ipath,'P'),P)
        print_csr_matrix(os.path.join(ipath,'A'),A)
        print_vector(os.path.join(ipath,'q'),q)
        print_vector(os.path.join(ipath,'l'),l)
        print_vector(os.path.join(ipath,'u'),u)
        # Create solver instance
        rqp_solve=gb.GUROBISolver()
        # help(rqp_solve.solve)
        result=rqp_solve.solve(example)
        print_res(os.path.join(ipath,'gurobi_sol'),result)
        results.append(result)