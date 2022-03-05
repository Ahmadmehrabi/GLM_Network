import torch
import numpy as np

def der_fun(model,x_star,act_der,num_lay):
    ind = [2*i for i in range(num_lay)]
    A = [model[j].weight for j in ind]
    x = x_star
    z = []
    sig_pri = []
    for j in ind[:-1]:
        zz = model[j](x)
        x = model[j+1](zz)
        z.append(zz)
        sig_pri.append(act_der(zz))
        
     
    der = A[0].T
    for i in range(num_lay-1):
        t = sig_pri[i]*der
        der = torch.matmul(t,A[i+1].T)

    return der  

  
