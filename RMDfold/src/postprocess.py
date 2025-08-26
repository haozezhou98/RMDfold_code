import torch
import math
import numpy as np
import torch.nn.functional as F


def constraint_matrix_batch(x):
    """
    this function is referred from e2efold utility function, located at https://github.com/ml4bio/e2efold/tree/master/e2efold/common/utils.py
    """
    base_a= x[:, :, 0]
    base_u= x[:, :, 1]
    base_c= x[:, :, 2]
    base_g= x[:, :, 3]
    batch= base_a.shape[0]
    length= base_a.shape[1]
    au= torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua= au + torch.transpose(au, -1, -2)
    cg= torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc= cg + torch.transpose(cg, -1, -2)
    ug= torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu= ug + torch.transpose(ug, -1, -2)

    mask= au_ua + cg_gc + ug_gu

    #Mask sharp loop
    for b1 in range(batch):
      for d2 in range(1,2):
        for i in range(d2,length):
          mask[b1,i-d2,i]= 0
        for i in range(length-d2):
          mask[b1,i+d2,i]= 0

    return mask

def symmetric_a(a_hat,m1):
    a= a_hat* a_hat
    b= torch.transpose(a, -1, -2)
    a= (a+ b)/ 2
    a= a* m1
    return a




def constraint_matrix_batch_addnc(x):
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    ## add non-canonical pairs
    ac = torch.matmul(base_a.view(batch, length, 1), base_c.view(batch, 1, length))
    ac_ca = ac + torch.transpose(ac, -1, -2)
    ag = torch.matmul(base_a.view(batch, length, 1), base_g.view(batch, 1, length))
    ag_ga = ag + torch.transpose(ag, -1, -2)
    uc = torch.matmul(base_u.view(batch, length, 1), base_c.view(batch, 1, length))
    uc_cu = uc + torch.transpose(uc, -1, -2)
    aa = torch.matmul(base_a.view(batch, length, 1), base_a.view(batch, 1, length))
    uu = torch.matmul(base_u.view(batch, length, 1), base_u.view(batch, 1, length))
    cc = torch.matmul(base_c.view(batch, length, 1), base_c.view(batch, 1, length))
    gg = torch.matmul(base_g.view(batch, length, 1), base_g.view(batch, 1, length))
    return au_ua + cg_gc + ug_gu + ac_ca + ag_ga + uc_cu + aa + uu + cc + gg

def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a

def sign(x):
    return (x > 0).type(x.dtype)


def soft_sign(x):
    k = 1
    return 1.0/(1.0+torch.exp(-2*k*x))


def postprocess_new(u1, x1, L1, lr_min, lr_max, num_itr, rho=0.0, with_L1=False,s9=math.log(9.0)):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m= constraint_matrix_batch(x1).float()
    u1= soft_sign(u1 - s9)* u1

    # Initialization
    a_hat= (torch.sigmoid(u1))* soft_sign(u1-s9).detach()
    s_hat= symmetric_a(a_hat, m)

    # constraint Y_hat
    sumcol_hat= torch.sum(s_hat, dim=-1)
    lambd= F.relu(sumcol_hat- 1).detach()


    # gradient descent approach
    for t in range(num_itr):
      s_hat= symmetric_a(a_hat, m)
      sumcol_hat= torch.sum(s_hat, dim=-1)
      grad_a= (lambd * soft_sign(sumcol_hat- 1)).unsqueeze_(-1).expand(u1.shape)-u1/2
      grad= a_hat* m* (grad_a + torch.transpose(grad_a, -1, -2))
      a_hat-= lr_min * grad
      a_hat= F.relu(a_hat)
      lr_min*= 0.99

      if with_L1:
        a_hat= F.relu(torch.abs(a_hat) - rho* lr_min)

      lambd_grad= F.relu(torch.sum( symmetric_a(a_hat, m),dim=-1)- 1 )
      lambd+= lr_max * lambd_grad
      lr_max*= 0.99

    # Constraint A+AT
    ya= symmetric_a(a_hat,m)
    s2=torch.squeeze(torch.sum((ya>0.5),1))
    
    # Find single row
    for a1 in range(L1):
      if (s2[a1]>1):
        Id1= torch.nonzero(ya[0,a1,:])
        Idm= torch.argmax(ya[0,a1,:])
        for a2 in Id1:
          if not (a2==Idm):
            ya[0,a1,a2]=0
            ya[0,a2,a1]=0

    return ya


def postprocess_new_nc(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False,s=math.log(9.0)):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m = constraint_matrix_batch_addnc(x).float()
    # m = 1.0
    # u = (u > math.log(9.0)).type(torch.FloatTensor) * u
    u = soft_sign(u - s) * u

    # initialization
    a_hat = (torch.sigmoid(u)) * soft_sign(u - s).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

    # gradient descent
    for t in range(num_itr):

        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99

    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a



