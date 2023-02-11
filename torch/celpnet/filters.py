import torch
from torch import nn
import torch.nn.functional as F
import math

def toeplitz_from_filter(a):
    device = a.device
    L = a.size(-1)
    size = (*(a.shape[:-1]), L, L)
    A = torch.zeros(size).to(device)
    #print(L, A.shape)
    #Compute lower-triangular Toeplitz
    if True:
        #Copy in log2(L) steps
        A[:,:,:,0] = a[:,:,:]
        LL = int(math.log2(L))
        N=1
        for i in range(LL):
            A[:,:,N:,N:2*N] = A[:,:,:-N,:N]
            N *= 2
        A[:,:,N:,N:L] = A[:,:,:-N,:L-N]
    else:
        for i in range(L):
            A[:,:,i:,i] = a[:,:,:L-i]
    return A

def filter_iir_response(a, N):
    device = a.device
    L = a.size(-1)
    ar = a.flip(dims=(2,))
    size = (*(a.shape[:-1]), N)
    R = torch.zeros(size).to(device)
    R[:,:,0] = torch.ones((*(a.shape[:-1]))).to(device)
    for i in range(1, L):
        R[:,:,i] = - torch.sum(ar[:,:,L-i-1:-1] * R[:,:,:i], axis=-1)
        #R[:,:,i] = - torch.einsum('ijk,ijk->ij', ar[:,:,L-i-1:-1], R[:,:,:i])
    for i in range(L, N):
        R[:,:,i] = - torch.sum(ar[:,:,:-1] * R[:,:,i-L+1:i], axis=-1)
        #R[:,:,i] = - torch.einsum('ijk,ijk->ij', ar[:,:,:-1], R[:,:,i-L+1:i])
    return R

if __name__ == '__main__':
    #a = torch.tensor([ [[1, -.9, 0.02], [1, -.8, .01]], [[1, .9, 0], [1, .8, 0]]])
    a = torch.tensor([ [[1, -.9, 0.02], [1, -.8, .01]]])
    A = toeplitz_from_filter(a)
    #print(A)
    R = filter_iir_response(a, 5)
    
    RA = toeplitz_from_filter(R)
    print(RA)
