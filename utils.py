import torch
import torch.nn as nn
import numpy as np
import transformers
from transformers import GenerationMixin
def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse, mean_rmse
#添加
def Endmember_extract(x, p):
    [D, N] = x.shape
    # If no distf given, use Euclidean distance function
    Z1 = np.zeros((1, 1))
    O1 = np.ones((1, 1))
    # Find farthest point
    d = np.zeros((p, N))
    I = np.zeros((p, 1))
    V = np.zeros((1, N))
    ZD = np.zeros((D, 1))
    # if nargin<4
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), ZD)
    # d[0,i]=l1_distance(x[:,i].reshape(D,1),ZD)
    # else
    #     for i=1:N
    #         d(1,i)=distf(x(:,i),zeros(D,1),opt);

    I = np.argmax(d[0, :])

    # if nargin<4
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, I].reshape(D, 1))
        # d[0,i] = l1_distance(x[:,i].reshape(D,1),x[:,I].reshape(D,1))

    # else
    #     for i=1:N
    #         d(1,i)=distf(x(:,i),x(:,I(1)),opt);
    for v in range(1, p):
        # D=[d[0:v-2,I] ; np.ones((1,v-1)) 0]
        D1 = np.concatenate((d[0:v, I].reshape((v, I.size)), np.ones((v, 1))), axis=1)
        D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
        D4 = np.concatenate((D1, D2), axis=0)
        D4 = np.linalg.inv(D4)
        for i in range(N):
            D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
            V[0, i] = np.dot(np.dot(D3.T, D4), D3)

        I = np.append(I, np.argmax(V))
        # if nargin<4
        for i in range(N):
            # d[v,i]=l1_distance(x[:,i].reshape(D,1),x[:,I[v]].reshape(D,1))
            d[v, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, I[v]].reshape(D, 1))

        # else
        #     for i=1:N
        #         d(v,i)=distf(x(:,i),x(:,I(v)),opt);
    per = np.argsort(I)
    I = np.sort(I)
    d = d[per, :]
    return I, d
def Eucli_dist(x,y):
    a=np.subtract(x, y)
    return np.dot(a.T,a)
#结束

def compute_re(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt(((x_true - x_pred) ** 2).sum() / (img_w * img_h * img_c))


def compute_sad(inp, target):
    p = inp.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        inp_norm = np.linalg.norm(inp[:, i], 2)
        tar_norm = np.linalg.norm(target[:, i], 2)

        summation = np.matmul(inp[:, i].T, target[:, i])

        sad_err[i] = np.arccos(summation / (inp_norm * tar_norm))
    mean_sad = np.mean(sad_err)
    return sad_err, mean_sad


def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape
    inp = torch.reshape(inputs, (band, h * w))
    out = torch.norm(inp, p='nuc')
    return out


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()

    def __call__(self, inp, decay):
        inp = torch.sum(inp, 0, keepdim=True)
        loss = Nuclear_norm(inp)
        return decay * loss


class SumToOneLoss(nn.Module):
    def __init__(self, device):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float, device=device))
        self.loss = nn.L1Loss(reduction='sum')

    def get_target_tensor(self, inp):
        target_tensor = self.one
        return target_tensor.expand_as(inp)

    def __call__(self, inp, gamma_reg):
        inp = torch.sum(inp, 1)
        target_tensor = self.get_target_tensor(inp)
        loss = self.loss(inp, target_tensor)
        return gamma_reg * loss


class SAD(nn.Module):
    def __init__(self, num_bands):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, inp, target):
        try:
            input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
                                              inp.view(-1, self.num_bands, 1)))
            target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
                                               target.view(-1, self.num_bands, 1)))

            summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
            angle = torch.acos(summation / (input_norm * target_norm))

        except ValueError:
            return 0.0

        return angle


class SID(nn.Module):
    def __init__(self, epsilon: float = 1e5):
        super(SID, self).__init__()
        self.eps = epsilon

    def forward(self, inp, target):
        normalize_inp = (inp / torch.sum(inp, dim=0)) + self.eps
        normalize_tar = (target / torch.sum(target, dim=0)) + self.eps
        sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) +
                        normalize_tar * torch.log(normalize_tar / normalize_inp))

        return sid
