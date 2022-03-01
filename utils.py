import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist


def pdist2(x, y, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = (y**2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist


def sparse_minconv(cost, candidates_edges0, candidates_edges1, alpha, unroll_factor = 32):
    reg_cost = torch.zeros_like(cost)
    split = np.array_split(np.arange(cost.shape[0]), unroll_factor)
    for i in range(unroll_factor):
        reg_cost[split[i]] = torch.min(cost[split[i]].unsqueeze(1) + alpha*(candidates_edges0[split[i]].unsqueeze(1) - candidates_edges1[split[i]].unsqueeze(2)).pow(2).sum(3), 2)[0]
    return reg_cost


def knn_graph(kpts, k, include_self=False):
    B, N, D = kpts.shape
    device = kpts.device

    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros(B, N, N).to(device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1

    return ind, dist * A, A


def lbp_graph(kpts_fixed, k, device='cuda'):
    A = knn_graph(kpts_fixed, k, include_self=False)[2][0]
    edges = A.nonzero()
    edges_idx = torch.zeros_like(A).long()
    edges_idx[A.bool()] = torch.arange(edges.shape[0]).to(device)
    edges_reverse_idx = edges_idx.t()[A.bool()]
    return edges, edges_reverse_idx


def farthest_point_sampling(kpts, num_points):
    _, N, _ = kpts.size()
    ind = torch.zeros(num_points).long()
    ind[0] = torch.randint(N, (1,))
    dist = torch.sum((kpts - kpts[:, ind[0], :]) ** 2, dim=2)
    for i in range(1, num_points):
        ind[i] = torch.argmax(dist)
        dist = torch.min(dist, torch.sum((kpts - kpts[:, ind[i], :]) ** 2, dim=2))

    while N < num_points:
        add_points = min(N, num_points - N)
        ind = torch.cat([ind[:N], ind[:add_points]])
        N += add_points

    return kpts[:, ind, :], ind


class TPS:
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device

        n = c.shape[0]
        f_dim = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device) * lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n + 4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n + 4, n + 4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.linalg.solve(A, v)
        # theta = torch.solve(v, A)[0]
        return theta

    @staticmethod
    def d(a, b):
        ra = (a ** 2).sum(dim=1).view(-1, 1)
        rb = (b ** 2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float('inf'))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r ** 2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()


def thin_plate_dense(x1, y1, shape, step, lambd=.0, unroll_step_size=2 ** 12):
    device = x1.device
    D, H, W = shape
    D1, H1, W1 = D // step, H // step, W // step

    x2 = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D1, H1, W1), align_corners=True).view(-1, 3)
    tps = TPS()
    theta = tps.fit(x1[0], y1[0], lambd)

    y2 = torch.zeros((1, D1 * H1 * W1, 3), device=device)
    N = D1 * H1 * W1
    n = math.ceil(N / unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        y2[0, j1:j2, :] = tps.z(x2[j1:j2], x1[0], theta)

    y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)
    y2 = F.interpolate(y2, (D, H, W), mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)

    return y2


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x, ind):
        B, N, D = x.shape
        k = ind.shape[2]

        y = x.view(B * N, D)[ind.view(B * N, k)].view(B, N, k, D)
        x = x.view(B, N, 1, D).expand(B, N, k, D)

        x = torch.cat([y - x, x], dim=3)

        x = self.conv(x.permute(0, 3, 1, 2))
        x = F.max_pool2d(x, (1, k))
        x = x.squeeze(3).permute(0, 2, 1)

        return x


class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()

        self.conv1 = EdgeConv(3, 32)
        self.conv2 = EdgeConv(32, 32)
        self.conv3 = EdgeConv(32, 64)

        self.conv4 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False),
                                   nn.InstanceNorm1d(64),
                                   nn.Conv1d(64, 64, 1))

    def forward(self, kpts_fixed, kpts_moving, k):
        fixed_ind = knn_graph(kpts_fixed, k, include_self=True)[0]
        x = self.conv1(kpts_fixed, fixed_ind)
        x = self.conv2(x, fixed_ind)
        x = self.conv3(x, fixed_ind)

        moving_ind = knn_graph(kpts_moving, k * 4, include_self=True)[0]
        y = self.conv1(kpts_moving, moving_ind)
        y = self.conv2(y, moving_ind)
        y = self.conv3(y, moving_ind)

        x = self.conv4(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.conv4(y.permute(0, 2, 1)).permute(0, 2, 1)

        return x, y


# self-ensembling
def get_current_consistency_weight(weight, epoch, rampup):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return weight * sigmoid_rampup(epoch, rampup)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def update_ema_variables(student, teacher, alpha):
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Augmenter(object):
    def __init__(self):
        self.rot = 10.
        self.transl = 0.1
        self.scale = 0.1

    def __call__(self, pcd1, pcd2):
        B, D, N = pcd1.shape
        dtype = pcd1.dtype
        device = pcd1.device

        angles = torch.deg2rad((torch.rand(B, dtype=dtype, device=device) - 0.5) * 2 * self.rot)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        z = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        rot_mat_x = torch.stack((ones, z, z, z, cos, -sin, z, sin, cos), dim=1).view(B, 3, 3)
        angles = torch.deg2rad((torch.rand(B, dtype=dtype, device=device) - 0.5) * 2 * self.rot)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        z = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        rot_mat_y = torch.stack((cos, z, sin, z, ones, z, -sin, z, cos), dim=1).view(B, 3, 3)
        angles = torch.deg2rad((torch.rand(B, dtype=dtype, device=device) - 0.5) * 2 * self.rot)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        z = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        rot_mat_z = torch.stack((cos, -sin, z, sin, cos, z, z, z, ones), dim=1).view(B, 3, 3)
        rot_mat = torch.bmm(torch.bmm(rot_mat_z, rot_mat_y), rot_mat_x)

        transl = (torch.rand(B, 1, 3, dtype=dtype, device=device) - 0.5) * 2 * self.transl
        scale = (torch.rand(B, 1, 1, dtype=dtype, device=device) - 0.5) * 2 * self.scale + 1

        pcd1 = torch.bmm(pcd1, rot_mat) * scale + transl
        pcd2 = torch.bmm(pcd2, rot_mat) * scale + transl

        return pcd1, pcd2, rot_mat, transl, scale