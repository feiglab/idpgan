import numpy as np

import torch
import torch.nn as nn

# from idp_lib.coords import get_dmap_from_xyz


def get_activation(activation, slope=0.2):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "lrelu":
        return nn.LeakyReLU(slope)
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise KeyError(activation)
        

# def get_layer_sn(layer, use_sn=False):
#     if not use_sn:
#         return layer
#     else:
#         # torch.nn.utils.spectral_norm(layer)
#         # return layer
#         return torch.nn.utils.spectral_norm(layer)


# class FCBlock(nn.Module):

#     def __init__(self, in_dim, out_dim, use_bn,
#                  activation="relu",
#                  use_sn=False):
        
#         super(FCBlock, self).__init__()
        
#         linear = nn.Linear(in_dim, out_dim)
#         if use_sn:
#             # torch.nn.utils.parametrizations.spectral_norm(linear)
#             torch.nn.utils.spectral_norm(linear)
#         layers = [linear]
#         if use_bn:
#             layers.append(nn.BatchNorm1d(out_dim))
#             # layers.append(nn.LayerNorm(out_dim))
#         if activation is not None:
#             layers.append(get_activation(activation))
#         self.fc = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.fc(x)
    
    
# class RBFLayer(nn.Module):
    
#     def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
#         super(RBFLayer, self).__init__()
#         offset = torch.linspace(start, stop, num_gaussians)
#         self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
#         offset = offset.view(1, 1, offset.shape[0])  # Custom.
#         self.register_buffer('offset', offset)

#     def forward(self, dist):
#         # dist = dist.view(-1, 1) - self.offset.view(1, -1)
#         # return torch.exp(self.coeff * torch.pow(dist, 2))
#         dist = dist.view(dist.shape[0], dist.shape[1], 1)
#         # print(dist.shape, self.offset.shape)
#         dist = dist - self.offset  # .view(1, -1)
#         return torch.exp(self.coeff * torch.pow(dist, 2))
    
# # check_rbf = False
# # if check_rbf:
# #     rbf_layer = RBFLayer(start=0.0, stop=8.0, num_gaussians=50)
# #     in_rbf = torch.tensor(data_train[0:250])
# #     print(in_rbf.shape)
# #     out_rbf = rbf_layer(in_rbf)
# #     print(out_rbf.shape)
# #     i = 250
# #     j = in_rbf[i].argmax()
# #     print("in:", in_rbf[i][j])
# #     # print("off:", rbf_layer.offset.view(-1))
# #     # print("out:", out_rbf[i][j])
# #     print("out:")
# #     for m, v in zip(rbf_layer.offset.view(-1), out_rbf[i][j]):
# #         print(m.item(), in_rbf[i][j].item(), v.item())


# def randn_truncation(shape, device):
#     r = torch.randn(shape, device=device).view(-1)
#     while True:
#         m = torch.abs(r) >= 1.5
#         n_out = m.sum()
#         if n_out == 0:
#             break
#         rr = torch.randn(n_out, device=device)
#         r[m] = rr
#     return r.view(shape)


# def predict_chunks(net, noise, x=None, chunk_size=250):
#     with torch.no_grad():
#         fake_xyz = []
#         for chunk_i in range(0, noise.shape[0], chunk_size):
#             if x is None:
#                 fake_xyz_i = net(noise[chunk_i:chunk_i+chunk_size],
#                                  get_coords=True).cpu().numpy()
#             else:
#                 fake_xyz_i = net(noise[chunk_i:chunk_i+chunk_size],
#                                  x[chunk_i:chunk_i+chunk_size],
#                                  get_coords=True).cpu().numpy()
#             fake_xyz.append(fake_xyz_i)
#         fake_xyz = np.concatenate(fake_xyz)
#         fake_dmap = get_dmap_from_xyz(fake_xyz)
#         # fake_xyz = netG(fixed_noise, get_coords=True).cpu().numpy()
#         # fake_dmap = get_dmap_from_xyz(fake_xyz)
#     return fake_xyz, fake_dmap