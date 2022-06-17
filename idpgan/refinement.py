import numpy as np
import torch
import torch.nn as nn


def refinement_energy(xyz, xyz_original=None, get_grad=True,
                      pos_c=1.0,
                      rt=1, epsilon=1e-12):
    
    prot_len = xyz.shape[1]
    
    dmap = torch.sqrt(
             torch.sum(
               torch.square(xyz[:,None,...] - xyz[:,:,None,...]),
             axis=-1)+epsilon)
    
    # Bond length.
    bl_m = 0.3832356
    bl_s = 0.024169842
    
    bl_ids = torch.cat([torch.arange(0, prot_len-1).view(-1, 1),
                        torch.arange(1, prot_len).view(-1, 1)], axis=1).T
    bl_vals = dmap[:, bl_ids[0], bl_ids[1]]
    e_bl = torch.sum(rt*0.5*(bl_vals-bl_m)**2/bl_s, axis=1)
    
    # Clash.
    nb_min = 0.5812876224517822  # New FF (20-120).
    nb_tau = 0
    
    nb_ids = torch.triu_indices(prot_len, prot_len, offset=3)
    nb_vals = dmap[:, nb_ids[0], nb_ids[1]]
    e_nb = torch.sum(
             rt*nn.functional.relu(nb_min - nb_tau - nb_vals),  # **2/0.1,
           axis=1)
    
    # Positional restraints.
    if xyz_original is not None:
        e_pos = torch.sum(
                  pos_c*rt*torch.sum(
                    torch.square(xyz - xyz_original),
                  axis=2),
                axis=1)
    else:
        e_pos = 0
    e_tot = e_bl+e_nb+e_pos
    
    results = e_tot, {"bl": e_bl, "nb": e_nb, "pos": e_pos}
    if not get_grad:
        return results
    else:
        grad = torch.autograd.grad(inputs=xyz, outputs=e_tot,
                                   retain_graph=True, create_graph=True,
                                   grad_outputs=torch.ones_like(e_tot)                          
                                  )[0]
        return results[0], results[1], grad


def refine(xyz, xyz_original=None, n_iters=100, epsilon=0.001,
           use_z=True, mh_correction=True, verbose=True):
    
    if mh_correction and not use_z:
        raise ValueError("When using 'mh_correction', 'use_z' must"
                         " be set to 'True'.")
        
    sampling_device = xyz.device
    history = {"e": []}
        
    for i in range(n_iters):

        if verbose:
            print("Iter %s/%s." % (i+1, n_iters), end=" ")
            
        if i == 0:
            xyz_i = xyz.clone().detach().requires_grad_(True).to(sampling_device)
        else:
            xyz_i = xyz_i.clone().detach().requires_grad_(True).to(sampling_device)
            
        e_i, _, grad_i = refinement_energy(xyz_i, xyz_original=xyz_original)
        if use_z:
            z_i = torch.randn_like(xyz_i, device=sampling_device)
        else:
            z_i = 0
        xyz_c = xyz_i - epsilon*0.5*grad_i + np.sqrt(epsilon)*z_i
        
        if not mh_correction:
            xyz_i = xyz_c
        else:
            with torch.no_grad():
                e_c, _ = refinement_energy(xyz_c, xyz_original=xyz_original,
                                           get_grad=False)
            temperature = 0.25
            u_i = torch.exp(-(e_c-e_i)/temperature)
            w_i = torch.rand_like(u_i, device=sampling_device)
            # print(u_i)
            # print(u_i > 0)
            # torch.where((t == 0), decoder_nll, kl_loss)
            flag_i = u_i > w_i
            flag_i = flag_i.reshape(-1, 1, 1).repeat(1, xyz_i.shape[1], xyz_i.shape[2])
            xyz_i = torch.where(flag_i, xyz_c, xyz_i)
            
        e_global_i = e_i.mean().item()
        if verbose:
            print("Energy: %s" % e_global_i)
        history["e"].append(e_global_i)
        # print(_["bl"].mean().item(), _["nb"].mean().item())
    xyz_opt = xyz_i.clone().detach().requires_grad_(False).to(sampling_device)
    return xyz_opt, history


if __name__ == "__main__":
    xyz_gen = torch.tensor(ev_gen.xyz_pred)
    xyz_original = xyz_gen
    xyz_opt, history = refine(xyz_gen, xyz_original=xyz_gen, 
                              use_z=False, mh_correction=False, n_iters=5)