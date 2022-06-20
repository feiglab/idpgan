import numpy as np

import torch
import torch.nn as nn

from idpgan.common import get_activation


class IdpGANBlock(nn.Module):
    
    def __init__(self, embed_dim, d_model=192, nhead=12,
                 dim_feedforward=128,
                 dropout=0.1, layer_norm_eps=1e-5,
                 embed_dim_2d=None,
                 use_bias_2d=True,
                 embed_dim_1d=None,
                 project_embed_1d=False,
                 activation="relu",
                 dp_attn_norm="d_model"):
        
        super(IdpGANBlock, self).__init__()
        
        self.use_norm = layer_norm_eps is not None
        self.project_embed_1d = project_embed_1d
        self.use_embed_2d = embed_dim_2d is not None
        self.use_embed_1d = embed_dim_1d is not None
        
        # Transformer layer.
        self.idp_attn = IdpGANLayer(in_dim=embed_dim,
                                    d_model=d_model,
                                    nhead=nhead,
                                    dp_attn_norm=dp_attn_norm,
                                    in_dim_2d=embed_dim_2d,
                                    use_bias_2d=use_bias_2d)

        if self.project_embed_1d:
            self.project_embed_1d_linear = nn.Linear(embed_dim_1d, embed_dim)
            contact_embed_1d_dim = 0
        else:
            contact_embed_1d_dim = embed_dim_1d
        updater_in_dim = embed_dim+contact_embed_1d_dim
            
        # Updater module (implementation of Feedforward model of the original
        # transformer).
        self.linear1 = nn.Linear(updater_in_dim, dim_feedforward)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        if self.use_norm:
            self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

        self.update_module = [self.linear1, self.activation]
        if self.dropout_p is not None:
            self.update_module.append(self.dropout)
        self.update_module.append(self.linear2)
        self.update_module = nn.Sequential(*self.update_module)
            

    def _check_embedding(self, use_embed, embed, var_name, embed_name):
        if use_embed:
            if embed is None:
                raise ValueError("'%s' can not be None when using %s embeddings." % (
                                 var_name, embed_name))
        else:
            if embed is not None:
                raise ValueError("'%s' must be None when using %s embeddings." % (
                                 var_name, embed_name))


    def forward(self, s, x=None, p=None):
        # Check the input.
        self._check_embedding(self.use_embed_2d, p, "p", "2d")
        self._check_embedding(self.use_embed_1d, x, "x", "1d")
        
        # Actually run the transformer block.
        s2 = self.idp_attn(s, s, s, p=p)[0]
        if self.dropout_p is not None:
            s = s + self.dropout1(s2)
        else:
            s = s + s2
        if self.use_norm:
            s = self.norm1(s)
        
        # Use amino acid conditional information.
        if self.use_embed_1d:
            if self.project_embed_1d:
                x = self.project_embed_1d_linear(x)
                um_in = s + x
            else:
                um_in = torch.cat([s, x], axis=-1)
        else:
            um_in = s
        
        # Use the updater module.
        s2 = self.update_module(um_in)
        if self.dropout_p is not None:
            s = s + self.dropout2(s2)
        else:
            s = s + s2
        if self.use_norm:
            s = self.norm2(s)
        return s
        

class IdpGANLayer(nn.Module):
    
    def __init__(self, in_dim,
                 d_model, nhead,
                 dp_attn_norm="d_model",
                 in_dim_2d=None,
                 use_bias_2d=True):
        super(IdpGANLayer, self).__init__()
        """d_model = c*n_head"""
        
        head_dim = d_model // nhead
        assert head_dim * nhead == d_model, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim
        self.in_dim_2d = in_dim_2d
        
        if dp_attn_norm not in ("d_model", "head_dim"):
            raise KeyError("Unkown 'dp_attn_norm': %s" % dp_attn_norm)
        self.dp_attn_norm = dp_attn_norm
        
        # Linear layers for q, k, v for dot product affinities.
        self.q_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.k_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.v_linear = nn.Linear(in_dim, self.d_model, bias=False)
        
        # Output layer.
        out_linear_in = self.d_model
        self.out_linear = nn.Linear(out_linear_in, in_dim)
        
        # Branch for 2d representation.
        self.mlp_2d = nn.Sequential(# nn.Linear(in_dim_2d, in_dim_2d),
                                    # nn.ReLU(),
                                    # nn.Linear(in_dim_2d, in_dim_2d),
                                    # nn.ReLU(),
                                    nn.Linear(in_dim_2d, self.nhead, bias=use_bias_2d))
        
        
    verbose = False
    def forward(self, s, _k, _v, p=None):
        
        #----------------------
        # Prepare the  input. -
        #----------------------
        
        # Receives a (L, N, I) tensor.
        # L: sequence length,
        # N: batch size,
        # I: input embedding dimension.
        seq_l, b_size, _e_size = s.shape
        if self.dp_attn_norm == "d_model":
            w_t = 1/np.sqrt(self.d_model)
        elif self.dp_attn_norm == "head_dim":
            w_t = 1/np.sqrt(self.head_dim)
        else:
            raise KeyError(self.dp_attn_norm)
        
        #----------------------------------------------
        # Compute q, k, v for dot product affinities. -
        #----------------------------------------------
        
        # Compute q, k, v vectors. Will reshape to (L, N, D*H).
        # D: number of dimensions per head,
        # H: number of head,
        # E = D*H: embedding dimension.
        q = self.q_linear(s)
        k = self.k_linear(s)
        v = self.v_linear(s)

        # Actually compute dot prodcut affinities.
        # Reshape first to (N*H, L, D).
        q = q.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        q = q * w_t
        k = k.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)

        # Then perform matrix multiplication between two batches of matrices.
        # (N*H, L, D) x (N*H, D, L) -> (N*H, L, L)
        dp_aff = torch.bmm(q, k.transpose(-2, -1))
            
        #--------------------------------
        # Compute the attention values. -
        #--------------------------------
        
        tot_aff = dp_aff
        
        # Use the 2d branch.
        p = self.mlp_2d(p)
        # (N, L1, L2, H) -> (N, H, L2, L1)
        p = p.transpose(1, 3)
        # (N, H, L2, L1) -> (N, H, L1, L2)
        p = p.transpose(2, 3)
        # (N, H, L1, L2) -> (N*H, L1, L2)
        p = p.contiguous().view(b_size*self.nhead, seq_l, seq_l)
        tot_aff = tot_aff + p
            
        attn = nn.functional.softmax(tot_aff, dim=-1)
        # if dropout_p > 0.0:
        #     attn = dropout(attn, p=dropout_p)
        
        #-----------------
        # Update values. -
        #-----------------
        
        # Update values obtained in the dot product affinity branch.
        s_new = torch.bmm(attn, v)
        # Reshape the output, that has a shape of (N*H, L, D) back to (L, N, D*H).
        s_new = s_new.transpose(0, 1).contiguous().view(seq_l, b_size, self.d_model)
        
        # Compute the ouput.
        output = s_new
        output = self.out_linear(output)
        return (output, )
    
    
class IdpGANGenerator(nn.Module):
    
    def __init__(self, nz,
                 embed_dim,
                 d_model,
                 nhead, dim_feedforward, dropout,
                 num_layers,
                 layer_norm_eps,
                 dp_attn_norm="d_model",  # "d_model", "head_dim"
                 # MLP modules params.
                 n_hl_out=1,
                 n_hl_embed=1,
                 activation="relu",
                 # Embedding params.
                 use_embed_repeat=True,
                 embed_dim_1d=32,
                 project_embed_1d=False,
                 pos_embed="af2_2d_ib", pos_embed_dim=64,
                 use_bias_2d=True,
                 pos_embed_max_l=24,
                 *args, **kwargs):
        
        super(IdpGANGenerator, self).__init__()
        
        # Positional embedding.
        self.use_embed_repeat = use_embed_repeat
        self.pe_max_l = pos_embed_max_l
        self.embed_pos = nn.Embedding(self.pe_max_l*2+1, pos_embed_dim)
        
        # Embed z to a d_model space.
        self.nz = nz
        self.n_hl_embed = n_hl_embed
        self.embed_x = [nn.Linear(nz, embed_dim),
                        get_activation(activation)]
        for l in range(self.n_hl_embed-1):
            self.embed_x.extend([nn.Linear(embed_dim, embed_dim),
                                 get_activation(activation)])
        self.embed_x.append(nn.Linear(embed_dim, embed_dim))
        self.embed_x = nn.Sequential(*self.embed_x)
            
        # Embed amino acid sequences.
        self.embed_aa = nn.Embedding(20, embed_dim_1d)
        
        # Transformer blocks.
        self.num_layers = num_layers
        self.transformer = []
        for i in range(self.num_layers):
            if self.use_embed_repeat:
                _embed_dim_2d = pos_embed_dim
                _embed_dim_1d = embed_dim_1d
            else:
                _embed_dim_2d = pos_embed_dim if i == 0 else None
                _embed_dim_1d = embed_dim_1d if i == 0 else None
            l_i = IdpGANBlock(embed_dim=embed_dim,
                              d_model=d_model, nhead=nhead,
                              dim_feedforward=dim_feedforward,
                              dropout=dropout,
                              layer_norm_eps=layer_norm_eps,
                              embed_dim_2d=_embed_dim_2d,
                              use_bias_2d=use_bias_2d,
                              embed_dim_1d=_embed_dim_1d,
                              project_embed_1d=project_embed_1d,
                              activation=activation,
                              dp_attn_norm=dp_attn_norm)
            self.transformer.append(l_i)
        self.transformer = nn.ModuleList(self.transformer)
        
        # Fully-connected module to produce output 3d coords.
        self.mlp_3d = []
        for l in range(n_hl_out):
            self.mlp_3d.extend([nn.Linear(embed_dim, embed_dim),
                                get_activation(activation)])
        self.mlp_3d.append(nn.Linear(embed_dim, 3))
        self.mlp_3d = nn.Sequential(*self.mlp_3d)
        
        
    epsilon = 1e-12
    def forward(self, z, x, get_coords=False):
        """
        z: random noise tensor. Must be of shape (N, nz, L).
        x: amino acid one hot encoding tensor. Must be of shape (N, 20, L).
        get_coords: if 'True' returns a (N, L, 3) tensor storing the
            generated xyz coordinates. If 'False' returns a (N, 1, L, L)
            distance matrix computed from the generated xyz coordinates.
        """
        
        # Positional features.
        if z.shape[2] != x.shape[2]:
            raise ValueError("Incopatible sizes for z and x tensors: z=%s, x=%s" % (
                             repr(z.shape[2]), repr(x.shape[2])))
        prot_l = z.shape[2]
        p = torch.arange(0, prot_l, device=z.device)
        p = p[None,:] - p[:,None]
        bins = torch.linspace(-self.pe_max_l, self.pe_max_l,
                              self.pe_max_l*2+1, device=z.device)
        b = torch.argmin(
              torch.abs(bins.view(1, 1, -1) - p.view(p.shape[0], p.shape[1], 1)),
              axis=-1)
        p = self.embed_pos(b)
        p = p.repeat(z.shape[0], 1, 1, 1)
        h = z
        
        # Embed input. Input is (N, E, L). Reshape to (L, N, E).
        h = torch.transpose(h, 2, 1)
        h = torch.transpose(h, 1, 0)
        
        # Embed input features and z.
        e_aa = self.embed_aa(torch.argmax(x, axis=1))
        e_aa = torch.transpose(e_aa, 1, 0)
        
        embed_x_in = h
        h = self.embed_x(embed_x_in)
            
        # Run through the transformer.
        for l, t_l in enumerate(self.transformer):
            # Input the embeddings at all layers.
            if self.use_embed_repeat:
                h = t_l(h, x=e_aa, p=p)
            # Only use the embeddings in the first layer.
            else:
                if l == 0:
                    h = t_l(h, x=e_aa, p=p)
                else:
                    h = t_l(h, x=None, p=None)
        
        # Produce 3d coords.
        r = self.mlp_3d(h)
        # Shape is now (L, N, 3). Reshape to (N, L, 3).
        r = torch.transpose(r, 1, 0)
        
        # Just return the xyz coordinates (N, L, 3).
        if get_coords:
            return r
        # Return the distance matrix (N, 1, L, L).
        else:
            d = self.get_dmap(r)
            d = d.squeeze(1)
            return d
    
    def get_dmap(self, r, epsilon=None):
        epsilon = epsilon if epsilon is not None else self.epsilon
        d = torch.sqrt(torch.square(r[:,None,:,:] - r[:,:,None,:]).sum(axis=3) + epsilon)
        d = d.unsqueeze(1)
        return d
    
    
    out_feature = "xyz"
    def predict_idp(self, n_samples, aa_seq, device="cpu", batch_size=2048):
        """
        Generates a conformational ensemble for a protein with amino acid
        sequence 'aa_seq'.
        n_samples: number of samples in the ensemble to generate.
        aa_seq: string containing the amino acid sequence of the protein.
        device: pytorch device where to store intermediary data.
        batch_size: batch size used to generated all the conformation in
            the ensemble.
        """
        # Prepare the latent tensor with shape (N, nz, L).
        z = torch.randn(n_samples, self.nz, len(aa_seq), device=device)
        # Prepare the amino acid sequence information tensor with shape
        # (N, 20, L).
        a = torch.tensor(get_features_from_seq(aa_seq), dtype=torch.long,
                         device=device)
        a = a.reshape(1, a.shape[0], a.shape[1]).repeat(z.shape[0], 1, 1)
        # Actually run the generator network to genenerate 
        with torch.no_grad():
            r = []
            for k in range(0, n_samples, batch_size):
                z_k = z[k:k+batch_size]
                a_k = a[k:k+batch_size]
                r_k = self.forward(z_k, a_k, get_coords=True)
                r.append(r_k)
            r = torch.cat(r, axis=0)
        return r


aa_list = list("QWERTYIPASDFGHKLCVNM")
def get_features_from_seq(seq):
    """
    Given an amino acid sequence string 'seq', returns a numpy
    array of shape (20, L), where L is the sequence length.
    """
    for aa_i in seq:
        if aa_i not in aa_list:
            raise ValueError(
                    "Invalid amino acid character in sequence: %s" % aa_i)
    n_res = len(seq)
    n_features = len(aa_list)
    features = np.zeros((n_features, n_res))
    # Residues one hot encoding.
    for i, aa_i in enumerate(seq):
        features[aa_list.index(aa_i), i] = 1
    return features


def load_netg_article(model_fp=None, device="cpu"):
    """
    Loads an instance of the idpGAN generator as presented in the idpGAN article.
    Use the 'model_fp' argument to provide the saved pytorch weights.
    """
    netg = IdpGANGenerator(nz=16, embed_dim=64, d_model=128, nhead=8,
                       dim_feedforward=128, dropout=0.0, num_layers=8,
                       layer_norm_eps=1e-05, dp_attn_norm="d_model",
                       n_hl_out=1, n_hl_embed=1,
                       activation="lrelu",
                       use_embed_repeat=True, embed_dim_1d=32,
                       project_embed_1d=False,
                       pos_embed="af2_2d_ib", pos_embed_dim=64,
                       use_bias_2d=True, pos_embed_max_l=24)
    if model_fp is not None:
        netg.load_state_dict(torch.load(model_fp))
    netg = netg.to(device)
    return netg