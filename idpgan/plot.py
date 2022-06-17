import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm


def plot_average_dmap_comparison(dmap_ref, dmap_gen, title,
                                 ticks_fontsize=14,
                                 cbar_fontsize=14,
                                 title_fontsize=14,
                                 dpi=96,
                                 max_d=6.8,
                                 use_ylabel=True):
    plt.figure(dpi=dpi)
    if dmap_ref is not None:
        _dmap_ref = dmap_ref.mean(axis=0)
    else:
        _dmap_ref = np.zeros(dmap_gen.shape[1:])
    _dmap_gen = dmap_gen.mean(axis=0)
    tril_ids = np.tril_indices(dmap_gen.shape[2], 0)
    _dmap_ref[tril_ids[0],tril_ids[1]] = 0
    _dmap_gen[tril_ids[0],tril_ids[1]] = 0
    dmap_sel = _dmap_ref.T + _dmap_gen
    diag_ids = np.diag_indices(dmap_sel.shape[0])
    dmap_sel[diag_ids] = np.nan
    if dmap_ref is None:
        dmap_sel[tril_ids[0],tril_ids[1]] = np.nan
    plt.imshow(dmap_sel)
    plt.xticks(fontsize=ticks_fontsize)
    if use_ylabel:
        plt.yticks(fontsize=ticks_fontsize)
    else:
        plt.yticks([])
    plt.title(title, fontsize=title_fontsize)
    cbar = plt.colorbar()
    cbar.set_label(r"Average $d_{ij}$ [nm]")
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    plt.clim(0, max_d)
    plt.show()
    
    
def plot_cmap_comparison(cmap_ref, cmap_gen, title,
                         ticks_fontsize=14,
                         cbar_fontsize=14,
                         title_fontsize=14,
                         dpi=96, cmap_min=-3.5,
                         use_ylabel=True):
    plt.figure(dpi=dpi)
    cmap = cm.get_cmap("jet")
    norm = colors.Normalize(cmap_min, 0)
    if cmap_ref is not None:
        _cmap_ref = np.log10(cmap_ref.copy())
    else:
        _cmap_ref = np.zeros(cmap_gen.shape)
    _cmap_gen = np.log10(cmap_gen.copy())
    tril_ids = np.tril_indices(cmap_gen.shape[0], 0)
    _cmap_ref[tril_ids[0],tril_ids[1]] = 0
    _cmap_gen[tril_ids[0],tril_ids[1]] = 0
    cmap_sel = _cmap_ref.T + _cmap_gen
    diag_ids = np.diag_indices(cmap_sel.shape[0])
    cmap_sel[diag_ids] = np.nan
    if cmap_ref is None:
        cmap_sel[tril_ids[0],tril_ids[1]] = np.nan
    plt.imshow(cmap_sel, cmap=cmap, norm=norm)
    plt.xticks(fontsize=ticks_fontsize)
    if use_ylabel:
        plt.yticks(fontsize=ticks_fontsize)
    else:
        plt.yticks([])
    plt.title(title, fontsize=title_fontsize)
    cbar = plt.colorbar()
    cbar.set_label(r'$log_{10}(p_{ij})$')
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    plt.clim(cmap_min, 0)
    plt.show()
    
    
def plot_distances_comparison(prot_data, polyala_data, n_residues,
                              dpi=96, n_hist=5):

    all_pairs = np.array([(i,j) for i in range(n_residues) \
                          for j in range(n_residues) if i < j])
    sel_pairs = all_pairs[np.random.choice(all_pairs.shape[0], n_hist,
                                           replace=False)]

    h_args = {"histtype": "step", "density": True}
    protein_count = 0
    bins_list = []
    _polyala_data = (("polyala", polyala_data[0], polyala_data[0]), )
    for protein_s, dmap_md_s, dmap_gen_s in prot_data + _polyala_data:
        fig, ax = plt.subplots(1, n_hist, figsize=(2.5*n_hist, 3), dpi=dpi)
        hist_l = []
        labels_l = []
        for k in range(n_hist):
            i, j = sel_pairs[k]
            # MD data.
            h_0k = ax[k].hist(dmap_md_s[:,i,j], bins=50, **h_args)
            bins_ij = h_0k[1]
            if protein_count == 0:
                bins_list.append(bins_ij)
            else:
                bins_ij = bins_list[k]
            if k == 0:
                hist_l.append(h_0k[2][0])
                labels_l.append("MD")
            # Generated data.
            h_1k = ax[k].hist(dmap_gen_s[:,i,j], bins=bins_ij, **h_args)
            if k == 0:
                hist_l.append(h_1k[2][0])
                labels_l.append("GEN")
            # Polyala data.
            if protein_s != "polyala":
                h_2k = ax[k].hist(polyala_data[0][:,i,j],
                                  bins=bins_ij, alpha=0.6,
                                  **h_args)
                if k == 0:
                    hist_l.append(h_2k[2][0])
                    labels_l.append("ALA MD")
            ax[k].set_title("residues %s-%s" % (i, j))
            if k == 0:
                ax[k].set_ylabel("density")
            ax[k].set_xlabel("distance [nm]")
        protein_count += 1
        fig.legend(hist_l, labels_l, "upper right", ncol=3)
        plt.suptitle(protein_s, fontweight='bold')
        plt.tight_layout()
        plt.show()


def plot_rg_comparison(prot_data, polyala_data, n_bins=50,
                       bins_range=(1, 4.5), dpi=96):
    h_args = {"histtype": "step", "density": True}
    n_systems = len(prot_data)+1
    bins = np.linspace(bins_range[0], bins_range[1], n_bins+1)
    fig, ax = plt.subplots(1, n_systems, figsize=(3*n_systems, 3), dpi=dpi)
    for i, (name_i, rg_md_i, rg_gen_i) in enumerate(prot_data):
        ax[i].hist(rg_md_i, bins=bins, label="MD", **h_args)
        ax[i].hist(rg_gen_i, bins=bins, label="GEN", **h_args)
        ax[i].hist(polyala_data[0], bins=bins, alpha=0.6, label="ALA MD",
                   **h_args)
        ax[i].legend()
        ax[i].set_title(name_i)
        if i == 0:
            ax[i].set_ylabel("density")
        ax[i].set_xlabel("Rg [nm]")
    ax[-1].hist(polyala_data[0], bins=bins, label="MD", **h_args)
    ax[-1].hist(polyala_data[1], bins=bins, label="GEN", **h_args)
    ax[-1].legend()
    ax[-1].set_title("polyala")
    ax[-1].set_xlabel("Rg [nm]")
    plt.tight_layout()
    plt.show()
    
    
def plot_rg_distribution(rg_vals, title, n_bins=50, dpi=96):
    plt.figure(dpi=dpi)
    plt.hist(rg_vals, bins=n_bins, histtype="step", density=True)
    plt.xlabel("Rg [nm]")
    plt.ylabel("density")
    plt.show()
    
    
def plot_dmap_snapshots(dmap, title, n_snapshots, dpi=96):
    fig, ax = plt.subplots(1, n_snapshots,
                           figsize=(3*n_snapshots, 3), dpi=dpi,
                           constrained_layout=True)
    random_ids = np.random.choice(dmap.shape[0], n_snapshots,
                                  replace=dmap.shape[0] < n_snapshots)
    for i in range(n_snapshots):
        _dmap_i = dmap[random_ids[i]]
        im_i = ax[i].imshow(_dmap_i)
        ax[i].set_title("snapshot %s" % random_ids[i])
    cbar = fig.colorbar(im_i, ax=ax, location='right', shrink=0.8)
    cbar.set_label(r"Average $d_{ij}$ [nm]")
    plt.show()