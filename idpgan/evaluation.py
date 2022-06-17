import numpy as np


def score_mse_d(admap_ref, admap_hat):
    """
    admap_ref: reference average distance matrix of shape (L, L).
    admap_hat: distance matrix of the same shape used to approximate
        'admap_ref'.
    """
    triu_ids = np.triu_indices(admap_ref.shape[0])
    vals_ref = admap_ref[triu_ids[0],triu_ids[1]]
    vals_hat = admap_hat[triu_ids[0],triu_ids[1]]
    return np.mean(np.square(vals_ref-vals_hat))

def score_mse_c(cmap_ref, cmap_hat):
    """
    cmap_ref: reference contact matrix of shape (L, L).
    cmap_hat: contact matrix of the same shape used to approximate
        'cmap_ref'.
    """
    triu_ids = np.triu_indices(cmap_ref.shape[0])
    vals_ref = np.log(cmap_ref[triu_ids[0],triu_ids[1]])
    vals_hat = np.log(cmap_hat[triu_ids[0],triu_ids[1]])
    return np.mean(np.square(vals_ref-vals_hat))


def score_akld_d(dmap_true, dmap_pred, get_mean=True, *args, **kwargs):
    """
    dmap_true: reference distance matrix of shape (N, L, L).
    dmap_pred: predicted distance matrix of shape (M, L, L).
    """
    kl_l = []
    for i in range(dmap_true.shape[2]):
        for j in range(dmap_pred.shape[2]):
            if i >= j:
                continue
            yt = dmap_true[:,i,j]
            yp = dmap_pred[:,i,j]
            kl_i, _ = score_kl_approximation(yt, yp, *args, **kwargs)
            kl_l.append(kl_i)
    if get_mean:
        return np.mean(kl_l)
    else:
        return np.array(kl_l)

def score_kl_approximation(v_true, v_pred, n_bins=50, pseudo_c=0.001):
    """
    Scores an approximation of KLD by discretizing the values in
    'v_true' (data points from a reference distribution) and 'v_pred'
    (data points from a predicted distribution).
    """
    # Define bins.
    _min = min((v_true.min(), v_pred.min()))
    _max = max((v_true.max(), v_pred.max()))
    bins = np.linspace(_min, _max, n_bins+1)
    # Compute the frequencies in the bins.
    ht = (np.histogram(v_true, bins=bins)[0]+pseudo_c)/v_true.shape[0]
    hp = (np.histogram(v_pred, bins=bins)[0]+pseudo_c)/v_pred.shape[0]
    kl = -np.sum(ht*np.log(hp/ht))
    return kl, bins