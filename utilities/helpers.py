import numpy as np
import torch

def to_float(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def centroid_shift_mm(centroid, src_ptv, tgt_ptv, spacing=(3.5, 1.7, 3.5)):
    """
    Computes displacement (mm) between source and target PTV centroids
    """
    lr_s, si_s, ap_s = centroid.loss(src_ptv)
    lr_t, si_t, ap_t = centroid.loss(tgt_ptv)

    d_lr = (lr_t - lr_s) * spacing[0]
    d_si = (si_t - si_s) * spacing[1]
    d_ap = (ap_t - ap_s) * spacing[2]

    return d_lr, d_si, d_ap
