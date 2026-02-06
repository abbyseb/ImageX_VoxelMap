import torch
import torch.nn.functional as F
import pystrum.pynd.ndutils as nd
import numpy as np
import math


class image:
    """
    Computes the MSE between a predicted and ground-truth image
    """

    def loss(self, target_vol, predict_vol):
        error = target_vol - predict_vol
        return torch.mean(error ** 2)


class image_mask:
    """
    Computes the MSE between a predicted and ground-truth image inside a binary mask
    """

    def loss(self, target_vol, predict_vol, mask):
        error = target_vol - predict_vol
        error[mask == 0] = 0
        return torch.sum(error ** 2) / torch.count_nonzero(error)


class flow:
    """
    Computes the MSE between a predicted and ground-truth DVF
    """

    def loss(self, target_flow, predict_flow):
        error = target_flow - predict_flow
        return torch.mean(error ** 2)


class flow_mask:
    """
    Computes the MSE between a predicted and ground-truth DVF inside a binary mask
    """

    def loss(self, target_flow, predict_flow, mask):
        mask = torch.cat((mask, mask, mask), 1)
        error = target_flow - predict_flow
        error[mask == 0] = 0
        return torch.sum(error ** 2) / torch.count_nonzero(error)


class flow_ptv:
    """
    Computes the mean 3D flows inside a PTV mask
    """

    def loss(self, flow, mask):
        mask = mask[:, 0, :, :, :]
        lr = flow[:, 0, :, :, :]
        si = flow[:, 1, :, :, :]
        ap = flow[:, 2, :, :, :]

        lr[mask == 0] = 0
        si[mask == 0] = 0
        ap[mask == 0] = 0

        lr = torch.sum(lr) / torch.count_nonzero(lr)
        si = torch.sum(si) / torch.count_nonzero(si)
        ap = torch.sum(ap) / torch.count_nonzero(ap)
        return lr, si, ap


class centroid_error:
    """
    Computes the lr,si,ap error between the centroids of two masks
    """

    def loss(self, target, pred):
        ind = np.nonzero(target)
        lr_tar = int(np.mean(ind[0]))
        si_tar = int(np.mean(ind[1]))
        ap_tar = int(np.mean(ind[2]))

        ind = np.nonzero(pred)
        lr_pre = int(np.mean(ind[0]))
        si_pre = int(np.mean(ind[1]))
        ap_pre = int(np.mean(ind[2]))

        lr = lr_tar - lr_pre
        si = si_tar - si_pre
        ap = ap_tar - ap_pre
        return lr, si, ap


# class centroid_ptv:
#     """
#     Computes the lr,si,ap position of a binary mask
#     """
#
#     def loss(self, mask):
#         metric_input = mask.cpu().detach().numpy()
#         mask = np.asarray(metric_input[0][:][:][:][0], dtype=np.float32)
#         ind = np.nonzero(mask)
#
#         lr = int(np.mean(ind[0]))
#         si = int(np.mean(ind[1]))
#         ap = int(np.mean(ind[2]))
#
#         return lr, si, ap

class centroid_ptv:
    """
    Computes LR, SI, AP centroid of a (binary or soft) 3D mask.

    - Returns FLOAT voxel coordinates (sub-voxel).
    - Works on torch tensors directly (no numpy round-trips).
    - Handles empty masks safely (returns NaNs by default).
    """

    def __init__(self, threshold: float = 0.0, empty_value=float("nan")):
        """
        Args:
            threshold: voxels > threshold are treated as 'in mask' for binary centroid.
                       If your mask is soft/probabilistic, keep threshold=0 and it will
                       behave like binary unless you change it.
            empty_value: value returned for lr/si/ap if mask is empty.
        """
        self.threshold = threshold
        self.empty_value = empty_value

    @torch.no_grad()
    def loss(self, mask: torch.Tensor):
        """
        Args:
            mask: tensor shaped (B, 1, D, H, W) or (1, D, H, W) or (D, H, W)
                  Values can be {0,1} or soft.

        Returns:
            (lr, si, ap) as python floats in voxel coordinates.
            NOTE: These correspond to indices along (D, H, W) respectively.
        """
        # Normalize shape to (D,H,W)
        if mask.dim() == 5:
            m = mask[0, 0]
        elif mask.dim() == 4:
            m = mask[0]
        elif mask.dim() == 3:
            m = mask
        else:
            raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}")

        # Binary support for centroid (keeps behaviour consistent with your previous code)
        m_bin = (m > self.threshold)

        # If empty, return NaNs (or configured value)
        if not torch.any(m_bin):
            v = float(self.empty_value)
            return v, v, v

        # Get coordinates of non-zero voxels: (N,3) with columns [D,H,W]
        coords = m_bin.nonzero(as_tuple=False).float()

        # Mean coordinate = centroid in voxel units (sub-voxel)
        lr = coords[:, 0].mean().item()
        si = coords[:, 1].mean().item()
        ap = coords[:, 2].mean().item()

        return lr, si, ap


class l2:
    """
    Computes the squared L2-norm of a predicted DVF
    """

    def loss(self, predict_flow):
        return torch.mean(predict_flow ** 2)


class l2_mask:
    """
    Computes the squared L2-norm of a predicted DVF inside a binary mask
    """

    def loss(self, predict_flow, mask):
        mask = torch.cat((mask, mask, mask), 1)
        predict_flow[mask == 0] = 0
        return torch.sum(predict_flow ** 2) / torch.count_nonzero(mask)


class grad:
    """
    Simplified gradient loss
    """

    def loss(self, predict_flow):
        dy = torch.abs(predict_flow[:, :, 1:, :, :] - predict_flow[:, :, :-1, :, :])
        dx = torch.abs(predict_flow[:, :, :, 1:, :] - predict_flow[:, :, :, :-1, :])
        dz = torch.abs(predict_flow[:, :, :, :, 1:] - predict_flow[:, :, :, :, :-1])
        d = torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)
        return d / 3


class grad_mask:
    """
    Simplified gradient loss inside a binary mask
    """

    def loss(self, predict_flow, mask):
        mask = torch.cat((mask, mask, mask), 1)
        predict_flow[mask == 0] = 0

        dy = predict_flow[:, :, 1:, :, :] - predict_flow[:, :, :-1, :, :]
        dx = predict_flow[:, :, :, 1:, :] - predict_flow[:, :, :, :-1, :]
        dz = predict_flow[:, :, :, :, 1:] - predict_flow[:, :, :, :, :-1]
        d = torch.sum(dx ** 2) + torch.sum(dy ** 2) + torch.sum(dz ** 2)
        return d / (3 * torch.count_nonzero(mask))


class dist3d:
    """
    Mean 3D error between a predicted and ground-truth DVF
    """

    def loss(self, target_flow, predict_flow):
        dx = target_flow[:, 0, :, :, :] - predict_flow[:, 0, :, :, :]
        dy = target_flow[:, 1, :, :, :] - predict_flow[:, 1, :, :, :]
        dz = target_flow[:, 2, :, :, :] - predict_flow[:, 2, :, :, :]
        return torch.mean(torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2))


class dist3d_mask:
    """
    Mean 3D error between a predicted and ground-truth DVF inside a binary mask
    """

    def loss(self, target_flow, predict_flow, mask):
        mask = torch.cat((mask, mask, mask), 1)
        target_flow[mask == 0] = 0
        predict_flow[mask == 0] = 0

        dx = target_flow[:, 0, :, :, :] - predict_flow[:, 0, :, :, :]
        dy = target_flow[:, 1, :, :, :] - predict_flow[:, 1, :, :, :]
        dz = target_flow[:, 2, :, :, :] - predict_flow[:, 2, :, :, :]
        return torch.sum(torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)) / torch.count_nonzero(mask)


class BindingEnergy:
    """
    3D binding energy loss
    """

    def loss(self, flow):
        # compute derivatives
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dx2 = torch.abs(dx[:, :, :, 1:, :] - dx[:, :, :, :-1, :])
        dxdy = torch.abs(dx[:, :, 1:, :, :] - dx[:, :, :-1, :, :])

        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dy2 = torch.abs(dy[:, :, 1:, :, :] - dy[:, :, :-1, :, :])
        dydz = torch.abs(dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1])

        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        dz2 = torch.abs(dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1])
        dxdz = torch.abs(dx[:, :, :, :, 1:] - dx[:, :, :, :, :-1])

        # reshape tensors
        dx2 = dx2[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]
        dxdy = dxdy[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]

        dy2 = dy2[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]
        dydz = dydz[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]

        dz2 = dz2[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]
        dxdz = dxdz[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]

        # sum values
        loss = torch.mean(dx2 * dx2)
        loss += torch.mean(dy2 * dy2)
        loss += torch.mean(dz2 * dz2)
        loss += 2 * torch.mean(dxdy * dxdy)
        loss += 2 * torch.mean(dydz * dydz)
        loss += 2 * torch.mean(dxdz * dxdz)
        return loss


class dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return dice


class jacobian_determinant:
    """
    jacobian determinant of a displacement field.
    """

    def loss(self, disp):

        # check inputs
        volshape = disp.shape[:-1]
        nb_dims = len(volshape)
        assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

        # compute grid
        grid_lst = nd.volsize2ndgrid(volshape)
        grid = np.stack(grid_lst, len(volshape))

        # compute gradients
        J = np.gradient(disp + grid)

        # 3D glow
        if nb_dims == 3:
            dx = J[0]
            dy = J[1]
            dz = J[2]

            # compute jacobian components
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
            Jdet = Jdet0 - Jdet1 + Jdet2

            # return the proportion of element for which Jdet <= 0 #sum(i <= 0 for i in Jdet.flatten()) / Jdet.size
            return Jdet

        else:  # must be 2

            dfdx = J[0]
            dfdy = J[1]
            Jdet = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

            return Jdet


class image_l1:
    """
    Computes the L1 distance between a predicted and ground-truth image/volume.
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def loss(self, target_vol, predict_vol):
        error = torch.abs(target_vol - predict_vol)
        if self.reduction == "sum":
            return torch.sum(error)
        if self.reduction == "none":
            return error
        return torch.mean(error)


class motion_image_joint_loss:
    """
    Combines supervised motion loss (L2 on flow) with auxiliary image L1 loss.
    Returns total loss along with individual components for logging.
    """

    def __init__(self, lambda_image: float = 0.1, motion_loss_fn=None, image_loss_fn=None):
        self.lambda_image = lambda_image
        self._motion_loss_fn = motion_loss_fn or flow()
        self._image_loss_fn = image_loss_fn or image_l1()

    def loss(self, target_flow, predict_flow, target_volume, predict_volume, mask=None):
        if mask is not None:
            try:
                motion_loss = self._motion_loss_fn.loss(target_flow, predict_flow, mask)
            except TypeError:
                motion_loss = self._motion_loss_fn.loss(target_flow, predict_flow)
        else:
            motion_loss = self._motion_loss_fn.loss(target_flow, predict_flow)
        image_loss = self._image_loss_fn.loss(target_volume, predict_volume)
        total_loss = motion_loss + (self.lambda_image * image_loss)
        return total_loss, motion_loss, image_loss


import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self, layer_index=8, slices_per_vol=16, device='cuda', sampling_mode='random'):
        """
        Robust Feature Reconstruction Loss for 3D Volumes.

        Args:
            layer_index (int): Layer to extract features from (8 is relu2_2).
            slices_per_vol (int): Number of slices to sample per volume to save memory.
            device (str): 'cuda' or 'cpu'
            sampling_mode (str): 'random' for stochastic slices, 'even' for deterministic coverage.
        """
        super().__init__()
        # Load VGG16
        vgg = models.vgg16(pretrained=True)
        self.loss_network = nn.Sequential(*list(vgg.features)[:layer_index + 1]).eval()

        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.loss_network.to(device)
        self.slices_per_vol = slices_per_vol
        self.sampling_mode = sampling_mode

        # ImageNet Normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def normalize_vgg_input(self, x):
        """
        Assumes input x is in range [-1, 1] (from Tanh).
        1. Scales to [0, 1].
        2. Clamps to ensure numerical stability.
        3. Normalizes with ImageNet mean/std.
        """
        # [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0, 1)

        # Normalize
        return (x - self.mean) / self.std

    def forward(self, generated_vol, target_vol):
        """
        Args:
            generated_vol: (B, 1, D, H, W) range [-1, 1]
            target_vol: (B, 1, D, H, W) range [-1, 1]
        """
        b, c, d, h, w = generated_vol.shape

        # --- Slice Sampling ---
        # Instead of reshaping all D slices, we pick a subset.
        if self.slices_per_vol < d:
            if self.sampling_mode == 'even':
                indices = torch.linspace(
                    0, d - 1, steps=self.slices_per_vol, device=generated_vol.device
                ).long()
            else:
                indices = torch.randperm(d, device=generated_vol.device)[: self.slices_per_vol]

            gen_sampled = generated_vol.index_select(2, indices)
            tgt_sampled = target_vol.index_select(2, indices)
        else:
            gen_sampled = generated_vol
            tgt_sampled = target_vol

        # Reshape to (Batch * Slices, 1, H, W)
        gen_2d = gen_sampled.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        tgt_2d = tgt_sampled.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

        # Repeat channels: (B*S, 1, H, W) -> (B*S, 3, H, W)
        gen_2d = gen_2d.repeat(1, 3, 1, 1)
        tgt_2d = tgt_2d.repeat(1, 3, 1, 1)

        # Normalize for VGG
        gen_2d = self.normalize_vgg_input(gen_2d)
        tgt_2d = self.normalize_vgg_input(tgt_2d)

        # Extract features
        with torch.no_grad():
            target_features = self.loss_network(tgt_2d)

        gen_features = self.loss_network(gen_2d)

        # Feature Loss (MSE)
        return F.mse_loss(gen_features, target_features)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLossMask(nn.Module):
    """
    Masked Perceptual Loss for 3D Volumes.

    Computes perceptual loss only within the masked region, similar to how
    flow_mask works for motion loss.
    """

    def __init__(self, layer_index=8, slices_per_vol=16, device='cuda', sampling_mode='random'):
        """
        Args:
            layer_index (int): Layer to extract features from (8 is relu2_2).
            slices_per_vol (int): Number of slices to sample per volume to save memory.
            device (str): 'cuda' or 'cpu'
            sampling_mode (str): 'random' for stochastic slices, 'even' for deterministic coverage.
        """
        super().__init__()
        # Load VGG16
        vgg = models.vgg16(pretrained=True)
        self.loss_network = nn.Sequential(*list(vgg.features)[:layer_index + 1]).eval()

        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.loss_network.to(device)
        self.slices_per_vol = slices_per_vol
        self.sampling_mode = sampling_mode
        self.device = device

        # ImageNet Normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def normalize_vgg_input(self, x):
        """
        Assumes input x is in range [-1, 1] (from Tanh).
        1. Scales to [0, 1].
        2. Clamps to ensure numerical stability.
        3. Normalizes with ImageNet mean/std.
        """
        # [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0, 1)

        # Normalize
        return (x - self.mean) / self.std

    def forward(self, generated_vol, target_vol, mask):
        """
        Args:
            generated_vol: (B, 1, D, H, W) range [-1, 1]
            target_vol: (B, 1, D, H, W) range [-1, 1]
            mask: (B, 1, D, H, W) binary mask (1 = inside region of interest, 0 = outside)

        Returns:
            Perceptual loss computed only within the masked region
        """
        b, c, d, h, w = generated_vol.shape

        # --- Slice Sampling ---
        if self.slices_per_vol < d:
            if self.sampling_mode == 'even':
                indices = torch.linspace(
                    0, d - 1, steps=self.slices_per_vol, device=generated_vol.device
                ).long()
            else:
                indices = torch.randperm(d, device=generated_vol.device)[: self.slices_per_vol]

            gen_sampled = generated_vol.index_select(2, indices)
            tgt_sampled = target_vol.index_select(2, indices)
            mask_sampled = mask.index_select(2, indices)
        else:
            gen_sampled = generated_vol
            tgt_sampled = target_vol
            mask_sampled = mask

        # Reshape to (Batch * Slices, 1, H, W)
        gen_2d = gen_sampled.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        tgt_2d = tgt_sampled.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        mask_2d = mask_sampled.permute(0, 2, 1, 3, 4).reshape(-1, 1, h, w)

        # Repeat channels: (B*S, 1, H, W) -> (B*S, 3, H, W)
        gen_2d = gen_2d.repeat(1, 3, 1, 1)
        tgt_2d = tgt_2d.repeat(1, 3, 1, 1)

        # Normalize for VGG
        gen_2d = self.normalize_vgg_input(gen_2d)
        tgt_2d = self.normalize_vgg_input(tgt_2d)

        # Extract features
        with torch.no_grad():
            target_features = self.loss_network(tgt_2d)

        gen_features = self.loss_network(gen_2d)

        # --- Apply Mask to Features ---
        # The mask needs to be downsampled to match the feature map size
        # VGG features at layer 8 (relu2_2) are typically downsampled by 2x
        feat_h, feat_w = gen_features.shape[2], gen_features.shape[3]

        # Downsample mask to feature resolution
        mask_features = F.interpolate(
            mask_2d,
            size=(feat_h, feat_w),
            mode='bilinear',
            align_corners=True
        )

        # Binarize after interpolation to keep it as a mask
        mask_features = (mask_features > 0.5).float()

        # Apply mask to features (zero out regions outside the mask)
        gen_features_masked = gen_features * mask_features
        target_features_masked = target_features * mask_features

        # Compute MSE only on non-zero (masked) elements
        error = (gen_features_masked - target_features_masked) ** 2

        # Sum over all dimensions and normalize by number of valid elements
        # Count valid elements per channel
        num_valid = torch.count_nonzero(mask_features)

        if num_valid == 0:
            # If mask is empty, return zero loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Normalize by number of valid spatial locations and channels
        loss = torch.sum(error) / (num_valid * gen_features.shape[1])

        return loss


class image_l1_mask:
    """
    L1 loss within a binary mask (for comparison with perceptual loss).
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def loss(self, target_vol, predict_vol, mask):
        """
        Args:
            target_vol: (B, 1, D, H, W)
            predict_vol: (B, 1, D, H, W)
            mask: (B, 1, D, H, W) binary mask
        """
        error = torch.abs(target_vol - predict_vol)
        error[mask == 0] = 0

        num_valid = torch.count_nonzero(mask)
        if num_valid == 0:
            return torch.tensor(0.0, device=target_vol.device)

        return torch.sum(error) / num_valid