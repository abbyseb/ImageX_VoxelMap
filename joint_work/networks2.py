import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities import layers
from utilities.modelio import LoadableModel, store_config_args


class ProjectionEmbedder(nn.Module):
    """Embeds 2D projection(s) into feature space compatible with 3D volumes."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        out = self.activation(conv1 + conv2)
        return out


class Encoder3D(nn.Module):
    """3D Encoder module with down-sampling"""

    def __init__(self, in_channels, enc_nf):
        super().__init__()
        self.enc_nf = enc_nf
        self.downarm = nn.ModuleList()

        prev_nf = in_channels
        for nf in enc_nf:
            self.downarm.append(DownBlock3d(prev_nf, nf))
            prev_nf = nf

    def forward(self, x):
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        return x_enc


class Decoder3D(nn.Module):
    """3D Decoder module with up-sampling and optional skip connections"""

    def __init__(self, enc_nf, out_channels=3, use_skip_connections=False):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        dec_nf = enc_nf[::-1]
        dec_nf.append(out_channels)

        self.uparm = nn.ModuleList()
        prev_nf = enc_nf[-1]
        for i, nf in enumerate(dec_nf[:len(enc_nf)]):
            if use_skip_connections and i > 0:
                # Add skip connection from corresponding encoder level
                skip_nf = enc_nf[-(i + 1)]
                self.uparm.append(UpBlock(prev_nf + skip_nf, nf))
            else:
                self.uparm.append(UpBlock(prev_nf, nf))
            prev_nf = nf

        self.extras = nn.ModuleList()
        for nf in dec_nf[len(enc_nf):]:
            self.extras.append(ExtraBlock(prev_nf, nf))
            prev_nf = nf

    def forward(self, x_enc):
        x = x_enc[-1]
        decoder_features = []

        for i, layer in enumerate(self.uparm):
            if self.use_skip_connections and i > 0:
                # Concatenate skip connection from encoder
                skip_idx = -(i + 1)
                x = torch.cat([x, x_enc[skip_idx]], dim=1)
            x = layer(x)
            decoder_features.append(x)

        for layer in self.extras:
            x = layer(x)

        return x, decoder_features


# ============================================================================
# VARIANT 1: Single Encoder, Dual Decoders with Feature Concatenation
# ============================================================================

class SingleEncoderDualDecoder(LoadableModel):
    """
    Single encoder with separate motion and image decoders.
    Supports either staged training (motion/image modes) or fully joint
    training where both decoders run in a single forward pass.
    """

    @store_config_args
    def __init__(self, im_size, int_steps=7, skip_connections=False, proj_in_channels=2):
        super().__init__()

        self.im_size = im_size
        self.skip_connections = skip_connections
        self.proj_in_channels = proj_in_channels

        # Build feature dimensions
        enc_nf = [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]
        self.enc_nf = enc_nf

        # Projection embedder
        self.proj_embedder = ProjectionEmbedder(proj_in_channels, enc_nf[0])

        # Single shared encoder
        self.encoder = Encoder3D(enc_nf[0] + 1, enc_nf)

        # Motion decoder (no skip connections in motion decoder)
        self.motion_decoder = Decoder3D(enc_nf, out_channels=3, use_skip_connections=False)

        # Image decoder with adjusted input channels for concatenation
        if skip_connections:
            # Image decoder receives concatenated features from encoder and motion decoder
            self.image_decoder_uparm = nn.ModuleList()
            dec_nf = enc_nf[::-1]

            prev_nf = enc_nf[-1]
            for i in range(len(enc_nf)):
                current_dec_nf = dec_nf[i]

                if i > 0:
                    # Concatenate: upsampled + encoder skip + motion decoder feature
                    skip_nf = enc_nf[-(i + 1)]
                    motion_nf = dec_nf[i - 1]  # FIX: motion_decoder_features[i-1] has dec_nf[i-1] channels
                    self.image_decoder_uparm.append(UpBlock(prev_nf + skip_nf + motion_nf, current_dec_nf))
                else:
                    self.image_decoder_uparm.append(UpBlock(prev_nf, current_dec_nf))

                prev_nf = current_dec_nf

            self.image_decoder_extras = nn.ModuleList()
            self.image_decoder_extras.append(ExtraBlock(prev_nf, 1))
        else:
            # Without skip connections: just concatenate motion decoder features
            self.image_decoder_uparm = nn.ModuleList()
            dec_nf = enc_nf[::-1]

            prev_nf = enc_nf[-1]
            for i in range(len(enc_nf)):
                current_dec_nf = dec_nf[i]

                if i > 0:
                    motion_nf = dec_nf[i - 1]  # FIX: motion_decoder_features[i-1] has dec_nf[i-1] channels
                    self.image_decoder_uparm.append(UpBlock(prev_nf + motion_nf, current_dec_nf))
                else:
                    self.image_decoder_uparm.append(UpBlock(prev_nf, current_dec_nf))

                prev_nf = current_dec_nf

            self.image_decoder_extras = nn.ModuleList()
            self.image_decoder_extras.append(ExtraBlock(prev_nf, 1))

        # Flow integrator (single level at full resolution)
        vol_shape = [im_size, im_size, im_size]
        self.integrator = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None

        # Final transformer
        self.final_transformer = layers.SpatialTransformer(vol_shape)

    def _run_image_decoder(self, x_enc, motion_decoder_features, detach_motion_features: bool):
        """
        Runs the image decoder branch, optionally detaching motion features to
        prevent gradients from flowing back through the motion decoder.
        """
        x = x_enc[-1]
        for i, layer in enumerate(self.image_decoder_uparm):
            # Concatenate features BEFORE upsampling (for i > 0)
            if i > 0:
                features_to_concat = [x]

                if self.skip_connections:
                    # Get encoder skip connection
                    skip_idx = -(i + 1)
                    if abs(skip_idx) <= len(x_enc) - 1:  # -1 because x_enc includes input
                        skip_feat = x_enc[skip_idx]
                        # Ensure shapes match
                        if skip_feat.shape[2:] != x.shape[2:]:
                            skip_feat = F.interpolate(
                                skip_feat,
                                size=x.shape[2:],
                                mode='trilinear',
                                align_corners=True,
                            )
                        features_to_concat.append(skip_feat)

                # Get motion decoder feature (motion_decoder_features[i-1] corresponds to current level)
                motion_idx = i - 1
                if motion_idx < len(motion_decoder_features):
                    motion_feat = motion_decoder_features[motion_idx]
                    if detach_motion_features:
                        motion_feat = motion_feat.detach()
                    # Ensure shapes match
                    if motion_feat.shape[2:] != x.shape[2:]:
                        motion_feat = F.interpolate(
                            motion_feat,
                            size=x.shape[2:],
                            mode='trilinear',
                            align_corners=True,
                        )
                    features_to_concat.append(motion_feat)

                # Concatenate if we have additional features
                if len(features_to_concat) > 1:
                    x = torch.cat(features_to_concat, dim=1)

            # Apply layer (upsamples)
            x = layer(x)

        for layer in self.image_decoder_extras:
            x = layer(x)
        return x

    def forward(self, source_proj, target_proj, source_vol, mode='motion'):
        """
        Args:
            mode: 'motion', 'image', or 'joint'
        Returns:
            Depending on the mode:
              - 'motion'/'image': (warped_volume, flow)
              - 'joint': dict with motion/image sub-dicts containing volume+flow
        """
        if target_proj is None:
            raise ValueError("target_proj must be provided for dual-decoder models.")

        # Combine projections and embed
        proj_pair = torch.cat([source_proj, target_proj], dim=1)
        target_feat = self.proj_embedder(proj_pair)
        target_feat = target_feat.unsqueeze(2)
        depth = source_vol.shape[2]
        target_feat = target_feat.expand(-1, -1, depth, -1, -1)

        # Concatenate and encode
        x = torch.cat([target_feat, source_vol], dim=1)
        x_enc = self.encoder(x)

        if mode == 'motion':
            flow, _ = self.motion_decoder(x_enc)
            if self.integrator is not None:
                flow = self.integrator(flow)
            y_source = self.final_transformer(source_vol, flow)
            return y_source, flow

        if mode == 'image':
            with torch.no_grad():
                _, motion_decoder_features = self.motion_decoder(x_enc)
            synth_volume = self._run_image_decoder(
                x_enc,
                motion_decoder_features,
                detach_motion_features=True,
            )
            return synth_volume, None

        if mode == 'joint':
            motion_flow, motion_decoder_features = self.motion_decoder(x_enc)
            if self.integrator is not None:
                motion_flow = self.integrator(motion_flow)
            motion_volume = self.final_transformer(source_vol, motion_flow)

            synth_volume = self._run_image_decoder(
                x_enc,
                motion_decoder_features,
                detach_motion_features=False,
            )

            return {
                'motion': {'volume': motion_volume, 'flow': motion_flow},
                'image': {'volume': synth_volume},
            }

        raise ValueError(f"Unsupported mode '{mode}'.")

##Dual ENCODER DUAL DECODER

class DualEncoderDualDecoder(LoadableModel):
    """Dual encoder / dual decoder architecture with feature fusion."""

    @store_config_args
    def __init__(self, im_size, int_steps=7, skip_connections=False, proj_in_channels=2):
        super().__init__()

        self.im_size = im_size
        self.skip_connections = skip_connections
        self.proj_in_channels = proj_in_channels

        enc_nf = [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]
        self.enc_nf = enc_nf

        # Independent projection embedders and encoders for motion and image branches
        self.motion_proj_embedder = ProjectionEmbedder(proj_in_channels, enc_nf[0])
        self.image_proj_embedder = ProjectionEmbedder(proj_in_channels, enc_nf[0])

        self.motion_encoder = Encoder3D(enc_nf[0] + 1, enc_nf)
        self.image_encoder = Encoder3D(enc_nf[0] + 1, enc_nf)

        self.motion_decoder = Decoder3D(enc_nf, out_channels=3, use_skip_connections=False)

        # Image decoder receives motion decoder features + (optional) image encoder skips
        self.image_decoder_uparm = nn.ModuleList()
        dec_nf = enc_nf[::-1]

        prev_nf = enc_nf[-1]
        for i in range(len(enc_nf)):
            current_dec_nf = dec_nf[i]

            if i > 0:
                in_channels = prev_nf
                if skip_connections:
                    skip_nf = enc_nf[-(i + 1)]
                    in_channels += skip_nf
                motion_nf = dec_nf[i - 1]
                in_channels += motion_nf
                self.image_decoder_uparm.append(UpBlock(in_channels, current_dec_nf))
            else:
                self.image_decoder_uparm.append(UpBlock(prev_nf, current_dec_nf))

            prev_nf = current_dec_nf

        self.image_decoder_extras = nn.ModuleList()
        self.image_decoder_extras.append(ExtraBlock(prev_nf, 1))

        vol_shape = [im_size, im_size, im_size]
        self.integrator = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None
        self.final_transformer = layers.SpatialTransformer(vol_shape)

    def _expand_projection(self, proj_embedder, source_proj, target_proj, depth):
        proj_pair = torch.cat([source_proj, target_proj], dim=1)
        feat = proj_embedder(proj_pair).unsqueeze(2)
        return feat.expand(-1, -1, depth, -1, -1)

    def _run_image_decoder(self, image_x_enc, motion_decoder_features, detach_motion_features: bool):
        x = image_x_enc[-1]

        for i, layer in enumerate(self.image_decoder_uparm):
            if i > 0:
                features_to_concat = [x]

                if self.skip_connections:
                    skip_idx = -(i + 1)
                    if abs(skip_idx) <= len(image_x_enc) - 1:
                        skip_feat = image_x_enc[skip_idx]
                        if skip_feat.shape[2:] != x.shape[2:]:
                            skip_feat = F.interpolate(
                                skip_feat,
                                size=x.shape[2:],
                                mode='trilinear',
                                align_corners=True,
                            )
                        features_to_concat.append(skip_feat)

                motion_idx = i - 1
                if motion_idx < len(motion_decoder_features):
                    motion_feat = motion_decoder_features[motion_idx]
                    if detach_motion_features:
                        motion_feat = motion_feat.detach()
                    if motion_feat.shape[2:] != x.shape[2:]:
                        motion_feat = F.interpolate(
                            motion_feat,
                            size=x.shape[2:],
                            mode='trilinear',
                            align_corners=True,
                        )
                    features_to_concat.append(motion_feat)

                if len(features_to_concat) > 1:
                    x = torch.cat(features_to_concat, dim=1)

            x = layer(x)

        for layer in self.image_decoder_extras:
            x = layer(x)

        return x

    def forward(self, source_proj, target_proj, source_vol, mode='motion'):
        if target_proj is None:
            raise ValueError("target_proj must be provided for dual-decoder models.")

        depth = source_vol.shape[2]

        motion_feat = self._expand_projection(self.motion_proj_embedder, source_proj, target_proj, depth)
        motion_x = torch.cat([motion_feat, source_vol], dim=1)
        motion_x_enc = self.motion_encoder(motion_x)

        image_feat = self._expand_projection(self.image_proj_embedder, source_proj, target_proj, depth)
        image_x = torch.cat([image_feat, source_vol], dim=1)
        image_x_enc = self.image_encoder(image_x)

        if mode == 'motion':
            flow, _ = self.motion_decoder(motion_x_enc)
            if self.integrator is not None:
                flow = self.integrator(flow)
            y_source = self.final_transformer(source_vol, flow)
            return y_source, flow

        if mode == 'image':
            with torch.no_grad():
                _, motion_decoder_features = self.motion_decoder(motion_x_enc)
            synth_volume = self._run_image_decoder(
                image_x_enc,
                motion_decoder_features,
                detach_motion_features=True,
            )
            return synth_volume, None

        if mode == 'joint':
            motion_flow, motion_decoder_features = self.motion_decoder(motion_x_enc)
            if self.integrator is not None:
                motion_flow = self.integrator(motion_flow)
            motion_volume = self.final_transformer(source_vol, motion_flow)

            synth_volume = self._run_image_decoder(
                image_x_enc,
                motion_decoder_features,
                detach_motion_features=False,
            )

            return {
                'motion': {'volume': motion_volume, 'flow': motion_flow},
                'image': {'volume': synth_volume},
            }

        raise ValueError(f"Unsupported mode '{mode}'.")


# ============================================================================
# Original Model (for reference)
# ============================================================================

class OriginalModel(LoadableModel):
    """Original single encoder-decoder architecture"""

    @store_config_args
    def __init__(self, im_size, int_steps=7, proj_in_channels=2):
        super().__init__()

        self.im_size = im_size

        enc_nf = [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]
        self.enc_nf = enc_nf

        self.proj_embedder = ProjectionEmbedder(proj_in_channels, enc_nf[0])
        self.encoder = Encoder3D(enc_nf[0] + 1, enc_nf)
        self.decoder = Decoder3D(enc_nf, out_channels=3)

        # Flow integrator (single level)
        vol_shape = [im_size, im_size, im_size]
        self.integrator = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None

        # Final transformer
        self.final_transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, source_proj, target_proj, source_vol, mode='motion'):
        proj_pair = torch.cat([source_proj, target_proj], dim=1)
        target_feat = self.proj_embedder(proj_pair)
        target_feat = target_feat.unsqueeze(2)
        depth = source_vol.shape[2]
        target_feat = target_feat.expand(-1, -1, depth, -1, -1)

        x = torch.cat([target_feat, source_vol], dim=1)
        x_enc = self.encoder(x)
        flow, decoder_features = self.decoder(x_enc)

        # Integrate flow if needed
        if self.integrator is not None:
            flow = self.integrator(flow)

        # Warp source volume
        y_source = self.final_transformer(source_vol, flow)

        return y_source, flow


# ============================================================================
# Building Blocks
# ============================================================================

class DownBlock3d(nn.Module):
    """Residual layers for the encoding direction"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn((self.conv2(conv1)))
        out = self.activation(conv1 + conv2)
        return out


class UpBlock(nn.Module):
    """Residual layers for the decoding direction"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn((self.conv2(conv1)))
        out = self.activation(conv1 + conv2)
        return out


class ExtraBlock(nn.Module):
    """Specific convolutional block with tanh activation"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        return out
