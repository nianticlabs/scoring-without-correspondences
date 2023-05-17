"""
Some functions are borrowed from LoFTR: Detector-Free Local
Feature Matching with Transformers (https://github.com/zju3dv/LoFTR)
If using this code, please consider citing LoFTR.
"""

import copy
import torch
import torch.nn as nn
from einops.einops import rearrange
from FSNet.modules.linear_attention import MixAttention
from FSNet.modules.utils import define_sampling_grid
from FSNet.modules.epipolar_sampling import epipolar_sampling

class EncoderLayer(nn.Module):
    """
        Transformer encoder layer containing the linear self and cross-attention, and the epipolar attention.
        Arguments:
            d_model: Feature dimension of the input feature maps (default: 128d).
            nhead: Number of heads in the multi-head attention.
            epi_dim: Number of sampling positions along the epipolar line.
            attention: Type of attention for the common transformer block. Options: linear, full.
    """
    def __init__(self, d_model, nhead, epi_dim, attention='linear'):
        super(EncoderLayer, self).__init__()

        # Transformer encoder layer parameters
        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention definition
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        full_att = False if attention == 'linear' else True
        self.attention = MixAttention(epi_dim=epi_dim, full_att=full_att)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, is_epi_att=False):
        """
        Args:
            x (torch.Tensor): [N, L, C] (L = im_size/down_factor ** 2)
            source (torch.Tensor): [N, S, C]
            if is_epi_att:
                S = (im_size/down_factor/step_grid) ** 2 * sampling_dim
            else:
                S = im_size/down_factor ** 2
            is_epi_att (bool): Indicates whether it applies epipolar cross-attention
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, is_epi_att=is_epi_att)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class Transformer(nn.Module):
    """This class implement linear attention and epipolar cross-attention.
        Arguments:
            im_size: Original input size to FSNet (default: 256).
            d_model: Feature dimension after feature extractor (default: 128d).
            down_extractor: Downsampling factor due to feature extractor.
            aggregator_conf: Configuration dictionary containing the parameters for the transformer module.
            sampling_conf: Configuration dictionary containing the parameters for the epipolar sampling.
            device: Indicates on which device tensors should be
    """

    def __init__(self, im_size, d_model, down_factor, aggregator_conf, sampling_conf, device):
        super(Transformer, self).__init__()

        # Define the transformer parameters
        self.d_model = d_model
        layer_names = aggregator_conf['TRANSFORMER']['LAYER_NAMES'] * aggregator_conf['TRANSFORMER']['NUMBER_LAYER']
        attention = aggregator_conf['TRANSFORMER']['ATTENTION']
        self.nheads = aggregator_conf['TRANSFORMER']['NHEADS']
        self.layer_names = layer_names
        self.F_dim = sampling_conf['SAMPLING_DIM']
        encoder_layer = EncoderLayer(d_model, self.nheads, self.F_dim, attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

        # Define Sampling module
        # Sampling distance for image 256x256 is 8.
        # Modify sampling distance or
        sampling_dist = 8 * (im_size / 256)

        # step_grid reduces the epipolar sampling query points
        self.step_grid = 2
        query_kps = define_sampling_grid(im_size, down_factor, self.step_grid)
        self.query_kps = torch.from_numpy(query_kps).float().unsqueeze(0)

        # Define epipolar sampling module
        self.F_sampling = epipolar_sampling(sampling_conf, down_extractor=down_factor,
                                            sampling_dist=sampling_dist, device=device)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward_common_transf(self, feat_a, feat_b):
        """
            Runs the common self and cross-attention module.
            Args:
                feats_a: Features from image A (source) ([N, d_model, im_size/down_factor, im_size/down_factor]).
                feats_b: Features from image B (destination) ([N, d_model, im_size/down_factor, im_size/down_factor]).
            Output:
                feats_a: Self and cross-attended features corresponding to image A (source)
                ([N, d_model, im_size/down_factor, im_size/down_factor])
                feats_b: Self and cross-attended features corresponding to image B (destination)
                ([N, d_model, im_size/down_factor, im_size/down_factor]).
        """

        assert self.d_model == feat_a.size(1), "The feature size and transformer must be equal"

        b, c, h, w = feat_a.size()

        feat_a = rearrange(feat_a, 'n c h w -> n (h w) c')
        feat_b = rearrange(feat_b, 'n c h w -> n (h w) c')

        # Apply linear self and cross attention to feat_a and feat_b features
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':

                feat_a = layer(feat_a, feat_a)
                feat_b = layer(feat_b, feat_b)

            elif name == 'cross':

                feat_a_ = layer(feat_a, feat_b)
                feat_b = layer(feat_b, feat_a)
                feat_a = feat_a_

        feat_a = feat_a.transpose(2, 1).reshape((b, c, h, w))
        feat_b = feat_b.transpose(2, 1).reshape((b, c, h, w))

        return feat_a, feat_b


    def forward_epi_crossAtt(self, feat_a, feat_b, F, disp_a, scale_a, disp_b, scale_b):
        """
            Runs the epipolar and cross-attention module.
            Args:
                feats_a: Features from image A (source) ([N, d_model, im_size/down_factor, im_size/down_factor]).
                feats_b: Features from image B (destination) ([N, d_model, im_size/down_factor, im_size/down_factor]).
                F: Query fundamental matrices ([N, 3, 3]).
                disp_src: Reference displacement to center crop ([N, 2]).
                scale_src: Reference scale factor to desired image size ([N, 2]).
                disp_src: Destination displacement to center crop ([N, 2]).
                scale_src: Destination scale factor to desired image size ([N, 2]).
            Output:
                feats_a: Epipolar cross-attended features corresponding to image A (source)
                ([N, d_model, im_size/down_factor/step_grid, im_size/down_factor/step_grid])
                feats_b: Epipolar cross-attended features corresponding to image B (destination)
                ([N, d_model, im_size/down_factor/step_grid, im_size/down_factor/step_grid]).
        """

        assert self.d_model == feat_a.size(1), "the feature number and transformer must be equal"
        b, c, h, w = feat_a.size()

        # Define epipolar line positions where to sample candidate features
        feat_a_shape = feat_a.shape
        feat_b_shape = feat_b.shape
        query_kps = torch.tile(self.query_kps, (b, 1, 1)).to(feat_a.device)
        sampling_a, _ = self.F_sampling.get_sampling_pos(query_kps, feat_a_shape, F.transpose(2, 1), disp_a=disp_b,
                                                scale_a=scale_b, disp_b=disp_a, scale_b=scale_a)
        sampling_b, _ = self.F_sampling.get_sampling_pos(query_kps, feat_b_shape, F, disp_a=disp_a, scale_a=scale_a,
                                                disp_b=disp_b, scale_b=scale_b)

        feat_a_Fsampled, _ = self.F_sampling.sample_along_epi_line(sampling_a, feat_a)
        feat_b_Fsampled, _ = self.F_sampling.sample_along_epi_line(sampling_b, feat_b)

        # Normalise points for grid sampling
        x1_norm = 2. * self.query_kps[:, :, 0] / (feat_a.shape[3] - 1) - 1.0
        y1_norm = 2. * self.query_kps[:, :, 1] / (feat_a.shape[2] - 1) - 1.0
        grid_sample = torch.stack((x1_norm, y1_norm), 2).unsqueeze(2)
        grid_sample = torch.clamp(grid_sample, min=-2, max=2)
        grid_sample = torch.tile(grid_sample, [b, 1, 1, 1]).to(feat_a.device)
        feat_a_grid = torch.nn.functional.grid_sample(feat_a, grid_sample, mode='bilinear',
                                                      align_corners=True).squeeze(-1).permute(0, 2, 1)
        feat_b_grid = torch.nn.functional.grid_sample(feat_b, grid_sample, mode='bilinear',
                                                      align_corners=True).squeeze(-1).permute(0, 2, 1)

        # Prepare features
        feat_a_Fsampled = rearrange(feat_a_Fsampled, 'n c q d -> n (q d) c')
        feat_b_Fsampled = rearrange(feat_b_Fsampled, 'n c q d -> n (q d) c')

        # Apply epipolar attention
        feat_a = self.layers[-1](feat_a_grid, feat_b_Fsampled, is_epi_att=True)
        feat_b = self.layers[-1](feat_b_grid, feat_a_Fsampled, is_epi_att=True)

        # Reshape features before the error regressor block
        feat_a = feat_a.transpose(2, 1).reshape((b, c, h//2, w//2))
        feat_b = feat_b.transpose(2, 1).reshape((b, c, h//2, w//2))

        return feat_a, feat_b
