import torch
import torch.nn as nn
import numpy as np
from FSNet.modules.utils import return_fmat_score
from FSNet.modules.backbone import ResNetFPN_4
from FSNet.modules.transformer import Transformer
from FSNet.modules.error_regressor import Features2PoseError

class FSNet_model(nn.Module):
    """
        Fundamental Scoring Network (FSNet)

        Given two images and a query fundamental matrix,
        FSNet predicts the relative translation and rotation pose errors.

        Axel Barroso-Laguna, Eric Brachmann, Victor Adrian Prisacariu,
        Gabriel Brostow and Daniyar Turmukhambetov. Two-view Geometry
        Scoring Without Correspondences. In CVPR, 2023.
    """

    def __init__(self, conf, device):
        super(FSNet_model, self).__init__()

        d_model = conf['ENCODER']['ENCODER_DIM']
        im_size = conf['ENCODER']['IM_SIZE']
        aggregator_conf = conf['AGGREGATOR']
        sampling_conf = conf['FMAT_SAMPLING']

        # Feature extractor
        self.extractor = ResNetFPN_4(conf['ENCODER'])
        self.down_factor = 4

        # Transformer block
        self.transformer = Transformer(im_size, d_model, self.down_factor, aggregator_conf, sampling_conf, device)

        # Error regressor
        self.rel_err = Features2PoseError(d_model, aggregator_conf)

    def forward_extractor(self, im_a, im_b):
        """
            Runs feature extraction from input images
            Args:
                im_a: Image A (source) ([N, 3, im_size, im_size])
                im_b: Image B (destination) ([N, 3, im_size, im_size]).
            Output:
                feats_a: Computed features from image A (source) ([N, d_model, im_size, im_size])
                feats_b: Computed features from image B (destination) ([N, d_model, im_size, im_size]).
        """

        b = im_a.size(0)
        feats = self.extractor(torch.cat([im_a, im_b], dim=0))
        return feats[:b], feats[b:]

    def forward_common_transf(self, feats_a, feats_b):
        """
            Runs the common self and cross attention module.
            Args:
                feats_a: Features from image A (source) ([N, d_model, im_size/down_factor, im_size/down_factor])
                feats_b: Features from image B (destination) ([N, d_model, im_size/down_factor, im_size/down_factor]).
            Output:
                feats_a: Self and cross-attended features corresponding to image A (source)
                ([N, d_model, im_size/down_factor, im_size/down_factor])
                feats_b: Self and cross-attended features corresponding to image B (destination)
                ([N, d_model, im_size/down_factor, im_size/down_factor]).
        """

        feats_a, feats_b = self.transformer.forward_common_transf(feats_a, feats_b)

        return feats_a, feats_b

    def forward_epi_crossAtt(self, feats_a, feats_b, F, disp_a, scale_a, disp_b, scale_b):
        """
            Runs the fundamental matrix specific epipolar cross attention module and computes its corresponding
            translation and rotation pose errors.
            Args:
                feats_a: Features from image A (source) ([N, d_model, im_size/down_factor, im_size/down_factor])
                feats_b: Features from image B (destination) ([N, d_model, im_size/down_factor, im_size/down_factor]).
                F: Query fundamental matrices ([N, 3, 3]).
                disp_src: Reference displacement to center crop ([N, 2])
                scale_src: Reference scale factor to desired image size ([N, 2])
                disp_src: Destination displacement to center crop ([N, 2])
                scale_src: Destination scale factor to desired image size ([N, 2])
            Output:
                pose_err: Predicted translation and rotation pose error for input fundamental matrices ([N, 2])
        """
        scale_a = scale_a / self.down_factor
        scale_b = scale_b / self.down_factor

        feats_a, feats_b = self.transformer.forward_epi_crossAtt(feats_a, feats_b, F, disp_a, scale_a, disp_b, scale_b)

        pose_err = self.rel_err(feats_a, feats_b)

        return pose_err


class FSNet_model_handler(nn.Module):
    """
        FSNet_model_handler

        It deals with the logic to select the best fundamental matrix
        in a batch. Given two input images, a set of fundamental matrices,
        FSNet_model_handler returns the top scoring fundamental matrix,
        and its predicted rotation and translation errors.

        Args:
            conf: FSNet configuration
            weights: Path to the FSNet trained weights
            device: Device id to run FSNet
    """
    def __init__(self, conf, weights, device):
        super(FSNet_model_handler, self).__init__()

        self.model = FSNet_model(conf, device)

        state_dict = torch.load(weights)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.Fmats_batch = conf['FMAT_SAMPLING']['FMAT_BARCH_SIZE']

    def forward(self, input_data, F_data):
        """
            Run FSNet for a pair of images and a batch of fundamental matrices.

            Args:
                input_data: Dictionary containing. Inference code supported for N=1:
                    src_img: Reference image ([N, 3, IM_SIZE, IM_SIZE] (IM_SIZE=256))
                    disp_src: Reference displacement to center crop ([N, 2])
                    scale_src: Reference scale factor to desired image size ([N, 2])
                    dst_img: Destination image. ([N, 3, IM_SIZE, IM_SIZE] (IM_SIZE=256))
                    disp_src: Destination displacement to center crop ([N, 2])
                    scale_src: Destination scale factor to desired image size ([N, 2])
                F_data: Query fundamental matrices ([Nf, 3, 3], where Nf refers to
                 the total number of fundamental matrices to score)
            Outputs:
                best_fmat: Index of the best fundamental matrix based on FSNet scoring.
                fmat_errs: Predicted pose error for query fundamental matrices ([Nf, 2]).
        """
        src_img = input_data['src_img']
        disp_src = input_data['disp_src']
        scale_src = input_data['scale_src']
        dst_img = input_data['dst_img']
        disp_dst = input_data['disp_dst']
        scale_dst = input_data['scale_dst']

        fmat_errs = torch.zeros([0, 2], device=src_img.device)
        num_fmats = len(F_data)
        max_score = -1 * np.inf
        best_fmat = 0

        src_feats, dst_feats = self.model.forward_extractor(src_img, dst_img)
        src_feats, dst_feats = self.model.forward_common_transf(src_feats, dst_feats)

        num_loops = num_fmats // self.Fmats_batch
        num_loops += 1 if num_fmats % self.Fmats_batch else 0
        for i_fmats in range(0, num_loops):
            F_batch = F_data[i_fmats * self.Fmats_batch: (1 + i_fmats) * self.Fmats_batch]
            num_fmats_i = len(F_batch)
            errs = self.model.forward_epi_crossAtt(src_feats.expand([num_fmats_i, -1, -1, -1]),
                                   dst_feats.expand([num_fmats_i, -1, -1, -1]), F_batch,
                                   disp_src.expand([num_fmats_i, -1]), scale_src.expand([num_fmats_i, -1]),
                                   disp_dst.expand([num_fmats_i, -1]), scale_dst.expand([num_fmats_i, -1]))

            fmat_score, index_max = return_fmat_score(errs)
            fmat_errs = torch.concat([fmat_errs, errs], dim=0)

            if max_score < fmat_score:
                max_score = fmat_score
                best_fmat = i_fmats * self.Fmats_batch + index_max

        return best_fmat, fmat_errs