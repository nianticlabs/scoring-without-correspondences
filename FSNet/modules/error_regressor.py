import torch
import torch.nn as nn
from FSNet.modules.regressor_layers import DeepResBlock, MLP

class Features2PoseError(nn.Module):
    '''
        Features2PoseError computes the relative pose error for a query fundamental matrix
        given the dense feature extracted from image A and image B. It outputs the translation
        and rotation pose errors independently.
        Arguments:
            conf: Configuration dictionary containing the parameters for the error regressor block.
            d_model: Feature dimension after feature extractor and attention layers (default: 128d).
    '''

    def __init__(self, d_model, conf):
        super(Features2PoseError, self).__init__()

        dim_vector = 512
        self.process_volume = DeepResBlock(d_model, dim_vector, conf['BATCH_NORM'])
        self.flat_conv = MLP(dim_vector, 1, batch_norm=conf['MLP_BN'])
        self.flat_conv_r = MLP(dim_vector, 1, batch_norm=conf['MLP_BN'])
        self.maxpool = torch.nn.MaxPool2d((2, 1))

    def forward(self, xa, xb):
        """ Proccess features and compute the order-invariant translation and rotation pose error.
        Args:
            xa (torch.Tensor): features from image A after common and epipolar attention blocks.
            xb (torch.Tensor): features from image B after common and epipolar attention blocks.
        Outputs:
            x: Translation and rotation errors (N, 2).
        """
        b = xa.size(0)
        x = self.process_volume(torch.cat([xa, xb], dim=0))
        x = self.pred_symm_fmat_err(x[:b], x[b:])

        return x

    def pred_symm_fmat_err(self, xa, xb):
        """ Predict the order-invariant translation and rotation pose error.
        Args:
            xa (torch.Tensor): Post-processed features from image A.
            xb (torch.Tensor): Post-processed features from image B.
        Outputs:
            x: Translation and rotation errors (N, 2).
        """
        xa = xa.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        xb = xb.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        x = torch.cat([xa, xb], dim=2)
        x = self.maxpool(x)

        err_t = self.flat_conv(x).squeeze(-1).squeeze(-1)
        err_r = self.flat_conv_r(x).squeeze(-1).squeeze(-1)
        err = torch.cat([err_t, err_r], dim=-1)

        return err
