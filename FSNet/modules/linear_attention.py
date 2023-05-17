"""
Some functions are borrowed from LoFTR: Detector-Free Local
Feature Matching with Transformers (https://github.com/zju3dv/LoFTR) and modified here.
If using this code, please consider citing LoFTR.
"""

import torch
from torch.nn import Module, Dropout

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class MixAttention(Module):
    def __init__(self, eps=1e-6, epi_dim=45, use_dropout=False, full_att=False, attention_dropout=0.1):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps
        self.epi_dim = epi_dim
        self.use_dropout = use_dropout
        self.epi_dim = epi_dim
        self.dropout = Dropout(attention_dropout)
        self.full_att = full_att

    def forward_full(self, queries, keys, values):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()

    def forward_linear(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
            Args:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            Returns:
                queried_values: (N, L, H, D)
            """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


    def forward_epi_cross(self, queries, keys, values):
        """ Multi-head scaled epipolar dot-product attention
            Args:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            Returns:
                queried_values: (N, L, H, D)
        """
        b, num_pts, h, c = queries.size()

        keys = keys.reshape((b, num_pts, self.epi_dim, h, c))
        values = values.reshape((b, num_pts, self.epi_dim, h, c))

        QK = (queries.unsqueeze(2) * keys).sum(-1)

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3) ** .5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = (A.unsqueeze(-1) * values).sum(2)

        return queried_values.contiguous()

    def forward(self, queries, keys, values, is_epi_att=False):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
            Args:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            Returns:
                x: queried values (N, L, H, D)
        """
        if is_epi_att:
            x = self.forward_epi_cross(queries, keys, values)
        else:
            if self.full_att:
                x = self.forward_full(queries, keys, values)
            else:
                x = self.forward_linear(queries, keys, values)
        return x
