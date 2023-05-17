import torch
import torch.nn as nn

class epipolar_sampling(nn.Module):
    """
        Module for sampling features from a map along epipolar lines.
        Arguments:
            conf: Configuration dictionary containing:
                interpolation: Interpolation mode in grid_sampling. Values: bilinear, nearest, or bicubic
                sampling_dim: Number of points along the epipolar line that will be sampled. Size: (1).
            down_extractor: Downsampling factor due to feature extractor
            sampling_dist: Distance between points in the epipolar line. Corresponds to h^2 = dx^2 + dy^2. Size: (1)
            device: Indicates on which device tensors should be
    """

    def __init__(self, conf, down_extractor=1, sampling_dist=8, device='cuda'):
        super(epipolar_sampling, self).__init__()

        # Define epipolar sampling parameters
        self.interpolation = conf['F_INTER']
        # sampling_dim defines the number of points that are sampled along the epipolar line
        self.sampling_dim = conf['SAMPLING_DIM']
        self.sampling_dist = sampling_dist
        self.down_extractor = down_extractor
        self.eps = 1e-16
        self.device = device
        self.define_auxiliary_variables()

    def define_auxiliary_variables(self):

        # Define auxiliary parameters
        self.ones = torch.ones((1, 1, 1))
        self.sampling_x = torch.arange(self.sampling_dim).view(1, 1, self.sampling_dim).float()

        self.x_0s_start_pos = torch.zeros((1, 1, 1))
        self.y_0s_start_pos = torch.zeros((1, 1, 1))
        self.x_ws_start_pos = torch.ones((1, 1, 1))
        self.y_hs_start_pos = torch.ones((1, 1, 1))
        self.ones_start_pos = torch.ones((1, 1))
        self.zeros_start_pos = torch.zeros((1, 1))

    def get_sampling_pos(self, query_kps, feat_maps_shape, F,
                disp_a=None, scale_a=None,
                disp_b=None, scale_b=None):
        ''' Code for sampling features from a map along epipolar lines.
            Input:
                query_kps: Pixel positions (x, y) in map 1 corresponding to the desired epipolar lines in map 2.
                    Size: (B, N, 2), where N refers to the query points.
                feat_maps: Feature map from which query the values along epipolar lines. Size: (B, C, H, W)
                F: Fundamental matrix between map 1 and map 2. Size: (B, 3, 3)
                disp: Displacement on the feature map due to possible cropping. Size: (B, 2). If None, no displacement.
                scale: Scaling factor on the image before feature extraction. Size: (B, 2). If None, no scaling.
            Output:
                sampling_kpts_map: Generated query points (map resolution). Size: (B, N, sampling_dim, 2)
                sampling_kpts: Generated query points (original image resolution). Size: (B, N, sampling_dim, 2)
                Note: sampling_dim determines the number of points that are sampled in every epipolar line
        '''

        with torch.no_grad():

            b, c, h, w = feat_maps_shape
            b, num_kpts, q_chan = query_kps.shape
            ones = self.ones.expand([b, num_kpts, 1]).to(query_kps.device)

            # Map points to original resolution
            new_kps = self.map_points_to_original_res(query_kps, scale_a, disp_a)

            # Homogeneous coordinates
            new_kps = torch.cat([new_kps, ones], dim=-1)

            # Compute epipolar lines for every query keypoint
            lines = (F @ new_kps.transpose(-2, -1)).transpose(-2, -1)
            tmp = lines[:, :, 1]
            lines[:, :, 1] = torch.where(tmp == torch.zeros_like(tmp), torch.zeros_like(tmp) + self.eps, tmp)
            lines_norm = (-1. * lines / lines[:, :, 1].unsqueeze(-1)).unsqueeze(-1)

            # Define starting positions for every epipolar line
            x1_0_start = self.create_starting_sampling_pos(lines_norm, h, w, scale_b, disp_b)

            # Map sampling distance to original image plane
            im_sampling_dist = self.sampling_dist / (scale_b[:, 0].unsqueeze(-1).unsqueeze(-1) * self.down_extractor)
            displ = self.compute_x_displacement(lines_norm[:, :, 0], im_sampling_dist)

            # As we do equidistant sampling in the image plane, we need to map the points again
            x1_0 = torch.cat([x1_0_start, ones], dim=-1)
            x1_0 = self.map_points_to_original_res(x1_0, scale_b, disp_b)[:, :, 0].unsqueeze(-1)

            # Compute all sampling positions based on starting point and displacement
            sampling_x = torch.tile(self.sampling_x, (b, num_kpts, 1)).to(x1_0.device)
            x1 = x1_0 + displ * sampling_x
            y1 = lines_norm[:, :, 0] * x1 + lines_norm[:, :, 2]
            sampling_kpts = torch.cat([x1.unsqueeze(-1), y1.unsqueeze(-1)], dim=-1)

            # Map points to feature map resolution
            sampling_kpts_map = self.map_points_to_feature_map_res(sampling_kpts, scale_b, disp_b)

        return sampling_kpts_map, sampling_kpts

    def sample_along_epi_line(self, sampling_kpts_map, feat_maps):

        # Normalise points for grid sampling
        x1_norm = 2. * sampling_kpts_map[:, :, :, 0] / (feat_maps.shape[3] - 1) - 1.0
        y1_norm = 2. * sampling_kpts_map[:, :, :, 1] / (feat_maps.shape[2] - 1) - 1.0
        grid_sample = torch.stack((x1_norm, y1_norm), 3)
        grid_sample = torch.clamp(grid_sample, min=-2, max=2)

        sampled_map = torch.nn.functional.grid_sample(feat_maps, grid_sample,
                                                      mode=self.interpolation, align_corners=True)

        return sampled_map, grid_sample

    def compute_x_displacement(self, m, sampling_dist):
        ''' Code for computing the displacement in the x-axis during sampling. The computed dx assures that
            the sampling distance between points along the epipolar line are equidistant.
            Input:
                m: The slope of the epipolar line. Size: (B, N, 1).
                sampling_dist: Desired sampling distance between points. Size: (B, 1, 1).
            Output:
                dx: displacement in x for getting equidistant points along the epipolar line. Size: (B, N, 1).
        '''

        # Only takes into account positives displacements.
        dx = sampling_dist * (1 / (1 + m**2.)**0.5)
        return dx


    def create_starting_sampling_pos(self, lines, H, W, scale, disp):
        ''' Code for computing the starting position for sampling in the x-axis during sampling.
                Input:
                    lines: The epipolar lines to find the starting sampling position. Size: (B, N, 3, 1)
                    H, W: resolution feature map
                    disp: Displacement on the feature map due to possible cropping. Size: (B, 2). If None, no displacement.
                    scale: Scaling factor on the image before feature extraction. Size: (B, 2). If None, no scaling.
                Output:
                    starting_x0: Starting position for sampling in the x-axis. Size: (B, N, 1)
        '''

        b, n, _, _ = lines.shape

        x_0s = self.x_0s_start_pos.expand([b, 1, 1]).to(lines.device)
        y_0s = self.y_0s_start_pos.expand([b, 1, 1]).to(lines.device)
        x_ws = W * self.x_ws_start_pos.expand([b, 1, 1]).to(lines.device)
        y_hs = H * self.y_hs_start_pos.expand([b, 1, 1]).to(lines.device)
        ones = self.ones_start_pos.expand([b, n]).to(lines.device)

        # Define anchor points
        top_left = torch.cat([x_0s, y_0s], dim=-1)
        bottom_right = torch.cat([x_ws, y_hs], dim=-1)

        query_kps = torch.cat([top_left, bottom_right], dim=1)
        new_kps = self.map_points_to_original_res(query_kps, scale, disp)

        # Add dimension for multiple lines
        new_kps = new_kps.unsqueeze(1).unsqueeze(-1)
        new_kps = torch.tile(new_kps, (1, n, 1, 1, 1))

        # Find x points intersecting with the image plane
        pts_x0 = torch.cat([new_kps[:, :, 0, 0], new_kps[:, :, 0, 0]*lines[:, :, 0, :]+lines[:, :, 2, :]], dim=-1).unsqueeze(2)
        pts_xw = torch.cat([new_kps[:, :, 1, 0], new_kps[:, :, 1, 0]*lines[:, :, 0, :]+lines[:, :, 2, :]], dim=-1).unsqueeze(2)
        tmp = lines[:, :, 0, :]
        lines[:, :, 0, :] = torch.where(tmp == torch.zeros_like(tmp), torch.zeros_like(tmp) + self.eps, tmp)

        # Find the corresponding y positions
        pts_y0 = torch.cat([(new_kps[:, :, 0, 1]-lines[:, :, 2, :])/lines[:, :, 0, :], new_kps[:, :, 0, 1]], dim=-1).unsqueeze(2)
        pts_yh = torch.cat([(new_kps[:, :, 1, 1]-lines[:, :, 2, :])/lines[:, :, 0, :], new_kps[:, :, 1, 1]], dim=-1).unsqueeze(2)

        # Map points to feature map resolution
        query_kps = torch.cat([pts_x0, pts_xw, pts_y0, pts_yh], dim=2)
        sampling_kpts = self.map_points_to_feature_map_res(query_kps, scale, disp)

        # Find points out of the feature map plane
        valid_x_0 = torch.where((sampling_kpts[:, :, 0, 1] >= 0) * (sampling_kpts[:, :, 0, 1] < H),
                                sampling_kpts[:, :, 0, 0], 2*W*ones).unsqueeze(-1)
        valid_x_w = torch.where((sampling_kpts[:, :, 1, 1] >= 0) * (sampling_kpts[:, :, 1, 1] < H),
                                sampling_kpts[:, :, 1, 0], 2*W*ones).unsqueeze(-1)

        valid_x_0_y = torch.where((sampling_kpts[:, :, 2, 0] >= 0) * (sampling_kpts[:, :, 2, 0] < W),
                                  sampling_kpts[:, :, 2, 0], 2*W*ones).unsqueeze(-1)
        valid_x_h_y = torch.where((sampling_kpts[:, :, 3, 0] >= 0) * (sampling_kpts[:, :, 3, 0] < W),
                                  sampling_kpts[:, :, 3, 0], 2*W*ones).unsqueeze(-1)

        # Define the starting point positions
        starting_x0 = torch.cat([valid_x_0, valid_x_w, valid_x_0_y, valid_x_h_y], dim=-1)
        starting_x0, _ = torch.min(starting_x0, dim=-1, keepdim=True)

        return starting_x0

    def map_points_to_original_res(self, query_kps, scale, disp):
        ''' Code for mapping points from the feature map resolution to the original image size.
                Input:
                    query_kps: Input query points to map to the image resolution. Size: (B, N, 2)
                    disp: Displacement on the feature map due to possible cropping. Size: (B, 2). If None, no displacement.
                    scale: Scaling factor on the image before feature extraction. Size: (B, 2). If None, no scaling.
                Output:
                    new_kps: Query points in the image plane. Size: (B, N, 2)
            '''

        b, num_pts, _ = query_kps.shape

        # Map points to original resolution
        if scale is not None:
            new_kps = query_kps / scale.view(b, 1, 2)
        else:
            new_kps = query_kps

        if disp is not None:
            new_kps = new_kps + disp.view(b, 1, 2)

        return new_kps

    def map_points_to_feature_map_res(self, query_kps, scale, disp):
        ''' Code for mapping points from the imge plane to the feature map resolution.
                Input:
                    query_kps: Input query points to map to the feature map resolution. Size: (B, N, sampling_dim, 2)
                    disp: Displacement on the feature map due to possible cropping. Size: (B, 2). If None, no displacement.
                    scale: Scaling factor on the image before feature extraction. Size: (B, 2). If None, no scaling.
                Output:
                    new_kps: Query points in the feature map plane. Size: (B, N, sampling_dim, 2)
            '''

        b, num_pts, num_edges, _ = query_kps.shape

        # Map points to feature map resolution
        if disp is not None:
            disp = disp.view(b, 1, 1, 2)
            new_kps = query_kps - disp
        else:
            new_kps = query_kps
        if scale is not None:
            scale = scale.view(b, 1, 1, 2)
            new_kps = new_kps * scale

        return new_kps
