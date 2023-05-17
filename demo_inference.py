import numpy as np
import argparse
import torch
import os
import yaml
from FSNet.FSNet_model import FSNet_model_handler
from FSNet.modules.utils import ImagePairLoader
from FSNet.modules.utils import plot_epipolar_lines

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='F-Mat inference demo script')

    parser.add_argument('--conf_path', type=str, default='FSNet/config/conf.yaml',
                        help='Path to the FSNet configuration file')

    parser.add_argument('--weights_path', type=str, default='FSNet/weights/indoor_fundamentals.pth',
                        help='Path to the network weights')

    parser.add_argument('--src_path', type=str, default='resources/im_test/im_src.jpg',
        help='Path to the source test image')

    parser.add_argument('--dst_path', type=str, default='resources/im_test/im_dst.jpg',
                        help='Path to the destination test image')

    parser.add_argument('--fundamentals_path', type=str, default='resources/im_test/fundamentals.npy',
                        help='Path to the file (.npy) storing the fundamentals matrix to be scored.')

    parser.add_argument('--plot_epi_lines', type=bool, default=True,
                        help='Define whether a plot showing epipolar lines generated from top scoring '
                             'fundamental matrix should be created.')

    parser.add_argument('--path_epi_fig', type=str, default='im_test/epi_lines.jpg',
                        help='Path to the epipolar lines plot.')

    parser.add_argument('--gpu_id', type=str, default='0', help='GPU index where FSNet should run')

    args = parser.parse_args()

    with open(args.conf_path) as file:
        conf_model = yaml.load(file, Loader=yaml.FullLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.weights_path
    FSNet = FSNet_model_handler(conf_model, weights, device)

    # Prepare data loaders and image paths
    im_size = conf_model['ENCODER']['IM_SIZE']
    loader = ImagePairLoader(im_size, device)
    path_src = args.src_path
    path_dst = args.dst_path
    fmats_path = args.fundamentals_path

    with torch.no_grad():

        # Load and prepare fundamental matrices
        F_mats = np.load(fmats_path)
        F_mats_torch = torch.Tensor(F_mats).to(device)

        # Load images and prepare them for FSNet processing
        data_pair = loader.prepare_im_pair(path_src, path_dst)

        # Run FSNet scoring method
        best_fmat, errors = FSNet(data_pair, F_mats_torch)

    # Plot pose errors. Note that FSNet is design for ranking of
    # fundamental matrices, not for absolute pose error regression
    best_errors = errors[best_fmat].cpu().numpy()
    print('FSNet translation and rotation error predictions are {:.2f}° and {:.2f}°, respectively'.format(best_errors[0], best_errors[1]))

    # Plot epipolar lines for selected fundamental matrix
    path_epi_fig = args.path_epi_fig
    plot_epi_lines = args.plot_epi_lines
    if plot_epi_lines:
        plot_epipolar_lines(path_src, path_dst, F_mats[best_fmat], path_epi_fig)
