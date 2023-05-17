import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import cv2

def define_sampling_grid(im_size, feats_downsample=4, step=1):
    """
        Auxiliary function to generate the sampling grid from the feature map
        Args:
            im_size: original image size that goes into the network
            feats_downsample: rescaling factor that happens within the architecture due to downsampling steps
        Output:
            indexes_mat: dense grid sampling indexes, size: (im_size/feats_downsample, im_size/feats_downsample)
    """

    feats_size = int(im_size/feats_downsample)
    grid_size = int(im_size/feats_downsample/step)

    indexes = np.asarray(range(0, feats_size, step))[:grid_size]
    indexes_x = indexes.reshape((1, len(indexes), 1))
    indexes_y = indexes.reshape((len(indexes), 1, 1))

    indexes_x = np.tile(indexes_x, [len(indexes), 1, 1])
    indexes_y = np.tile(indexes_y, [1, len(indexes), 1])

    indexes_mat = np.concatenate([indexes_x, indexes_y], axis=-1)
    indexes_mat = indexes_mat.reshape((grid_size*grid_size, 2))

    return indexes_mat


def return_fmat_score(err_preds, eps=1e-16):
    """
        Auxiliary function to convert predicted errors into single score value
        Args:
            err_preds: errors (translation and rotation) predicted by FSNet ([N, 2])
        Output:
            fmat_score: top scoring value for the input batch of predicted errors
            max_idx: index position of the top scoring fundamental matrix
    """
    err_pose, _ = torch.max(torch.cat([err_preds[:, 0].unsqueeze(-1), err_preds[:, 1].unsqueeze(-1)], dim=-1), dim=-1)
    max_idx = torch.argmin(err_pose).detach().cpu().numpy().item()
    fmat_score = 1 / (max(err_pose[max_idx].detach().cpu().numpy().item(), eps))

    return fmat_score, max_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class ImagePairLoader():
    """
        Auxiliary class to read images in a way that FSNet can digest them. FSNet takes low-resolution images with
        same width and height size. This class deals with the center crop and rescaling operations and gives
        FSNet the transformation parameters to correctly sample along the epipolar lines
        Args:
            im_size: desired image size
            device: device index
        """
    def __init__(self, im_size, device):
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.im_size = im_size
        self.device = device

        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])
        self.mean = torch.as_tensor(mean_vector, dtype=torch.float32)
        self.std = torch.as_tensor(std_vector, dtype=torch.float32)

    def center_crop_and_resize(self, im, out_size):
        """
            Auxiliary function to center crop and resize to desired output resolution. We need to know the
            displacement as well as the scale to rectify epipolar lines
            Args:
                im: input image as a PyTorch tensor, shape [1x3xHxW]
                output_size: size tuple of the output images
            Output:
                im: output image as a PyTorch tensor, shape [1 x 3 x out_size x out_size]
                disp: Displacement on the feature map due to center cropping. Size: [1, 2].
                scale_factor: Scaling factor on the image before feature extraction. Size: [1, 2].
        """

        b, c, h, w = im.shape
        crop_to = min([h, w])

        disp = [(w - crop_to) // 2, (h - crop_to) // 2]  # crop to square images

        im = im[:, :, disp[1]:disp[1] + crop_to, disp[0]:disp[0] + crop_to]
        im = transforms.Resize(out_size)(im)

        disp = torch.tensor([disp[0], disp[1]]).float().unsqueeze(0)
        scale_factor = [out_size / crop_to, out_size / crop_to]  # scale factor to desired output size
        scale_factor = torch.tensor([scale_factor[0], scale_factor[1]]).float().unsqueeze(0)

        return im, disp, scale_factor


    def read_im(self, path_im):
        """
            Auxiliary function to read, center-crop and normalize the image.
            Args:
                path_im: path to the image to read
            Output:
                im: output image as a PyTorch tensor, shape [1 x 3 x out_size x out_size]
                disp: Displacement on the feature map due to center cropping. Size: [1, 2].
                scale_factor: Scaling factor on the image before feature extraction. Size: [1, 2].
        """

        image = self.loader(path_im)

        image = self.to_tensor(image).unsqueeze(0)
        image, disp, scale = self.center_crop_and_resize(image, self.im_size)

        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])

        return {'image': image.to(self.device), 'disp': disp.to(self.device), 'scale': scale.to(self.device)}

    def prepare_im_pair(self, path_src, path_dst):
        """
            Auxiliary function to read, center-crop and normalize the image pair. Returned dictionary is directly
            the input to FSNet architecture.
        """
        data_src = self.read_im(path_src)
        data_dst = self.read_im(path_dst)
        return {'src_img': data_src['image'], 'dst_img': data_dst['image'],
                'disp_src': data_src['disp'], 'disp_dst': data_dst['disp'],
                'scale_src': data_src['scale'], 'scale_dst': data_dst['scale']}

def plot_epipolar_lines(path_src, path_dst, F_mat, path_epi_fig):
    """
        Auxiliary function to plot the corresponding epipolar lines to the selected fundamental matrix. Hopefully,
        such script can help to inspect visually the selection of fundamental matrices by FSNet.
        Args:
            path_src: path to the source image
            path_dst: path to the destination image
            F_mat: selected fundamental matrix, shape 3x3
            path_epi_fig: path to save the generated image with the epipolar lines
    """
    im_src = cv2.imread(path_src)
    im_dst = cv2.imread(path_dst)

    shape_src = im_src.shape
    shape_dst = im_dst.shape

    im_size_sampling = max(shape_src)
    sampling_pos = define_sampling_grid(im_size_sampling, feats_downsample=1, step=200) + 100

    epi_lines = cv2.computeCorrespondEpilines(sampling_pos, 1, F_mat)

    new_im = 255*np.ones((max(shape_src[0], shape_dst[0]), shape_src[1] + shape_dst[1] + 50, 3))

    for idx in range(len(sampling_pos)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        point_id = sampling_pos[idx]
        im_src = cv2.circle(im_src, point_id, 16, color, -1)

        x0_np, y0_np = map(int, [0., -(epi_lines[idx, 0, 2]) / epi_lines[idx, 0, 1]])
        x1_np, y1_np = map(int, [im_size_sampling, -(epi_lines[idx, 0, 2] + epi_lines[idx, 0, 0] * im_size_sampling) / epi_lines[idx, 0, 1]])
        im_dst = cv2.line(im_dst, (int(x0_np), int(y0_np)), (int(x1_np), int(y1_np)), color=color, thickness=4)

    diff_1 = (new_im.shape[0] - shape_src[0])//2
    diff_2 = (new_im.shape[0] - shape_dst[0])//2
    new_im[diff_1:diff_1+shape_src[0], :shape_src[1], :] = im_src
    new_im[diff_2:diff_2+shape_dst[0], shape_src[1]+50:, :] = im_dst

    cv2.imwrite(path_epi_fig, new_im)