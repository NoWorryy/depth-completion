from PIL import Image
import torch.utils.data
import data_utils
import traceback
import json
import random
import numpy as np
from torchvision import transforms
import depth_map_utils


def random_crop(inputs, shape, intrinsics=None, RandCrop = False, tp_min=50):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : numpy[float32]
            3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        numpy[float32] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width, _ = shape
    o_height, o_width, _ = inputs[0].shape

    # bottom center crop
    tp = o_height - n_height
    lp = (o_width - n_width) // 2

    if RandCrop:
        tp = np.random.randint(tp_min, tp)
        lp = np.random.randint(0, o_width - n_width)

    # Crop each input into (n_height, n_width)
    tp_end = tp + n_height
    lp_end = lp + n_width
    outputs = [
        T[tp:tp_end, lp:lp_end, :] for T in inputs
    ]

    if intrinsics is not None:
        # Adjust intrinsics
        intrinsics = intrinsics + [[0.0, 0.0, -lp],
                                   [0.0, 0.0, -tp],
                                   [0.0, 0.0, 0.0]]

        return outputs, intrinsics
    else:
        return outputs


class KBNetTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image at time t-1, t, and t+1
        (2) sparse depth
        (3) camera intrinsics matrix

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
        shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 train_file_path,
                 shape=None,
                 RandCrop=False):

        self.shape = shape
        self.RandCrop = RandCrop
        self.transform = transforms.ToTensor()

        with open(train_file_path, 'r') as file:
            self.dataset = json.load(file)['train']

    
    def __getitem__(self, index):
        try_cnt = 0
        while True:
            try:
                try_cnt += 1
                if try_cnt > 10:
                    break
                
                entry = self.dataset[index]
                # Load image
                image = np.array(Image.open(entry['image']).convert('RGB'))   # (h, w, c)

                # Load depth
                z = np.array(Image.open(entry['sparse_depth']), dtype=np.float32) / 256.0   # Assert 16-bit (not 8-bit) depth map
                z[z <= 0.0001] = 0.0

                prefill_depth = depth_map_utils.fill_in_fast(
                    z, extrapolate=True, blur_type='gaussian')

                d_clear, _ = data_utils.outlier_removal(z)

                
                prefill_depth[prefill_depth <=0] = 0.0
                sparse_depth = np.expand_dims(z, axis=-1)  # (h,w,c)
                prefill_depth = np.expand_dims(prefill_depth, axis=-1)  # (h,w,c)
                d_clear = np.expand_dims(d_clear, axis=-1)  # (h,w,c)
              

                # Load gt
                gt = np.array(Image.open(entry['gt']), dtype=np.float32) / 256.0   # Assert 16-bit (not 8-bit) depth map
                gt[gt <= 0] = 0.0
                gt = np.expand_dims(gt, axis=-1)  # (h,w,c)

                # Load camera intrinsics
                intrinsics = None
                # intrinsics = np.load(entry['intrinsic']).astype(np.float32)

                # Crop image, depth and adjust intrinsics
                [image, sparse_depth, d_clear, prefill_depth, gt] = random_crop(
                    inputs=[image, sparse_depth, d_clear, prefill_depth, gt],
                    shape=self.shape,
                    intrinsics=intrinsics,
                    RandCrop=self.RandCrop)
                    
                inputs = {
                    'image': self.transform(image),   # 0~1 （）
                    'sparse_depth': self.transform(sparse_depth), # 真实值
                    'd_clear': self.transform(d_clear),
                    'prefill_depth': self.transform(prefill_depth),
                    # 'intrinsics': intrinsics.astype(np.float32),
                    'gt': self.transform(gt)
                }

                return inputs
            
            except Exception as e:
                print(f"read idx:{index}, {self.dataset[index]} error, try_time:{try_cnt}, {type(e).__name__}: {e}")
                print(traceback.format_exc())
                index = random.randint(0,  len(self.dataset) - 1)

    def __len__(self):
        return len(self.dataset)


class KBNetInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) camera intrinsics matrix

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
    '''

    def __init__(self,
                 val_file_path,
                 min_evaluate_depth,
                 max_evaluate_depth):

        # Read paths for training
        with open(val_file_path, 'r') as file:
            self.dataset = json.load(file)['test']

        self.min_evaluate_depth = min_evaluate_depth
        self.max_evaluate_depth = max_evaluate_depth
        self.transform = transforms.ToTensor()


    def __getitem__(self, index):
        entry = self.dataset[index]

        # Load image
        image = np.array(Image.open(entry['image']).convert('RGB'))   # (h, w, c) 整型

        # Load depth
        z = np.array(Image.open(entry['sparse_depth']), dtype=np.float32) / 256.0   # Assert 16-bit (not 8-bit) depth map
        z[z <= 0.0001] = 0.0

        prefill_depth = depth_map_utils.fill_in_fast(
                    z, extrapolate=True, blur_type='gaussian')

        d_clear, _ = data_utils.outlier_removal(z)

        prefill_depth[prefill_depth <=0] = 0.0
        sparse_depth = np.expand_dims(z, axis=-1)  # (h,w,c)
        prefill_depth = np.expand_dims(prefill_depth, axis=-1)  # (h,w,c)
        d_clear = np.expand_dims(d_clear, axis=-1)  # (h,w,c)

        # Load camera intrinsics
        intrinsics = np.reshape(np.loadtxt(entry['intrinsic']), (3, 3))

        # load gt
        z = np.array(Image.open(entry['gt']), dtype=np.float32)
        z = z / 256.0   # Assert 16-bit (not 8-bit) depth map
        z[z <= 0] = 0.0
        ground_truth = np.expand_dims(z, axis=-1)  # (h,w,c)

        inputs = {
            'image': self.transform(image),
            'sparse_depth': self.transform(sparse_depth),
            'd_clear': self.transform(d_clear),
            'prefill_depth': self.transform(prefill_depth),
            'intrinsics': intrinsics.astype(np.float32),
            'gt': self.transform(ground_truth),
        }

        return inputs

    def __len__(self):
        return len(self.dataset)
        # return len(self.image_paths)
    

    
if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg_path = '/home/sbq/codes/depth-completion/configs/kitti_train.yaml'

    args = OmegaConf.load(cfg_path)
    print(args['dataset_val_params'])
    
    dataset = KBNetTrainingDataset(**args['dataset_train_params'])
    # dataset = KBNetInferenceDataset(**args['dataset_val_params'])
    print(len(dataset))

    for i in range(len(dataset)):
        inputs = dataset[i]
        if i == 2:
            break