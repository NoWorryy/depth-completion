import torch.utils.data
import data_utils
import traceback
import json
import random
import numpy as np
from torchvision import transforms
from PIL import Image


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
                 train_image_path,
                 train_sparse_depth_path,
                 train_intrinsics_path,
                 shape=None,
                 RandCrop=False):
        
        # Read paths for training
        self.image_paths = data_utils.read_paths(train_image_path)     # kitti_train_image-clean.txt
        self.sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)
        self.intrinsics_paths = data_utils.read_paths(train_intrinsics_path)

        self.shape = shape
        self.RandCrop = RandCrop
        self.transform = transforms.ToTensor()


    def __getitem__(self, index):
        try_cnt = 0
        while True:
            try:
                try_cnt += 1
                if try_cnt > 10:
                    break

                # Load image
                images = Image.open(self.image_paths[index]).convert('RGB')
                images = np.array(images)   # (h, 3w, c)
                image1, image0, image2 = np.split(images, indices_or_sections=3, axis=1)   # (h, w, c)

                # Load depth
                z = np.array(Image.open(self.sparse_depth_paths[index]), dtype=np.float32)
                z = z / 256.0   # Assert 16-bit (not 8-bit) depth map
                z[z <= 0] = 0.0
                sparse_depth0 = np.expand_dims(z, axis=-1)  # (h,w,c)


                # Load camera intrinsics
                intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

                # Crop image, depth and adjust intrinsics
                if self.RandCrop:
                    [image0, image1, image2, sparse_depth0], intrinsics = random_crop(
                        inputs=[image0, image1, image2, sparse_depth0],
                        shape=self.shape,
                        intrinsics=intrinsics,
                        RandCrop=self.RandCrop)
                    
                inputs = {
                    'image0': self.transform(image0),   # 0~1 （）
                    'image1': self.transform(image1),
                    'image2': self.transform(image1),
                    'sparse_depth0': self.transform(sparse_depth0), # 真实值
                    'intrinsics': intrinsics.astype(np.float32)
                }

                return inputs
            except Exception as e:
                print(f"read idx:{index}, {self.image_paths[index]} error, try_time:{try_cnt}, {type(e).__name__}: {e}")
                print(traceback.format_exc())
                index = random.randint(0,  len(self.image_paths) - 1)

    def __len__(self):
        return len(self.image_paths)


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
            self.dataset = json.load(file)

        self.min_evaluate_depth = min_evaluate_depth
        self.max_evaluate_depth = max_evaluate_depth
        self.transform = transforms.ToTensor()


    def __getitem__(self, index):
        entry = self.dataset[index]

        # Load image
        image = Image.open(entry['image']).convert('RGB')
        image = np.array(image)   # (h, w, c) 整型

        # Load depth
        z = np.array(Image.open(entry['sparse_depth']), dtype=np.float32)
        z = z / 256.0   # Assert 16-bit (not 8-bit) depth map
        z[z <= 0] = 0.0
        sparse_depth = np.expand_dims(z, axis=-1)  # (h,w,c)
        v = z.astype(np.float32)
        v[z > 0] = 1.0
        validity_map_depth = np.expand_dims(v, axis=-1)  # (h,w,c)

        # Load camera intrinsics
        intrinsics = np.reshape(np.loadtxt(entry['intrinsic']), (3, 3))

        # load gt
        z = np.array(Image.open(entry['gt']), dtype=np.float32)
        z = z / 256.0   # Assert 16-bit (not 8-bit) depth map
        z[z <= 0] = 0.0
        ground_truth = np.expand_dims(z, axis=-1)  # (h,w,c)
        v = z.astype(np.float32)
        v[z > 0] = 1.0
        validity_map_gt = np.expand_dims(v, axis=-1)  # (h,w,c)

        gt_mask = (ground_truth > self.min_evaluate_depth) & (ground_truth < self.max_evaluate_depth) & validity_map_gt.astype(bool)

        inputs = {
            'image': self.transform(image),
            'sparse_depth': self.transform(sparse_depth),
            'intrinsics': intrinsics.astype(np.float32),
            'validity_map_depth': self.transform(validity_map_depth),
            'ground_truth': self.transform(ground_truth),
            'gt_mask': gt_mask.transpose(2, 0, 1)
        }

        return inputs

    def __len__(self):
        return len(self.dataset)
        # return len(self.image_paths)
    

    
if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg_path = '/media/data2/libihan/codes/calibrated-backprojection-network/configs/kitti_train.yaml'

    args = OmegaConf.load(cfg_path)
    print(args['dataset_train_params'])
    
    # dataset = KBNetTrainingDataset(**args['dataset_train_params'])
    dataset = KBNetInferenceDataset(**args['dataset_val_params'])
    print(len(dataset))

    for i in range(len(dataset)):
        inputs = dataset[i]
        