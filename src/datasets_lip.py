import os, sys, traceback
import random
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
import cv2
from torch.utils.data import Dataset
import pandas as pd
from src.data.augmentation import AllAugmentationTransform
import glob
from src.utils.cropper import Cropper
from src.data.data_utils import CustVideoReader


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name): # 读取目录下的所有文件
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):   # 直接读取视频文件
        # todo: resize
        video = np.array(mimread(name, memtest=False))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        # video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, crop_cfg, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.crop_cfg = crop_cfg
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg, flag_force_cpu=crop_cfg.flag_force_cpu)

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                # train_videos = {os.path.basename(video).split('#')[0] for video in      # idxxxxx
                #                 os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = []
                for root, dirs, files in os.walk(os.path.join(root_dir, 'train')):
                    if not dirs and files:
                        train_videos.append(root)   # 包含图片和视频的所有目录
            else:
                # train_videos = os.listdir(os.path.join(root_dir, 'train'))
                train_videos = []
                for root, dirs, files in os.walk(os.path.join(root_dir, 'train')):
                    if not dirs and files:
                        if files[0].endswith('.png') or files[0].endswith('.jpg'):
                            train_videos.append(root)   # 包含图片的所有目录
                        else:
                            train_videos += [os.path.join(root, file) for file in files]       # 所有mp4文件
            print('num of training videos: ', len(train_videos))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')    # VoxCeleb/first-order-256/train/
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try_cnt = 0
        path = None
        while True:
            try:
                try_cnt += 1
                if try_cnt > 10:
                    break
                if self.is_train and self.id_sampling:
                    name = self.videos[idx]
                    path = np.random.choice(glob.glob(os.path.join(name, '*.mp4')))     # VoxCeleb/first-order-256/train/idxxxx/¥#&*¥@/*.mp4
                else:
                    path = self.videos[idx] # 图片目录或者mp4文件

                video_name = os.path.basename(path)

                if self.is_train and os.path.isdir(path):   #图片目录
                    frames = os.listdir(path)
                    num_frames = len(frames)
                    frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))     # 从图片目录里任选两帧
                    # video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx] # 读取两张图片
                    video_array = [io.imread(os.path.join(path, frames[idx])) for idx in frame_idx] # 读取两张图片

                else:   # mp4文件
                    video_reader = CustVideoReader(path)
                    num_frames = len(video_reader)
                    assert num_frames > 0, f"get_pixel_values_from_video: {path} video length is 0"

                    # video = np.array(mimread(path, memtest=False))
                    # if len(video.shape) == 3:
                    #     video = np.array([gray2rgb(frame) for frame in video])
                    # if video.shape[-1] == 4:
                    #     video = video[..., :3]
                    # num_frames = len(video)
                    # video_array = video[frame_idx]
                    frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames) # 训练时选两张，测试选所有
                    video_array = video_reader.get_batch(frame_idx)     # rgb

                # scale变成随机
                self.crop_cfg.scale = random.uniform(2.2, 2.8)
                self.crop_cfg.flag_do_rot = random.choice([True, False])
                crop_info_source = self.cropper.crop_source_image(video_array[0], self.crop_cfg)
                source_lmk = crop_info_source['lmk_crop']   # (203, 2) 对应原图的
                source_img_crop_256 = crop_info_source['img_crop_256x256']

                source_lmk_h = np.hstack([source_lmk, np.ones((source_lmk.shape[0], 1))])   # (203, 3)
                source_lmk_crop = (source_lmk_h @ crop_info_source['M_o2c'].T)[:,:2]     # # convert lmk from o2c : (203,2) 对应crop完512的
                source_lmk_crop_256 = source_lmk_crop / 2.    # 转到256空间，source_img_crop_256对齐

                scale_crop_driving_video = random.uniform(2.2, self.crop_cfg.scale)
                ret_d = self.cropper.crop_driving_video([video_array[1]],
                                                        scale_crop_driving_video=scale_crop_driving_video)
                driving_img_crop, driving_lmk_crop = ret_d['frame_crop_lst'][0], ret_d['lmk_crop_lst'][0]   # 对应crop完512的
                driving_img_crop_256 = cv2.resize(driving_img_crop, (256, 256)) # (256,256,3)
                driving_lmk_crop_256 = driving_lmk_crop / 2.  # 转到256空间

                if self.transform is not None:
                    clips = [source_img_crop_256, driving_img_crop, driving_img_crop_256]
                    transformed_clip, transformed_lmk = self.transform([img_as_float32(clip) for clip in clips], [source_lmk_crop_256, driving_lmk_crop_256])

                out = {}
                if self.is_train:

                    # 归一化，# (h,w,3) ---> (3,h,w)
                    out['source'] = transformed_clip[0].transpose((2, 0, 1))
                    out['driving_512'] = transformed_clip[1].transpose((2, 0, 1))
                    out['driving'] = transformed_clip[2].transpose((2, 0, 1))

                    out['source_lmk'] = transformed_lmk[0]
                    out['driving_lmk'] = transformed_lmk[1]

                else:   #  val and test 需要改
                    video = np.array(video_array, dtype='float32')
                    out['video'] = video.transpose((3, 0, 1, 2))

                out['name'] = video_name

                return out
            except Exception as e:
                print(f"read idx:{idx}, {path} error, try_time:{try_cnt}, {type(e).__name__}: {e}")
                print(traceback.format_exc())
                idx = random.randint(0,  len(self.videos) - 1)


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from src.config.crop_config import CropConfig
    model_config = OmegaConf.load("config/stage1/training_local_stage1.yaml")
    print(model_config['dataset_params'])
    dataset = FramesDataset(is_train=True, crop_cfg=CropConfig, **model_config['dataset_params'])
    print(dataset.__len__())

    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        import cv2
        source = cv2.cvtColor(np.array(data['source'].transpose((1, 2, 0)) * 255, np.uint8), cv2.COLOR_RGB2BGR)
        driving = cv2.cvtColor(np.array(data['driving'].transpose((1, 2, 0)) * 255, np.uint8), cv2.COLOR_RGB2BGR)
        driving_512 = cv2.cvtColor(np.array(data['driving_512'].transpose((1, 2, 0)) * 255, np.uint8), cv2.COLOR_RGB2BGR)
        driving_512 = cv2.resize(driving_512, (256, 256))
        driving_lmk = data['driving_lmk']
        print(source.shape, driving_lmk.shape)
        # 绘制关键点
        # 108~144 轮廓点
        # 0~23 左眼睛
        # 24~47 右眼睛
        # 48~84 嘴巴外部
        # 85~107 嘴巴内唇
        # 145～164 左眉毛
        # 165～184 右眉毛
        # 185～196 鼻子外围
        # 197 左眼球
        # 198 右眼球
        # 199 两眼中间
        # 200 鼻中
        # 201 鼻尖
        # 202 两鼻孔之间
        for idx, (x, y) in enumerate(driving_lmk):
            # 绘制圆（关键点）
            if 0 <= idx < 203:
                cv2.circle(driving, (int(x), int(y)), radius=1, color=(0, 255, 0), thickness=-1)  # 绿色，填充模式

            # 添加文本标签（如果需要）
            # cv2.putText(driving, f'{idx}', (int(x) + 1, int(y)+1), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 0)
        cv2.imshow('source', np.hstack([source, driving, driving_512]))
        cv2.waitKey(0)








