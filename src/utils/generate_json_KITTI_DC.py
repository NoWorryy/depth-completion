"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    This script generates a json file for the KITTI Depth Completion dataset.
"""

import os
import argparse
import random
import json

parser = argparse.ArgumentParser(
    description="KITTI Depth Completion jason generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to the KITTI Depth Completion dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../data_json', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='kitti_dc.json', help="Output file name")
parser.add_argument('--num_train', type=int, required=False,
                    default=int(1e10), help="Maximum number of train data")
parser.add_argument('--num_val', type=int, required=False,
                    default=int(1e10), help="Maximum number of val data")
parser.add_argument('--num_test', type=int, required=False,
                    default=int(1e10), help="Maximum number of test data")
parser.add_argument('--seed', type=int, required=False,
                    default=7240, help='Random seed')
parser.add_argument('--test_data', action='store_true',
                    default=False, help='json for DC test set generation')

args = parser.parse_args()

random.seed(args.seed)


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


def generate_json():
    os.makedirs(args.path_out, exist_ok=True)
    check_dir_existence(args.path_out)

    # For train/val splits
    dict_json = {}
    for split in ['train', 'val']:
        path_base =os.path.join(args.path_root, 'data_depth_annotated', split)

        list_seq = os.listdir(path_base)
        list_seq = [seq for seq in list_seq if seq.endswith('_sync')]
        list_seq.sort()

        list_pairs = []
        for seq in list_seq:    # 2011_09_26_drive_0001_sync
            cnt_seq = 0
            # e.g., 2011_09_26
            seq_data = seq[:10]

            raw_path = os.path.join(args.path_root, 'rawdata', seq_data, seq)
            if not os.path.exists(raw_path):
                print(raw_path, "not exist")
                continue

            for cam in ['image_02', 'image_03']:
                list_depth_path = os.path.join(path_base, seq, 'proj_depth/groundtruth', cam)
                list_depth = os.listdir(list_depth_path)
                list_depth = [dp for dp in list_depth if dp.endswith('.png')]
                list_depth.sort()


                for idx, name in enumerate(list_depth):
                    
                    path_gt = os.path.join(list_depth_path, name)

                    # name_base = os.path.splitext(name)[0]
                    # name_1 = int(name_base) - 1
                    # name_1 = f'{name_1:010d}.png'
                    # name_2 = int(name_base) + 1
                    # name_2 = f'{name_2:010d}.png'

                    path_rgb = os.path.join(raw_path, cam, 'data', name)
                    # path_rgb1 = os.path.join(raw_path, cam, 'data', name_1)
                    # path_rgb2 = os.path.join(raw_path, cam, 'data', name_2)

                    path_depth = os.path.join(args.path_root, 'data_depth_velodyne', split, seq, 'proj_depth/velodyne_raw', cam, name)

                    path_reldepth = os.path.join(args.path_root, 'rel_depth', seq_data, seq, cam, name)

                    if cam == 'image_02':
                        path_calib = os.path.join(args.path_root, 'data_intrinsics', seq_data, 'intrinsics2.npy')
                    else:
                        # path_calib = 'data_intrinsics/' + seq_data + 'intrinsics3.npy'
                        path_calib = os.path.join(args.path_root, 'data_intrinsics', seq_data, 'intrinsics3.npy')

                    dict_sample = {
                        'image': path_rgb,
                        # 'image1': path_rgb1,
                        # 'image2': path_rgb2,
                        'sparse_depth': path_depth,
                        'intrinsic': path_calib,
                        'gt': path_gt,
                        'rel_depth': path_reldepth
                    }

                    flag_valid = True
                    for val in dict_sample.values():
                        flag_valid &= os.path.exists(val)
                        if not flag_valid:
                            print(f'{val} not exists! ')
                            break

                    if not flag_valid:
                        continue

                    list_pairs.append(dict_sample)
                    cnt_seq += 1

            print("{} : {} samples".format(seq, cnt_seq))

        dict_json[split] = list_pairs
        print("{} split : Total {} samples".format(split, len(list_pairs)))

    # For test split
    split = 'test'
    path_base = os.path.join(args.path_root, 'data_depth_selection/depth_selection/val_selection_cropped')

    list_depth = os.listdir(os.path.join(path_base, 'velodyne_raw'))
    list_depth = [dp for dp in list_depth if dp.endswith('.png')]
    list_depth.sort()

    list_pairs = []
    for name in list_depth:

        path_rgb = os.path.join(path_base, 'image', name.replace('velodyne_raw', 'image'))
        path_depth = os.path.join(path_base, 'velodyne_raw', name)
        path_gt = os.path.join(path_base, 'groundtruth_depth', name.replace('velodyne_raw', 'groundtruth_depth'))
        path_calib = os.path.join(path_base, 'intrinsics', name.replace('velodyne_raw', 'image').replace('png', 'txt'))
        path_reldepth = os.path.join(path_base, 'rel_depth', name.replace('velodyne_raw', 'rel_depth').replace('image', 'rel_depth'))


        dict_sample = {
            'image': path_rgb,
            'sparse_depth': path_depth,
            'gt': path_gt,
            'intrinsic': path_calib,
            'rel_depth': path_reldepth
        }

        flag_valid = True
        for val in dict_sample.values():
            flag_valid &= os.path.exists(val)
            if not flag_valid:
                print(f'{val} not exists! ')
                break

        if not flag_valid:
            continue

        list_pairs.append(dict_sample)

    dict_json[split] = list_pairs
    print("{} split : Total {} samples".format(split, len(list_pairs)))

    # random.shuffle(dict_json['train'])

    # Cut if maximum is set
    # for s in [('train', args.num_train), ('val', args.num_val),
    #           ('test', args.num_test)]:
    #     if len(dict_json[s[0]]) > s[1]:
    #         # Do shuffle
    #         random.shuffle(dict_json[s[0]])

    #         num_orig = len(dict_json[s[0]])
    #         dict_json[s[0]] = dict_json[s[0]][0:s[1]]
    #         print("{} split : {} -> {}".format(s[0], num_orig,
    #                                            len(dict_json[s[0]])))

    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


def generate_json_test():
    check_dir_existence(args.path_out)

    dict_json = {}

    # For test split
    split = 'test'
    path_base = args.path_root \
                + '/data_depth_selection/test_depth_completion_anonymous'

    list_depth = os.listdir(path_base + '/velodyne_raw')
    list_depth.sort()

    list_pairs = []
    for name in list_depth:
        path_rgb = 'data_depth_selection/test_depth_completion_anonymous/image/' \
                   + name
        path_depth = \
            'data_depth_selection/test_depth_completion_anonymous/velodyne_raw/' \
            + name
        path_gt = path_depth
        path_calib = \
            'data_depth_selection/test_depth_completion_anonymous/intrinsics/' \
            + name[:-4] + '.txt'

        dict_sample = {
            'rgb': path_rgb,
            'depth': path_depth,
            'gt': path_gt,
            'K': path_calib
        }

        flag_valid = True
        for val in dict_sample.values():
            flag_valid &= os.path.exists(args.path_root + '/' + val)
            if not flag_valid:
                break

        if not flag_valid:
            continue

        list_pairs.append(dict_sample)

    dict_json[split] = list_pairs
    print("{} split : Total {} samples".format(split, len(list_pairs)))

    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))
    print('')

    if args.test_data:
        generate_json_test()
    else:
        generate_json()
