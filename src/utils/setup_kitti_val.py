import os
import json

# 设定根目录和子目录
root_directory = 'data/kitti_depth_completion/validation'  # 请替换成你的根目录
dir_gt = os.path.join(root_directory, 'ground_truth')
dir_img = os.path.join(root_directory, 'image')
dir_int = os.path.join(root_directory, 'intrinsics')
dir2_sp = os.path.join(root_directory, 'sparse_depth')

# 获取dir1中的图片文件和dir2中的txt文件
gt_files = [f for f in os.listdir(dir_gt) if f.endswith('.png')]
img_files = [f.replace('groundtruth_depth', 'image') for f in gt_files]
int_files = [f.replace('groundtruth_depth', 'image').replace('.png', '.txt') for f in gt_files]
sp_files = [f.replace('groundtruth_depth', 'velodyne_raw') for f in gt_files]

# 确保文件数量相同
if len(gt_files) != len(img_files) or len(gt_files) != len(int_files) or len(gt_files) != len(sp_files):
    print("数量不匹配！")
else:
    # 创建一个字典来存储路径对
    path_pairs = []

    for gt, img, intrinsic, sp in zip(gt_files, img_files, int_files, sp_files):
        gt_path = os.path.join(dir_gt, gt)
        img_path = os.path.join(dir_img, img)
        int_path = os.path.join(dir_int, intrinsic)
        sp_path = os.path.join(dir2_sp, sp)
        path_pairs.append({'gt': gt_path,'image': img_path, 'intrinsic': int_path, 'sparse_depth': sp_path})

    # 将路径对写入JSON文件
    json_file_path = os.path.join('/media/data2/libihan/codes/calibrated-backprojection-network/validation', 'val_filepath.json')  # 配置文件路径
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(path_pairs, json_file, ensure_ascii=False, indent=4)

    print(f"已创建配置文件：{json_file_path}")
