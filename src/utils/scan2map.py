import open3d as o3d
import numpy as np
import os
import copy

def read_display_bin_pc(path):
    points = np.fromfile(path,dtype=np.float32).reshape(-1,4)
    points = points[:,:3]#open3d 只需xyz 与pcl不同
 
    #将array格式的点云转换为open3d的点云格式,若直接用open3d打开的点云数据，则不用转换
    pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
    pcd.points = o3d.utility.Vector3dVector(points)#转换格式

    return pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# 函数：点云配准 (ICP)
def icp_registration(source, target, threshold, trans_init):
    evaluation = o3d.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    # 使用ICP进行点云配准
    icp_result = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    
    print(icp_result)
    draw_registration_result(source, target, icp_result.transformation)

    return icp_result.transformation


# 函数：读取并下采样点云
def load_and_preprocess_pcd(pcd_file, voxel_size=0.05):
    # 读取点云
    # pcd = o3d.io.read_point_cloud(pcd_file)
    pcd = read_display_bin_pc(pcd_file)
    # 下采样点云，减小点数，提高配准速度
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    return pcd_downsampled


# 函数：更新地图点云
def update_map(map_cloud, new_cloud, transformation):
    # 使用变换矩阵将新点云转换到地图坐标系下
    new_cloud.transform(transformation)
    # 将新点云加入到地图点云中
    map_cloud += new_cloud
    # 进行体素下采样，避免地图点云过大
    map_cloud = map_cloud.voxel_down_sample(voxel_size=0.05)
    return map_cloud


# 函数：计算位姿变换中的平移和旋转变化
def compute_pose_change(transformation):
    # 提取旋转和平移部分
    rotation_matrix = transformation[:3, :3]
    translation_vector = transformation[:3, 3]

    # 计算平移的L2范数
    translation_magnitude = np.linalg.norm(translation_vector)

    # 计算旋转的角度变化（通过旋转矩阵转换为旋转轴和角度）
    rotation_angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)

    return translation_magnitude, rotation_angle



# 设定参数
threshold = 2.0  # ICP 配准时的距离阈值
voxel_size = 0.08  # 点云下采样的体素大小
translation_threshold = 3.0  # 平移变化阈值（单位：米）
rotation_threshold = 0.5     # 旋转变化阈值（单位：弧度）


# 读取存放点云文件的文件夹
pcd_folder = "/home/thinking/lbh/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/test/"
pcd_files = sorted([os.path.join(pcd_folder, f) for f in os.listdir(pcd_folder) if f.endswith(".bin")])


# 初始化：第一帧作为地图的起点
map_cloud = load_and_preprocess_pcd(pcd_files[0], voxel_size)
last_keyframe_transformation = np.eye(4)  # 第一个关键帧的全局位姿

# 初始位姿矩阵
trans_init = np.eye(4)
global_transformation = np.eye(4)  # 全局位姿（相对于第一帧的位姿）

# 记录上两次的位姿矩阵，用于计算匀速模型的速度
last_transformation = np.eye(4)  # 上一次的位姿变换
prev_transformation = np.eye(4)  # 上上一次的位姿变换


# 逐帧配准
for i in range(1, len(pcd_files)):
    # 读取当前帧点云（scan）
    scan_file = pcd_files[i]
    scan_cloud = load_and_preprocess_pcd(scan_file, voxel_size)

    # 使用匀速模型作为初始值：trans_init = last_transformation + (last_transformation - prev_transformation)
    velocity = np.dot(last_transformation, np.linalg.inv(prev_transformation))  # 计算速度（两次位姿差）
    trans_init = np.dot(last_transformation, velocity)  # 预测下一帧的初始位姿
    # 对当前帧(scan)与地图(map)进行ICP配准
    transformation = icp_registration(map_cloud, scan_cloud, threshold, trans_init)
    # 更新全局位姿：累积每一帧的相对位姿变换
    global_transformation = np.dot(global_transformation, transformation)
    # 计算当前帧与上一个关键帧之间的位姿变化
    translation_change, rotation_change = compute_pose_change(np.dot(np.linalg.inv(last_keyframe_transformation), global_transformation))
    print(translation_change, rotation_change)
    # 判断是否超过位姿变化阈值，决定是否更新关键帧
    if translation_change > translation_threshold or rotation_change > rotation_threshold:
        print(f"Adding frame {i} as keyframe.")
        # 更新地图：将当前帧点云转化到地图坐标系下，并加入地图
        map_cloud = update_map(map_cloud, scan_cloud, global_transformation)
        # 更新关键帧位姿
        last_keyframe_transformation = global_transformation


    # 更新上两次的位姿变换，用于下一次迭代
    prev_transformation = last_transformation  # 保存为上上次位姿
    last_transformation = transformation  # 保存为上次位姿
    print(f"Transformation for frame {i}: \n{global_transformation}", '\n')



# 保存最终的地图点云为 PCD 文件
o3d.io.write_point_cloud("final_map.pcd", map_cloud)