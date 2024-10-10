import open3d as o3d
import numpy as np
import os
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# 函数：点云配准 (ICP)
def icp_registration(source, target, threshold, trans_init):
    # 使用ICP进行点云配准
    icp_result = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())

    print(icp_result)
    draw_registration_result(source, target, icp_result.transformation)

    return icp_result.transformation

def read_display_bin_pc(path):
    points = np.fromfile(path,dtype=np.float32).reshape(-1,4)
    points = points[:,:3]#open3d 只需xyz 与pcl不同
 
    #将array格式的点云转换为open3d的点云格式,若直接用open3d打开的点云数据，则不用转换
    pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
    pcd.points = o3d.utility.Vector3dVector(points)#转换格式

    return pcd


# 函数：读取并下采样点云
def load_and_preprocess_pcd(pcd_file, voxel_size=0.05):
    # 读取点云
    # pcd = o3d.io.read_point_cloud(pcd_file)
    pcd = read_display_bin_pc(pcd_file)
    # 下采样点云，减小点数，提高配准速度
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    return pcd_downsampled



# 设定阈值
threshold = 2.0  # ICP 配准时的距离阈值
voxel_size = 0.08  # 点云下采样的体素大小
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

# 读取存放点云文件的文件夹
pcd_folder = "/media/data2/libihan/codes/calibrated-backprojection-network/data/vim_raw_data/2011_10_03/2011_10_03_drive_0042_sync/velodyne_points/data"
pcd_files = sorted([os.path.join(pcd_folder, f) for f in os.listdir(pcd_folder) if f.endswith(".bin")])


# 存储相邻帧的变换矩阵
transformations = []


# 初始化trans_init为单位矩阵
trans_init = np.eye(4)


# 逐帧配准
for i in range(len(pcd_files) - 1):
    # trans_init = np.linalg.inv(trans_init)

    source_file = pcd_files[i]
    target_file = pcd_files[i + 1]

    # 加载并预处理点云
    source_pcd = load_and_preprocess_pcd(source_file, voxel_size)
    target_pcd = load_and_preprocess_pcd(target_file, voxel_size)

    target_pcd.paint_uniform_color([1.0, 0, 0.0])
    source_pcd.paint_uniform_color([0.0, 0, 1.0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.add_geometry(source_pcd)
    vis.add_geometry(target_pcd)
    # vis.add_geometry(pts3)
    vis.add_geometry(axis_pcd)
    vis.run()
    vis.destroy_window()

    # 计算相邻帧的位姿变换矩阵，使用trans_init作为初始值
    transformation = icp_registration(source_pcd, target_pcd, threshold, trans_init)
    print(f"Transformation between {os.path.basename(source_file)} and {os.path.basename(target_file)}:\n", transformation)

    # 将变换矩阵保存到列表中
    transformations.append(transformation)

    # 使用当前的相对位姿变换作为下一次配准的初始值
    trans_init = transformation



# # 可选择将变换矩阵保存到文件
# np.save("transformations.npy", transformations)

