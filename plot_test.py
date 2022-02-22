import open3d as o3d
import numpy as np
import cv2
import os
from skimage.io import imread, imshow, imread_collection, concatenate_images

id = '1621396080.4505239'
pcd = o3d.io.read_point_cloud('results/' + id + '/1.ply')
pcd2 = o3d.io.read_point_cloud('results/' + id + '/2.ply')


depth_xyz_1 = np.load(os.path.join('results/1620115290.678698/', 'positions_1.npy'))
depth_xyz_2 = np.load(os.path.join('results/1620115290.678698/', 'positions_2.npy'))

img_1 = imread(os.path.join('results/1620115290.678698/', 'raw_1.jpg'))
img_1 = img_1 / 255.0
img_2 = imread(os.path.join('results/1620115290.678698/', 'raw_2.jpg'))
img_2 = img_2 / 255.0
    

def generate_pcd(depth_xyz):
        region_depth = depth_xyz
        xyz = []
        for i in range(region_depth.shape[0]):
            for j in range(region_depth.shape[1]):
                x, y, z = region_depth[i, j, :]
                xyz.append([y, -x, -z])

        xyz = np.asarray(xyz)
        return xyz


points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)


print(np.mean(points, axis=0))
points2 = np.asarray(pcd2.points)
colors2 = np.asarray(pcd2.colors)

R = np.array([[ 0.99422582, -0.08558915, 0.06472649],
 [ 0.10410368,  0.62301318, -0.77525286],
 [ 0.02602778,  0.77751468,  0.62832594]])
t = np.array([[-0.01888087],
 [ 0.88467501],
 [ 0.20701791]])

# Transform for Point cloud
# theta1 = np.pi/180*1         
# theta2 = np.pi/180*50
# theta3 = np.pi/180*1

# R1 = np.array([[ 1, 0, 0],
#  [ 0,  np.cos(theta1), -np.sin(theta1)],
#  [ 0,  np.sin(theta1),  np.cos(theta1)]])

# R2 = np.array([[ np.cos(theta2), 0, np.sin(theta2)],
#  [ 0,  1, 0],
#  [ -np.sin(theta2),  0,  np.cos(theta2)]])

# R3 = np.array([[ np.cos(theta3), -np.sin(theta3), 0],
#  [ np.sin(theta3),  np.cos(theta3), 0],
#  [ 0,  0,  1]])

# R = np.dot(R1, R2)
# R = np.dot(R, R3)

# t = np.array([[0.8008087],
#  [ -0.0108467501],
#  [ -0.31701791]])
# R = np.array([[ 0.80192037,  0.06696918,  0.5936656 ],
#  [ 0.07742565,  0.97366755, -0.21442193],
#  [-0.59239259,  0.21791426,  0.77561872]])
# t = np.array([[0.6010362378],
#  [  -0.685168533],
#  [ 0.1534787108]])


# Transform for positions get_xyz()
theta1 = np.pi/180*1         
theta2 = np.pi/180*50
theta3 = np.pi/180*1

R1 = np.array([[ 1, 0, 0],
 [ 0,  np.cos(theta1), -np.sin(theta1)],
 [ 0,  np.sin(theta1),  np.cos(theta1)]])

R2 = np.array([[ np.cos(theta2), 0, np.sin(theta2)],
 [ 0,  1, 0],
 [ -np.sin(theta2),  0,  np.cos(theta2)]])

R3 = np.array([[ np.cos(theta3), -np.sin(theta3), 0],
 [ np.sin(theta3),  np.cos(theta3), 0],
 [ 0,  0,  1]])

R = np.dot(R1, R2)
R = np.dot(R, R3)


t = np.array([[0.8008087],
 [ -0.0108467501],
 [ -0.31701791]])

new_left = (R@points2.T) + t
new_left = new_left.T


# xyz_left = generate_pcd(depth_xyz_1)
# color_1 = img_1.reshape((xyz_left.shape[0], 3))

# xyz_right = generate_pcd(depth_xyz_2)
# color_2 = img_2.reshape((xyz_right.shape[0], 3))

# t2 = np.array([[-0.1],
#  [ -0.1],
#  [ -0.0]])

# xyz_left = xyz_left.T + t2
# xyz_left = xyz_left.T

# xyz_right = xyz_right.T + t2
# xyz_right = xyz_right.T


# xyz_right = (R@xyz_right.T) + t
# xyz_right = xyz_right.T

points = np.vstack((points, new_left))
colors = np.vstack((colors, colors2))
colors[:, [0, 2]] = colors[:, [2, 0]]

# points = np.vstack((points, xyz_left))
# colors = np.vstack((colors, color_1))

# points = np.vstack((points, xyz_right))
# colors = np.vstack((colors, color_2))


# points = xyz_left
# colors = color_1

# points = np.vstack((points, xyz_right))
# colors = np.vstack((colors, color_2))


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])

    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0,0,0])
    #  # Draw a point on 3D coordinates: coordinate point [x,y,z] corresponds to R, G, B colors

    # mesh_r = copy.deepcopy(axis_pcd)
    # R = axis_pcd.get_rotation_matrix_from_xyz((0,np.pi,0))
    # mesh_r.rotate(R, center=(0,0,0))
    # # o3d.visualization.draw_geometries([mesh, mesh_r])

    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window()

    # # viewer.add_geometry(axis_pcd)
    # # viewer.add_geometry(mesh_r)
    # viewer.add_geometry(pcd)

    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = False
    # opt.background_color = np.asarray([1, 1, 1])
    # viewer.run()
