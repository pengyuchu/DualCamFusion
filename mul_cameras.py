import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import os
import time





# x1 = 340
# y1 = 340

# def click(event,x,y,flags,param):
#     global x1, y1
#     x1 = y
#     y1 = x

# Configure depth and color streams...
# ...from Camera 1
ctx = rs.context()
devices = ctx.query_devices()

pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('846112071067')
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('838212070239')
config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# Start streaming from both cameras

profile1 = pipeline_1.start(config_1)
profile2 = pipeline_2.start(config_2)

align_to = rs.stream.color
align = rs.align(align_to)


def get_position_matrix(depth, depth_intrinsic, depth_scale):
    xyz = np.zeros((depth.shape[0], depth.shape[1], 3), dtype = np.float32)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            point_location = rs.rs2_deproject_pixel_to_point(depth_intrinsic, [i, j],  depth[i, j] * depth_scale)
            xyz[i, j, :] = np.array([point_location[0], point_location[1], point_location[2]])


    return xyz


try:
    while True:

        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        frames_1 = align.process(frames_1)

        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        if not depth_frame_1 or not color_frame_1:
            continue
        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.5), cv2.COLORMAP_JET)
        depth_scale1 = profile1.get_device().first_depth_sensor().get_depth_scale()
        depth_intrinsics1 = depth_frame_1.profile.as_video_stream_profile().intrinsics
        # Camera 2
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        frames_2 = align.process(frames_2)

        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
        if not depth_frame_2 or not color_frame_2:
            continue
        # Convert images to numpy arrays
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2, alpha=0.5), cv2.COLORMAP_JET)
        depth_scale2 = profile2.get_device().first_depth_sensor().get_depth_scale()
        depth_intrinsics2 = depth_frame_2.profile.as_video_stream_profile().intrinsics
        # Stack all images horizontally
        # images = np.hstack((color_image_1, depth_colormap_1,color_image_2, depth_colormap_2))
        images = np.hstack((color_image_1,color_image_2))
        # point_location = rs.rs2_deproject_pixel_to_point(depth_intrinsics1, [x1, y1], depth_image_1[x1, y1]*depth_scale1)
        # print(point_location)
        # Show images from both cameras
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback('RealSense',click)
        cv2.imshow('RealSense', images)
        # Save images and depth maps from both cameras by pressing 's'


        ch = cv2.waitKey(1)
        if ch==ord('s'):
            base_dir = '/Users/chupengyu/Develop/PyCharm/MultiCam/results/'
            base_dir = base_dir+str(time.time())
            os.mkdir(base_dir)

            positions_1 = get_position_matrix(depth_image_1, depth_intrinsics1, depth_scale1)
            positions_2 = get_position_matrix(depth_image_2, depth_intrinsics2, depth_scale2)

            cv2.imwrite(os.path.join(base_dir, 'raw_1.jpg'),color_image_1)
            np.save(os.path.join(base_dir, 'raw_1.npy'),depth_image_1)
            np.save(os.path.join(base_dir, 'positions_1.npy'),positions_1)
            cv2.imwrite(os.path.join(base_dir, 'raw_2.jpg'),color_image_2)
            np.save(os.path.join(base_dir, 'raw_2.npy'),depth_image_2)
            np.save(os.path.join(base_dir, 'positions_2.npy'),positions_2)
            pc = rs.pointcloud();
            pc.map_to(color_frame_1);

            pointcloud = pc.calculate(depth_frame_1);
            pointcloud.export_to_ply(os.path.join(base_dir, '1.ply'), color_frame_1);
            pc = rs.pointcloud();
            pc.map_to(color_frame_2);
            pointcloud = pc.calculate(depth_frame_2);
            pointcloud.export_to_ply(os.path.join(base_dir, '2.ply'), color_frame_2);
            # cloud = PyntCloud.from_file("1.ply");
            # cloud.plot()
        elif ch == ord('q'):
            cv2.destroyAllWindows()
            break



finally:

    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
