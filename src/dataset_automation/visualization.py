import open3d as o3d
import cv2 as cv
import copy
import numpy as np

class Visualization:
    def __init__(self, vehicle):
        self.vis = o3d.visualization.Visualizer()
        self._lidar_added = False
        self.point_list_tot = o3d.geometry.PointCloud()

        self.lidar_create_window()
        print(vehicle.get_transform())
        print("got in visualization")
        
        #track previous state to handle "Toggle" logic 
        self.show_lidar = True
        self.show_camera = True
        self.lidar_key_prev = False
        self.cam_key_prev = False
        
        
    def lidar_create_window(self): 
        self.vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270)
        opt = self.vis.get_render_option()
        opt.background_color = [0.05, 0.05, 0.05]
        opt.point_size = 1
        opt.show_coordinate_frame = True
        
    def lidar_update_window(self, frame):
        if frame == 2 and not self._lidar_added:
            self.vis.add_geometry(self.point_list_tot)
            self._lidar_added = True

        if self._lidar_added:
            self.vis.update_geometry(self.point_list_tot)
            self.vis.poll_events()
            self.vis.update_renderer()
        # print(f"frame {frame} updated")
        
        
    # toggle logic
    def lidar_toggle(self, lidar, point_list, history):
        if self.lidar_key_prev:
            return

        self.show_lidar = not self.show_lidar
        print(f"\nLiDAR Visualization: {self.show_lidar}")

        if self.show_lidar:
            # after destroy_window(), re-create the Visualizer cleanly
            self.vis = o3d.visualization.Visualizer()
            history.clear()
            self.point_list_tot = o3d.geometry.PointCloud()
            self._lidar_added = False

            self.lidar_create_window()
            lidar.lidar.listen(lambda data: lidar.callback(data, point_list, self.point_list_tot))
        else:
            lidar.lidar.stop()
            try:
                self.vis.destroy_window()
            finally:
                history.clear()
                self.point_list_tot = o3d.geometry.PointCloud()
                self._lidar_added = False

        self.lidar_key_prev = True
                    
    def lidar_show(self, frame):
        if self.show_lidar:
                try:
                    self.lidar_update_window(frame)
                except:
                    pass  

    # toggle logic            
    def camera_toggle(self, camera):
        if not self.cam_key_prev:
            self.show_camera = not self.show_camera
            print(f"\nCamera Visualization: {self.show_camera}")
            
            if self.show_camera:
                camera.camera.listen(lambda image: camera.callback(image))
            else:
                camera.camera.stop()
                cv.destroyWindow("Camera View")
                cv.waitKey(1) 
            
            self.cam_key_prev = True
            
            
    def camera_show(self, camera_type, camera):
        if camera_type == "mono":
            if self.show_camera and camera.image is not None:
                cv.imshow("Camera View", camera.image)
                cv.waitKey(1)
        elif camera_type == "rgbd":
            if self.show_camera and camera.rgb_image is not None and camera.depth_image is not None:
                cv.imshow("Camera View", camera.rgb_image)
                #depth display: normalize to [0,1] and clip to max range for visualization: the max value is 65535
                depth_display = np.clip(camera.depth_image / 65535.0, 0.0, 1.0)
                cv.imshow("Depth View", depth_display)
                cv.waitKey(1)
        elif camera_type == "stereo":
            if self.show_camera and camera.left_image is not None and camera.right_image is not None:
                cv.imshow("Left Camera View", camera.left_image)
                cv.imshow("Right Camera View", camera.right_image)
                cv.waitKey(1)
        else:
            print(f"Unknown camera type '{camera_type}' in config.yaml. Cannot show camera feed.")
