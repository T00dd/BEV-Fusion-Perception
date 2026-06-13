import carla
import cv2 as cv
import numpy as np
import open3d as o3d
from matplotlib import cm
from collections import deque
import copy

VIRIDIS = np.array(cm._colormaps.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class Lidar:
    def __init__(self, world, vehicle, point_list, noise, point_list_tot):  
        
        self._history = deque(maxlen=5)

        blueprint_library = world.get_blueprint_library()  
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        #set the blueprint attributes
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', '120')
        lidar_bp.set_attribute('points_per_second', '500000')
        lidar_bp.set_attribute('upper_fov', '22.5')
        lidar_bp.set_attribute('lower_fov', '-22.5')
        lidar_bp.set_attribute('rotation_frequency', str(20))
        
        # for noise
        if not noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:
            lidar_bp.set_attribute('noise_stddev', '0.2')
        
        
        # position the lidar on the car
        relative_transform = carla.Transform(carla.Location(x=0.4, z=1.5))
        self.lidar = world.spawn_actor(lidar_bp, relative_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        print("lidar attached")
        
        # attach the callback
        self.lidar.listen(lambda data: self.callback(data, point_list, point_list_tot))
        print("lidar listening")        
        
    def callback(self, point_cloud, point_list, point_list_tot):
        
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        # Isolate the intensity and compute a color for it
        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        # Isolate the 3D data
        points = data[:, :-1]

        # We're negating the y to correclty visualize a world that matches
        # what we see in Unreal since Open3D uses a right-handed coordinate system
        points[:, 1] = -points[:, 1]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)

        self._history.append(copy.deepcopy(point_list))

        # merge last 5 into numpy arrays
        pts_list = []
        col_list = []
        for p in self._history:
            pts_list.append(np.asarray(p.points))
            col_list.append(np.asarray(p.colors))

        if len(pts_list) == 0:
            return

        pts = np.vstack(pts_list)
        cols = np.vstack(col_list) if len(col_list) else None

        # mutate the object that contains all 5 last frames
        point_list_tot.points = o3d.utility.Vector3dVector(pts)
        if cols is not None and cols.size:
            point_list_tot.colors = o3d.utility.Vector3dVector(cols)       