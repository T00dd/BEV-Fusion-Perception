import carla
import random
import time
import sys
import cv2 as cv
import numpy as np
import open3d as o3d
from matplotlib import cm
from collections import deque
import threading
import copy

VIRIDIS = np.array(cm._colormaps.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


class Lidar:
    def __init__(self, world, vehicle, point_list, point_list_tot, noise=False):
        self._history = deque(maxlen=5)
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', '64')
        bp.set_attribute('range', '120')
        bp.set_attribute('points_per_second', '1000000')

        #FOV verticale
        bp.set_attribute('upper_fov', '22.5')
        bp.set_attribute('lower_fov', '-45')

        #FOV orizzontale 
        bp.set_attribute('horizontal_fov', '120')

        bp.set_attribute('rotation_frequency', '20')
        if not noise:
            bp.set_attribute('dropoff_general_rate', '0.0')
            bp.set_attribute('dropoff_intensity_limit', '1.0')
            bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:
            bp.set_attribute('noise_stddev', '0.2')

        tf = carla.Transform(carla.Location(x=0.4, z=1.5))
        self.lidar = world.spawn_actor(bp, tf, attach_to=vehicle,
                                       attachment_type=carla.AttachmentType.Rigid)
        print('lidar attached')
        self.lidar.listen(lambda data: self.callback(data, point_list, point_list_tot))
        print('lidar listening')

    def callback(self, point_cloud, point_list, point_list_tot):
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.float32))
        data = np.reshape(data, (-1, 4))
        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])
        ]
        points = data[:, :-1]
        points[:, 1] = -points[:, 1]
        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)
        self._history.append(copy.deepcopy(point_list))
        pts = np.vstack([np.asarray(p.points) for p in self._history])
        cols = np.vstack([np.asarray(p.colors) for p in self._history])
        point_list_tot.points = o3d.utility.Vector3dVector(pts)
        if cols.size:
            point_list_tot.colors = o3d.utility.Vector3dVector(cols)

    # def callback(self, point_cloud, point_list, point_list_tot):
    #         # Il Semantic Lidar ha una struttura di 6 valori per punto (3 float per la pos, 1 float per l'angolo, 2 interi per gli ID)
    #         dt = np.dtype([
    #             ('x', np.float32), ('y', np.float32), ('z', np.float32),
    #             ('cos_angle', np.float32), ('obj_idx', np.uint32), ('obj_tag', np.uint32)
    #         ])
            
    #         # Leggiamo i dati grezzi usando la nuova struttura
    #         data = np.frombuffer(point_cloud.raw_data, dtype=dt)
            
    #         # Estraiamo le coordinate X, Y, Z e formiamo la matrice dei punti
    #         points = np.vstack((data['x'], data['y'], data['z'])).T
    #         points[:, 1] = -points[:, 1]  # Invertiamo la Y per Open3D
            
    #         # Estraiamo il tag semantico (cosa ha colpito il laser?)
    #         tags = data['obj_tag']
            
    #         # In CARLA, i tag vanno da 0 a circa 30. Li normalizziamo tra 0 e 1
    #         # per dargli in pasto la tua vecchia palette di colori (VIRIDIS)
    #         norm_tags = np.clip(tags / 30.0, 0.0, 1.0)
            
    #         # Coloriamo i punti in base al tipo di oggetto, non più in base all'intensità!
    #         int_color = np.c_[
    #             np.interp(norm_tags, VID_RANGE, VIRIDIS[:, 0]),
    #             np.interp(norm_tags, VID_RANGE, VIRIDIS[:, 1]),
    #             np.interp(norm_tags, VID_RANGE, VIRIDIS[:, 2])
    #         ]
            
    #         # Passiamo i punti e i colori a Open3D
    #         point_list.points = o3d.utility.Vector3dVector(points)
    #         point_list.colors = o3d.utility.Vector3dVector(int_color)
            
    #         self._history.append(copy.deepcopy(point_list))
    #         pts = np.vstack([np.asarray(p.points) for p in self._history])
    #         cols = np.vstack([np.asarray(p.colors) for p in self._history])
            
    #         point_list_tot.points = o3d.utility.Vector3dVector(pts)
    #         if cols.size:
    #             point_list_tot.colors = o3d.utility.Vector3dVector(cols)


class CameraStereo:
    def __init__(self, world, vehicle):
        lib = world.get_blueprint_library()
        self.frame_buffer = {}
        self._lock = threading.Lock()
        left_bp  = lib.find('sensor.camera.rgb')
        right_bp = lib.find('sensor.camera.rgb')
        for bp in (left_bp, right_bp):
            bp.set_attribute('sensor_tick', '0.01')
            bp.set_attribute('image_size_x', '640')
            bp.set_attribute('image_size_y', '480')
        self.left_camera = world.spawn_actor(
            left_bp, carla.Transform(carla.Location(x=0.4, y=-0.1, z=1.4)),
            attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.right_camera = world.spawn_actor(
            right_bp, carla.Transform(carla.Location(x=0.4, y=0.1, z=1.4)),
            attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        print('stereo cameras attached')
        self.left_image = self.right_image = None
        self.left_camera.listen(lambda d: self.callback('left', d))
        self.right_camera.listen(lambda d: self.callback('right', d))
        print('stereo cameras listening')

    def callback(self, side, data):
        frame = data.frame
        with self._lock:
            self.frame_buffer.setdefault(frame, {})[side] = data
            if 'left' in self.frame_buffer[frame] and 'right' in self.frame_buffer[frame]:
                self.left_image, self.right_image = self._process(frame)

    def _process(self, frame):
        def to_rgb(d):
            a = np.frombuffer(d.raw_data, dtype=np.uint8).reshape(d.height, d.width, 4)
            return a[:, :, :3]
        l = to_rgb(self.frame_buffer[frame]['left'])
        r = to_rgb(self.frame_buffer[frame]['right'])
        for f in [k for k in self.frame_buffer if k <= frame]:
            del self.frame_buffer[f]
        return l, r


class Visualization:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self._lidar_added = False
        self.point_list_tot = o3d.geometry.PointCloud()
        self.vis.create_window(window_name='Carla Lidar', width=960, height=540, left=480, top=270)
        opt = self.vis.get_render_option()
        opt.background_color = [0.05, 0.05, 0.05]
        opt.point_size = 1
        opt.show_coordinate_frame = True

    def update_lidar(self, frame):
        if frame == 2 and not self._lidar_added:
            self.vis.add_geometry(self.point_list_tot)
            self._lidar_added = True
        if self._lidar_added:
            self.vis.update_geometry(self.point_list_tot)
            self.vis.poll_events()
            self.vis.update_renderer()

    def show_stereo(self, camera):
        if camera.left_image is not None and camera.right_image is not None:
            combined = np.hstack((camera.left_image, camera.right_image))
            cv.imshow('Stereo View (Left | Right)', combined)
            cv.waitKey(1)


def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()


        original_settings = world.get_settings()
        settings = world.get_settings()
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(0)
        settings.fixed_delta_seconds = 0.01
        settings.synchronous_mode = True
        world.apply_settings(settings)
        
        tm.set_hybrid_physics_mode(True)
        tm.set_hybrid_physics_radius(70.0)  # raggio in metri intorno all'ego vehicle
        
        lib = world.get_blueprint_library()
        bp = lib.find('vehicle.audi.tt')
        bp.set_attribute('role_name', 'hero')
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        vehicle.set_autopilot(True, tm.get_port())
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # try:
        #     cone_bp = lib.find('static.prop.orangecone')
        #     veh_transform = vehicle.get_transform()
        #     spawn_loc = veh_transform.location + (veh_transform.get_forward_vector() * 5.0)
        #     spawn_loc.z += 1.0 # Lo facciamo cadere da 1 metro d'altezza
            
        #     test_cone = world.spawn_actor(cone_bp, carla.Transform(spawn_loc))
        #     test_cone.set_simulate_physics(True)
        #     actor_list.append(test_cone)
        #     print("Cono di test spawnato con successo!")
        # except Exception as e:
        #     print(f"Errore spawn cono test: {e}")

        point_list = o3d.geometry.PointCloud()
        vis = Visualization()

        camera = CameraStereo(world, vehicle)
        actor_list += [camera.left_camera, camera.right_camera]

        lidar = Lidar(world, vehicle, point_list, vis.point_list_tot, noise=False)
        actor_list.append(lidar.lidar)

        

        frame = 0
        print("Running... Ctrl+C to stop.")
        while True:
            try:
                vis.update_lidar(frame)
            except Exception:
                pass
            vis.show_stereo(camera)
            time.sleep(0.005)
            world.tick()
            sys.stdout.flush()
            frame += 1
            

    finally:
        world.apply_settings(original_settings)
        cv.destroyAllWindows()
        try:
            vis.vis.destroy_window()
        except Exception:
            pass
        for actor in actor_list:
            actor.destroy()
        print('All actors destroyed.')


if __name__ == '__main__':
    main()
