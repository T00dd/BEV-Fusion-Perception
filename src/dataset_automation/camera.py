import carla
import cv2 as cv
import numpy as np
import threading

class cameraMono:

    def __init__(self, world, vehicle, zmq_socket, socket_lock=None):   
        blueprint_library = world.get_blueprint_library()  
        self.zmq_socket = zmq_socket
        self.socket_lock = socket_lock
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('sensor_tick', str(0.05))  # set to 20 FPS
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        relative_transform = carla.Transform(carla.Location(x=0.4, z=1.4))
        self.camera = world.spawn_actor(camera_bp, relative_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        print("camera attached")
        self.image = None
        self.camera.listen((lambda image: self.callback(image)))
        print("camera listening")
        
    def callback(self, data):
        self.image = self.process_image(data)
    
    def process_image(self, data):
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        return array[:, :, :3]
    
class cameraRgbd:

    def __init__(self, world, vehicle, zmq_socket, socket_lock=None):   
        blueprint_library = world.get_blueprint_library()  
        self.zmq_socket = zmq_socket
        self.socket_lock = socket_lock
        
        self.frame_buffer = {}
        self._lock = threading.Lock()
        
        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        depth_camera_bp = blueprint_library.find('sensor.camera.depth')
        rgb_camera_bp.set_attribute('sensor_tick', str(0.01))  
        depth_camera_bp.set_attribute('sensor_tick', str(0.01))
        rgb_camera_bp.set_attribute('image_size_x', '640')
        rgb_camera_bp.set_attribute('image_size_y', '480')
        depth_camera_bp.set_attribute('image_size_x', '640')
        depth_camera_bp.set_attribute('image_size_y', '480')
        relative_transform = carla.Transform(carla.Location(x=0.4, z=1.4))
        self.rgb_camera = world.spawn_actor(rgb_camera_bp, relative_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.depth_camera = world.spawn_actor(depth_camera_bp, relative_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        print("rgbd camera attached")
        self.rgb_image = None
        self.depth_image = None
        self.rgb_camera.listen(lambda data: self.callback("rgb", data))
        self.depth_camera.listen(lambda data: self.callback("depth", data))
        print("camera listening")
        
    def callback(self, sensor_name, data):
            
        frame = data.frame
        
        with self._lock:
            if frame not in self.frame_buffer:
                self.frame_buffer[frame] = {}
                
            self.frame_buffer[frame][sensor_name] = data
            
            if 'rgb' in self.frame_buffer[frame] and 'depth' in self.frame_buffer[frame]:
                self.rgb_image, self.depth_image = self.process_frame(frame)
                
    
    def process_frame(self, frame):
        rgb_data = self.frame_buffer[frame]["rgb"]
        depth_data = self.frame_buffer[frame]["depth"]
        
        # rgb part
        rgb_array = np.frombuffer(rgb_data.raw_data, dtype=np.dtype("uint8"))
        rgb_array = np.reshape(rgb_array, (rgb_data.height, rgb_data.width, 4))
        rgb_processed = np.copy(rgb_array[:, :, :3])
        
        # depth part
        depth_array = np.frombuffer(depth_data.raw_data, dtype=np.dtype("uint8"))
        depth_array = np.reshape(depth_array, (depth_data.height, depth_data.width, 4))
        
        B = depth_array[:, :, 0].astype(np.float32)
        G = depth_array[:, :, 1].astype(np.float32)
        R = depth_array[:, :, 2].astype(np.float32)
        
        normalized_depth = (R + G * 256.0 + B * (256.0 ** 2)) / ((256.0 ** 3) - 1)
        depth_in_m =  1000.0 * 1000.0 * normalized_depth 
        # clip depth to max range of 65535mm and convert to uint16 for visualization and transmission
        depth_in_m_clipped = np.clip(depth_in_m, 0, 65535)       
        depth_in_m_clipped[depth_in_m_clipped == 65535] = 0  # set max depth to 0 for slam algorithms
        
        depth_uint16 = (depth_in_m_clipped).astype(np.uint16)
        
        # clip rgb image where depth is max range ()
        # rgb_processed[depth_uint16 == 0] = 0
        
        # clean frame buffer
        for f in list(self.frame_buffer.keys()):
            if f <= frame:
                del self.frame_buffer[f]
                
        return rgb_processed, depth_uint16
        
    
class cameraStereo:
    def __init__(self, world, vehicle, zmq_socket, socket_lock=None):
        blueprint_library = world.get_blueprint_library()  
        self.zmq_socket = zmq_socket
        self.socket_lock = socket_lock
        
        self.frame_buffer = {}
        self._lock = threading.Lock()
        left_rgb_camera = blueprint_library.find('sensor.camera.rgb')
        right_rgb_camera = blueprint_library.find('sensor.camera.rgb')
        left_rgb_camera.set_attribute('sensor_tick', str(0.01))
        right_rgb_camera.set_attribute('sensor_tick', str(0.01))
        left_rgb_camera.set_attribute('image_size_x', '640')
        left_rgb_camera.set_attribute('image_size_y', '480')
        right_rgb_camera.set_attribute('image_size_x', '640')
        right_rgb_camera.set_attribute('image_size_y', '480')
        relative_transform_left = carla.Transform(carla.Location(x=0.4, y=-0.2, z=1.4))
        relative_transform_right = carla.Transform(carla.Location(x=0.4, y=0.2, z=1.4))
        self.left_camera = world.spawn_actor(left_rgb_camera, relative_transform_left, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.right_camera = world.spawn_actor(right_rgb_camera, relative_transform_right, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        print("stereo cameras attached")
        self.left_image = None
        self.right_image = None
        self.left_camera.listen(lambda data: self.callback("left", data))
        self.right_camera.listen(lambda data: self.callback("right", data))
        print("stereo cameras listening")
        
    
    def callback(self, side, data):
        frame = data.frame
        
        with self._lock:
            if frame not in self.frame_buffer:
                self.frame_buffer[frame] = {}
                
            self.frame_buffer[frame][side] = data
            
            if 'left' in self.frame_buffer[frame] and 'right' in self.frame_buffer[frame]:
                self.left_image, self.right_image = self.process_frame(frame)
    
    def process_frame(self, frame):
        left_data = self.frame_buffer[frame]["left"]
        right_data = self.frame_buffer[frame]["right"]
        
        left_array = np.frombuffer(left_data.raw_data, dtype=np.dtype("uint8"))
        left_array = np.reshape(left_array, (left_data.height, left_data.width, 4))
        left_processed = left_array[:, :, :3]
        
        right_array = np.frombuffer(right_data.raw_data, dtype=np.dtype("uint8"))
        right_array = np.reshape(right_array, (right_data.height, right_data.width, 4))
        right_processed = right_array[:, :, :3]
        
        # clean frame buffer
        for f in list(self.frame_buffer.keys()):
            if f <= frame:
                del self.frame_buffer[f]
                
        return left_processed, right_processed