import carla
import random
import time
from camera import cameraMono, cameraRgbd, cameraStereo
from lidar import Lidar
from visualization import Visualization
import open3d as o3d
import sys
import cv2 as cv
import yaml

import math
import time

from pathlib import Path
import numpy as np
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import MAIN_CONFIG, DATA_DIR, RECONSTRUCTOR_BIN
from track_spawner import generate_and_spawn_track
from centerline_pipeline import compute_centerline_carla, draw_debug
from pursuit_controller import PurePursuitController


def main():
    actor_list = []
    cone_actors = []

    try:
        # creating the client and attaching to the port 2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # get config type from yaml
        with open(MAIN_CONFIG, "r") as f:
            config = yaml.safe_load(f)
            camera_type = config.get("camera", {}).get("type", "mono").lower()
            print(f"Camera type from config: {camera_type}")
            ctrl_cfg = config.get("controller", {}) or {}
            target_speed = ctrl_cfg.get("target_speed", 8.0)
            lookahead = ctrl_cfg.get("lookahead", 6.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()

        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        # disable traffic lights to avoid stopping the vehicle
        traffic_manager.set_random_device_seed(0)

        # with this delta we are simulating in real time, not sped up. since a lidar frame at this speed is not a complete pointcloud, we'll need to keep
        # the last 5 frames
        delta = 0.01

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.find("vehicle.audi.tt")
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # track + centerline generation
        cones, start_x, start_y, carla_scale, cone_actors = generate_and_spawn_track()

        waypoints = compute_centerline_carla(
            cones,
            start_x,
            start_y,
            carla_scale=carla_scale,
            data_dir=DATA_DIR,
            reconstructor_bin=RECONSTRUCTOR_BIN
        )

        # darw the centerline for debugging
        draw_debug(world, waypoints, life_time=60.0)

        # spwan the vehicle ath the first waypoint, oriented towards the second waypoint
        first = waypoints[0]
        ahead = waypoints[min(5, len(waypoints)-1)]
        yaw_deg = math.degrees(math.atan2(ahead[1] - first[1], ahead[0] - first[0]))
        spaw_tf = carla.Transform(
            carla.Location(x=float(first[0]), y=float(first[1]), z=float(first[2]) + 1.0),
            carla.Rotation(yaw=yaw_deg)
        )

        point_list = o3d.geometry.PointCloud()
        vehicle = world.spawn_actor(bp, spaw_tf)

        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        controller = PurePursuitController(waypoints, target_speed=target_speed, lookahead=lookahead)

        # create the visualization class

        visualization = Visualization(vehicle)
        print("created vis")

        # spawn sensors

        # camera
        camera = None
        if camera_type == "mono":
            camera = cameraMono(world, vehicle, None, None)
            actor_list.append(camera.camera)
            print('created %s' % camera.camera.type_id)
        elif camera_type == "rgbd":
            camera = cameraRgbd(world, vehicle, None, None)
            actor_list.append(camera.rgb_camera)
            actor_list.append(camera.depth_camera)
            print('created %s' % camera.rgb_camera.type_id)
            print('created %s' % camera.depth_camera.type_id)
        elif camera_type == "stereo":
            camera = cameraStereo(world, vehicle, None, None)
            actor_list.append(camera.left_camera)
            actor_list.append(camera.right_camera)
            print('created %s' % camera.left_camera.type_id)
            print('created %s' % camera.right_camera.type_id)
        else:
            print(f"Unknown camera type '{camera_type}' in config.yaml. Defaulting to mono.")
            camera = cameraMono(world, vehicle, None, None)

        # lidar
        lidar = Lidar(world, vehicle, point_list, False, visualization.point_list_tot)
        actor_list.append(lidar.lidar)
        print('created %s' % lidar.lidar.type_id)

        frame = 0
        print("Running... Press 'L' to toggle Lidar, 'C' to toggle Camera.")

        while True:

            vehicle.apply_control(controller.step(vehicle))
            if(controller.reached_end):
                print("Reached end of track, stopping vehicle.")
                break

            visualization.lidar_show(frame)

            visualization.camera_show(camera_type, camera)

            # This can fix Open3D jittering issues:
            time.sleep(0.005)
            world.tick()

            sys.stdout.flush()
            frame += 1


    finally:
        world.apply_settings(original_settings)

        cv.destroyAllWindows()
        try:
            visualization.vis.destroy_window()
        except:
            pass
        for actor in actor_list:
            actor.destroy()

        # Remove the spawned cones from the world on exit.
        if cone_actors:
            try:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(c.id) for c in cone_actors], True
                )
                print(f"Destroyed {len(cone_actors)} cone actors on shutdown.")
            except Exception:
                for c in cone_actors:
                    try:
                        c.destroy()
                    except RuntimeError:
                        pass


if __name__ == '__main__':

    main()