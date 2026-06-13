import math
import carla
import numpy as np

class PurePursuitController:
    def __init__(self, waypoints: np.ndarray, wheelbase: float=2.5, lookahead: float=6.0, target_speed: float=8.0,
                 kp_speed: float=0.5, max_steer_radius: float=math.radians(45)):
        self._wp = waypoints
        self._L = wheelbase
        self._Ld = lookahead
        self._v_target = target_speed
        self._kp = kp_speed
        self._max_steer = max_steer_radius
        self._last_idx = 0
    
    def find_lookahead_point(self, x: float, y: float):
        n = len(self._wp)
        ld2 = self._Ld ** 2
        for i in range(self._last_idx, n):
            dx = self._wp[i, 0] - x
            dy = self._wp[i, 1] - y
            if dx*dx + dy*dy >= ld2:
                self._last_idx = i
                return self._wp[i, 0], self._wp[i, 1]
        self._last_idx = n - 1
        return self._wp[-1, 0], self._wp[-1, 1]
    
    def step(self, vehicle) -> carla.VehicleControl:
        tf = vehicle.get_transform()
        x, y = tf.location.x, tf.location.y
        yaw = math.radians(tf.rotation.yaw)
        v = vehicle.get_velocity()
        speed = math.hypot(v.x, v.y)

        tx, ty = self.find_lookahead_point(x, y)
        dx, dy = tx - x, ty - y
        alpha = math.atan2(dy, dx) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        Ld_real = math.hypot(dx, dy)
        steer_rad = math.atan2(2.0*self._L*math.sin(alpha), Ld_real)
        steer = max(-1.0, min(1.0, steer_rad / self._max_steer))

        err = self._v_target - speed
        throttle = max(0.0, min(1.0, self._kp * err))
        brake = max(0.0, min(1.0, -self._kp * err))

        ctrl = carla.VehicleControl()
        ctrl.throttle = float(throttle)
        ctrl.steer = float(steer)
        ctrl.brake = float(brake)
        ctrl.hand_brake = False
        ctrl.reverse = False
        return ctrl

    @property
    def reached_end(self) -> bool:
        return self._last_idx >= len(self._wp) - 1