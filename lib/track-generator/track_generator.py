import math
import gpxpy
import os, yaml
import gpxpy.gpx
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from scipy import signal, spatial, interpolate
from scipy.io import savemat
from shapely.geometry.polygon import Point, LineString, Polygon

class TrackGenerator:
    """
    Generates a random track based on a bounded Voronoi diagram.
    Ensures that the tracks curvature is within limits and that the car starts at a straight section.
    """

    def __init__(self, 
                 n_points: int, 
                 n_regions: int, 
                 min_bound: float, 
                 max_bound: float, 
                 mode: Mode,
                 open_loop: bool,
                 missing_cone_ratio: float,
                 random_open_loop: bool,
                 behind_ratio: float,
                 ahead_ratio: float,
                 plot_track: bool, 
                 visualise_voronoi: bool,
                 create_output_file: bool, 
                 output_location: str,
                 z_offset: float = 0,
                 lat_offset: float = 0,
                 lon_offset: float = 0,
                 sim_type: SimType = SimType.FSSIM):
                 
        # Input parameters
        self._n_points = n_points                                               # [-]
        self._n_regions = n_regions                                             # [-]
        self._min_bound = min_bound                                             # [m]
        self._max_bound = max_bound                                             # [m]
        self._bounding_box = np.array([self._min_bound, self._max_bound] * 2)   # [x_min, x_max, y_min, y_max]
        self._mode = mode
        self._sim_type = sim_type

        # Track parameters
        self._track_width = 3.                                                  # [m]
        self._cone_spacing = 5.                                                 # [m]
        self._length_start_area = 6.                                            # [m]
        self._curvature_threshold = 1. / 3.75                                   # [m^-1]
        self._straight_threshold = 1. / 100.                                    # [m^-1]
        self._missing_cone_ratio = missing_cone_ratio                           # [0.0 - 1.0]

        # Open loop parameters
        self._open_loop = open_loop                                             # [bool]
        self._random_open_loop = random_open_loop                               # [bool]
        self._behind_ratio = behind_ratio                                       # [0.0 - 1.0]
        self._ahead_ratio = ahead_ratio                                         # [m]

        # Output options
        self._plot_track = plot_track
        self._visualise_voronoi = visualise_voronoi
        self._create_output_file = create_output_file
        self._output_location = output_location
        self._z_offset = z_offset
        self._lat_offset = lat_offset
        self._lon_offset = lon_offset

    def bounded_voronoi(self, input_points, bounding_box):
        """
        Creates a Voronoi diagram bounded by the bounding box.
        Mirror input points at edges of the bounding box.
        Then create Voronoi diagram using all five sets of points.
        This prevents having a Voronoi diagram with edges going off to infinity.
        
        Args:
            input_points (numpy.ndarray): Coordinates of input points for Voronoi diagram.
            bounding_box (numpy.ndarray): Specifies the boundaries of the Voronoi diagram, [x_min, x_max, y_min, y_max].
        
        Returns:
            scipy.spatial.qhull.Voronoi: Voronoi diagram object.
        """
        
        def _mirror(boundary, axis):
            mirrored = np.copy(points_center)
            mirrored[:, axis] = 2 * boundary - mirrored[:, axis]
            return mirrored
        
        x_min, x_max, y_min, y_max = bounding_box
        
        # Mirror points around each boundary
        points_center = input_points
        points_left = _mirror(x_min, axis=0) 
        points_right = _mirror(x_max, axis=0) 
        points_down = _mirror(y_min, axis=1)
        points_up = _mirror(y_max, axis=1)
        points = np.concatenate([points_center, points_left, points_right, points_down, points_up])
        
        # Compute Voronoi
        vor = spatial.Voronoi(points)
        
        # We only need the section of the Voronoi diagram that is inside the bounding box
        vor.filtered_points = points_center
        vor.filtered_regions = np.array(vor.regions, dtype=object)[vor.point_region[:vor.npoints//5]]
        return vor

    def create_track(self):
        """
        Creates a track from the vertices of a Voronoi diagram.
        1.  Create bounded Voronoi diagram.
        2.  Select regions of Voronoi diagram based on selection mode.
        3.  Get the vertices belonging to the regions and sort them clockwise.
        4.  Interpolate between vertices.
        5.  Calculate curvature of track to check wether the curvature threshold is exceeded.
        6.  If curvature threshold is exceeded, remove vertice where the curvature is the highest from its set.
            Repeat steps 4-6 until curvature is within limimts.
        7.  Check if track does not cross itself. If so, go to step 2 and reiterate.
        8.  Find long enough straight section to place start line and start position.
        9.  Translate and rotate track to origin.
        10. Create track yaml file.
        11. Return the generated cone arrays.
        """
        # Create bounded Voronoi diagram
        input_points = np.random.uniform(self._min_bound, self._max_bound, (self._n_points, 2))
        vor = self.bounded_voronoi(input_points, self._bounding_box)

        while True:
            
            if self._mode.value == 1:
                # Pick a random point and find its n closest neighbours
                random_index = np.random.randint(0, self._n_points)
                random_point_indices = [random_index]
                random_point = input_points[random_index]
                
                for i in range(self._n_regions - 1):
                    closest_point_index = closest_node(random_point, input_points, k=i+1)
                    random_point_indices.append(closest_point_index)
                    
            elif self._mode.value == 2:
                # Pick a random point, create a line extending from this point and find other points close to this line
                random_index = np.random.randint(0, self._n_points)
                random_heading = np.random.uniform(0, np.pi/2)
                random_point = input_points[random_index]
                
                start = (random_point[0] - 1./2. * self._max_bound * np.cos(random_heading), random_point[1] - 1./2. * self._max_bound * np.sin(random_heading))
                end = (random_point[0] + 1./2. * self._max_bound * np.cos(random_heading), random_point[1] + 1./2. * self._max_bound * np.sin(random_heading))
                line = LineString([start, end])
                distances = [Point(p).distance(line) for p in input_points]
                random_point_indices = np.argpartition(distances, self._n_regions)[:self._n_regions]
                
            elif self._mode.value == 3:
                # Select regions randomly
                random_point_indices = np.random.randint(0, self._n_points, self._n_regions)
            
            # From the Voronoi regions, get the regions belonging to the randomly selected points
            regions = np.array([np.array(region) for region in vor.regions], dtype=object)
            random_region_indices = vor.point_region[random_point_indices]
            random_regions = np.concatenate(regions[random_region_indices])
            
            # Get the vertices belonging to the random regions
            random_vertices = np.unique(vor.vertices[random_regions], axis=0)
            
            # Sort vertices
            sorted_vertices = clockwise_sort(random_vertices)
            sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])
            
            while True:
        
                # Interpolate
                tck, _ = interpolate.splprep([sorted_vertices[:,0], sorted_vertices[:,1]], s=0, per=True)
                t = np.linspace(0, 1, 1000)
                x, y = interpolate.splev(t, tck, der=0)
                dx_dt, dy_dt = interpolate.splev(t, tck, der=1)
                d2x_dt2, d2y_dt2 = interpolate.splev(t, tck, der=2)
                
                # Calculate curvature
                k = curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2)
                abs_curvature = np.abs(k)
                
                # Check if curvature exceeds threshold
                peaks, _ = signal.find_peaks(abs_curvature)
                exceeded_peaks = abs_curvature[peaks] > self._curvature_threshold
                max_peak_index = abs_curvature[peaks].argmax()
                is_curvature_exceeded = exceeded_peaks[max_peak_index]
                
                if is_curvature_exceeded:
                    # Find vertice where curvature is exceeded and delete vertice from sorted vertices. Reiterate
                    max_peak = peaks[max_peak_index]
                    peak_coordinate = (x[max_peak], y[max_peak])
                    vertice = closest_node(peak_coordinate, sorted_vertices, k=0)
                    sorted_vertices = np.delete(sorted_vertices, vertice, axis=0)
                    
                    # Make sure that first and last coordinate are the same for periodic interpolation
                    if not np.array_equal(sorted_vertices[0], sorted_vertices[-1]):
                        sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])
                else:
                    break
            
            # Create track boundaries
            track = Polygon(zip(x, y))
            track_left = track.buffer(self._track_width / 2)
            track_right = track.buffer(-self._track_width / 2)
            
            # Check if track does not cross itself
            if track.is_valid and track_left.is_valid and track_right.is_valid:
                if track.geom_type == track_left.geom_type == track_right.geom_type == 'Polygon':
                    break

        # Calculate cone spacing
        cone_spacing_left = np.linspace(0, track_left.length, np.ceil(track_left.length / self._track_width).astype(int) + 1)[:-1]
        cone_spacing_right= np.linspace(0, track_right.length, np.ceil(track_right.length / self._track_width).astype(int) + 1)[:-1]
        
        # Determine coordinates of cones
        cones_left = np.asarray([np.asarray(track_left.exterior.interpolate(sp).xy).flatten() for sp in cone_spacing_left])
        cones_right = np.asarray([np.asarray(track_right.exterior.interpolate(sp).xy).flatten() for sp in cone_spacing_right])

        # Set a random starting point
        start_ratio = np.random.uniform(0.0, 1.0)
        cones_left_roll = int(cones_left.shape[0] * start_ratio)
        cones_left = np.roll(cones_left, cones_left_roll, axis=0)
        cones_right_roll = int(cones_right.shape[0] * start_ratio)
        cones_right = np.roll(cones_right, cones_right_roll, axis=0)

        start_position = np.array([cones_left[0, 0] / 2 + cones_right[0, 0] / 2,cones_left[0, 1] / 2 + cones_right[0, 1] / 2 ])
        v = cones_right[1] - cones_left[1]
        tangent = np.array([v[1], -v[0]])
        start_heading = float(np.arctan2(tangent[1], tangent[0]))

        # Translate and rotate track to origin
        M = transformation_matrix(-start_position, start_heading - np.pi/2)
        cones_left = M.dot(np.c_[cones_left, np.ones(len(cones_left))].T)[:-1].T
        cones_right = M.dot(np.c_[cones_right, np.ones(len(cones_right))].T)[:-1].T

        # Reorder cones so that the closest cone to the origin is first
        cones_left, cones_right = reorder_cones(cones_left, cones_right)

        # Operate open loop if specified
        if self._open_loop:
            if self._random_open_loop:
                self._ahead_ratio = np.random.uniform(0.1, 1.0)
                self._behind_ratio = np.random.uniform(0.0, 1-self._ahead_ratio if self._behind_ratio <= 0.5 else 0.5)
                print("Random ahead ratio: ", self._ahead_ratio)
                print("Random behind ratio: ", self._behind_ratio)

            cones_left, cones_right = open_loop(ahead_ratio=self._ahead_ratio
                                                , cones_left=cones_left, cones_right=cones_right, behind_ratio=self._behind_ratio)
            
        # Remove cones to simulate missing cones
        cones_left = remove_cones(cones_left, self._missing_cone_ratio)
        cones_right = remove_cones(cones_right, self._missing_cone_ratio)

        tb_saved = {'cones_left': cones_left, 'cones_right': cones_right}
        savemat('cones.mat', tb_saved)

        # Create track file
        if self._visualise_voronoi: self.visualise_voronoi(vor, sorted_vertices, random_point_indices, input_points, x, y)
        if self._plot_track: self.plot_track(cones_left, cones_right)
        if self._create_output_file: self.output_yaml(cones_left.tolist(), cones_right.tolist())

        return tb_saved

    def visualise_voronoi(self, vor, sorted_vertices, random_point_indices, input_points, x, y):
        """
        Visualises the voronoi diagram and the resulting track. 

        Args:
            vor (scipy.spatial.qhull.Voronoi): Voronoi diagram object.
            sorted_vertices (numpy.ndarray): Selected vertices sorted clockwise.
            random_point_indices (numpy.ndarray): Selected points.
            input_points (numpy.ndarray): All Voronoi points.
        """
        # Plot initial points
        plt.figure()
        plt.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.')

        # Plot vertices points
        for region in vor.filtered_regions:
            vertices = vor.vertices[region, :]
            plt.plot(vertices[:, 0], vertices[:, 1], 'go')
            
        # Plot edges
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            plt.plot(vertices[:, 0], vertices[:, 1], 'k-')

        # Plot selected vertices
        plt.scatter(sorted_vertices[:,0], sorted_vertices[:,1], color='y', s=200, label='Selected vertices')

        # Plot selected points
        plt.scatter(*input_points[random_point_indices].T, s=100, marker='x', color='b', label='Selected points')

        # Plot track
        plt.scatter(x, y)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.legend()
        plt.show()

    def plot_track(self, cones_left, cones_right):
        """
        Plots the resulting track. The car will start at the origin.

        Args: 
            cones_left (numpy.ndarray): Nx2 numpy array of left cone coordinates.
            cones_right (numpy.ndarray): Nx2 numpy array of right cone coordinates.       
        """
        plt.figure()
        plt.scatter(*cones_left.T, color='b', s=1)
        plt.scatter(*cones_right.T, color='y', s=1)

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.grid()
        plt.show()
        
    def output_yaml(self, cones_left, cones_right):
        """
        Writes the track data to a yaml file.

        Args:
            cones_left (list): Nx2 list of left cone coordinates.
            cones_right (list): Nx2 list of right cone coordinates.
        """
        abs_path_dir = os.path.realpath(os.path.dirname(__file__))
        track_file_dir = abs_path_dir + self._output_location
        
        if(self._sim_type == SimType.FSSIM):
            track_file_name = track_file_dir + 'random_track.yaml'

            with open(track_file_name, 'w') as outfile:
                data = dict()
                data['cones_left'] = cones_left
                data['cones_right'] = cones_right
                data['cones_orange'] = []
                data['cones_orange_big'] = [[4.7, 2.5], [4.7, -2.5], [7.3, 2.5], [7.3, -2.5]]
                data['starting_pose_cg'] = [0., 0., 0.]
                data['tk_device'] = [[6., 3.], [6., -3.]]
                yaml.dump(data, outfile)
            
        elif(self._sim_type == SimType.FSDS):
            track_file_name = track_file_dir + 'random_track.csv'
            
            print("Saving " + track_file_name)
            
            with open(track_file_name, 'w') as outfile:
                for cone in cones_left:
                    outfile.write("blue," + str(cone[0]) + ',' + str(cone[1]) + ',0,0.01,0.01,0\n')
                    
                for cone in cones_right:
                    outfile.write("yellow," + str(cone[0]) + ',' + str(cone[1]) + ',0,0.01,0.01,0\n')
                    
                outfile.write("big_orange,4.7,2.2,0,0.01,0.01,0\n")
                outfile.write("big_orange,4.7,-2.2,0,0.01,0.01,0\n")
                outfile.write("big_orange,7.3,2.2,0,0.01,0.01,0\n")
                outfile.write("big_orange,7.3,-2.2,0,0.01,0.01,0\n")
        elif(self._sim_type == SimType.GPX):
            track_file_name = track_file_dir + 'random_track.gpx'
            gpx = gpxpy.gpx.GPX()

            # Create first track in our GPX:
            gpx_track = gpxpy.gpx.GPXTrack()
            gpx.tracks.append(gpx_track)
            
            # Create points:
            for cone in cones_left:
                lat  = self._lat_offset  + (cone[1] / 6378100) * (180 / math.pi)
                lon = self._lon_offset + (cone[0] / 6378100) * (180 / math.pi) / math.cos(self._lat_offset * math.pi/180)
                gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=0 + self._z_offset))
                
            for cone in cones_right:
                lat  = self._lat_offset  + (cone[1] / 6378100) * (180 / math.pi)
                lon = self._lon_offset + (cone[0] / 6378100) * (180 / math.pi) / math.cos(self._lat_offset * math.pi/180)
                gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=0 + self._z_offset))
            
            with open(track_file_name, 'w') as outfile:
                outfile.writelines(gpx.to_xml())


def remove_cones(cones, missing_cone_ratio):
    """
    Removes a percentage of cones to simulate missing cones.

    Args:
        cones (numpy.ndarray): Nx2 numpy array of cone coordinates.
        missing_cone_ratio (float): Ratio of cones to remove [0.0 - 1.0].

    Returns:
        numpy.ndarray: Nx2 numpy array of cone coordinates with missing cones removed.
    """
    n_cones = len(cones)
    n_cones_to_remove = int(n_cones * missing_cone_ratio)
    indices_to_remove = np.random.choice(n_cones, n_cones_to_remove, replace=False)
    cones_with_missing = np.delete(cones, indices_to_remove, axis=0)
    return cones_with_missing   


def open_loop(ahead_ratio, cones_left, cones_right, behind_ratio):
    """
    Generates an open loop path based on the cones.

    Args:
        look_ahead_distance (float): Look ahead distance for the open loop path [m].
        cones_left (numpy.ndarray): Nx2 numpy array of left cone coordinates.
        cones_right (numpy.ndarray): Nx2 numpy array of right cone coordinates.
        visited_ratio (float): Ratio of already visited cones [0.0 - 1.0].

    Returns:
        cones left (numpy.ndarray): Updated Nx2 numpy array of left cone coordinates.
        cones right (numpy.ndarray): Updated Nx2 numpy array of right cone coordinates.
    """

    if ahead_ratio + behind_ratio > 1.0:
        raise ValueError("Look ahead distance and behind distance ratios must sum to less than or equal to 1.0")

    open_cones_left = []
    open_cones_right = []

    last_cone_left = len(cones_left)
    last_cone_right = len(cones_right)

    cones_ahead_left = cones_left[0:int(last_cone_left * ahead_ratio)]
    cones_ahead_right = cones_right[0:int(last_cone_right * ahead_ratio)]

    print("Number of cones ahead left: ", len(cones_ahead_left))
    print("Number of cones ahead right: ", len(cones_ahead_right))

    print("Total number of cones left: ", last_cone_left + 1)
    print("Total number of cones right: ", last_cone_right + 1)
    cones_behind_left = cones_left[last_cone_left - int(last_cone_left * behind_ratio):last_cone_left]
    cones_behind_right = cones_right[last_cone_right - int(last_cone_right * behind_ratio):last_cone_right]

    print("Number of cones behind left: ", len(cones_behind_left))
    print("Number of cones behind right: ", len(cones_behind_right))

    open_cones_left = np.append(cones_behind_left, cones_ahead_left, axis=0)
    open_cones_right = np.append(cones_behind_right, cones_ahead_right, axis=0)

    print("Total number of open cones left: ", len(open_cones_left))
    # print("Cones left after open loop operation: ", open_cones_left)
    print("Total number of open cones right: ", len(open_cones_right))
    # print("Cones right after open loop operation: ", open_cones_right)

    return open_cones_left, open_cones_right


def reorder_cones (cones_left, cones_right) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorders the cones so that the closest cone to the origin is first.

    Args:
        left_cones (numpy.ndarray): Nx2 numpy array of left cone coordinates.
        right_cones (numpy.ndarray): Nx2 numpy array of right cone coordinates.
    """

    print("First element of left cones: ", cones_left[0])
    print("First element of right cones: ", cones_right[0])
    
    distances_left = np.linalg.norm(cones_left, axis=1)
    distances_right = np.linalg.norm(cones_right, axis=1)
    first_cone_left_index = np.argmin(distances_left)
    first_cone_right_index = np.argmin(distances_right)

    print("First cone left index: ", first_cone_left_index, "out of the total of ", len(cones_left))
    print("First cone right index: ", first_cone_right_index, "out of the total of ", len(cones_right))

    cones_left = np.roll(cones_left, -first_cone_left_index, axis=0)
    cones_right = np.roll(cones_right, -first_cone_right_index, axis=0)

    return cones_left, cones_right