#!/usr/bin/env python3
import os
import cv2
import yaml
import time
import rclpy
import heapq
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from copy import copy
from rclpy.node import Node
from PIL import Image, ImageOps
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist


# ======================================================================
# MAP + A* SUPPORT CLASSES
# ======================================================================

class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits, self.map_cv2 = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array, extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self, map_name):
        package_share_directory = get_package_share_directory('turtlebot3_gazebo')
        map_path = os.path.join(package_share_directory, 'maps')

        with open(os.path.join(map_path, map_name + '.yaml'), 'r') as f:
            map_df = pd.json_normalize(yaml.safe_load(f))

        img_name = map_df.image[0]
        im = Image.open(os.path.join(map_path, img_name))
        im = ImageOps.grayscale(im)

        map_image = cv2.imread(os.path.join(map_path, img_name), cv2.IMREAD_GRAYSCALE)
        map_bin = cv2.threshold(map_image, 200, 255, cv2.THRESH_BINARY)[1]

        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin, xmax, ymin, ymax], map_bin

    def __get_obstacle_map(self, map_im, map_df):
        """
        image_array:
          0   = obstacle (seed for inflation)
          255 = free
        """
        img_array = np.reshape(
            list(self.map_im.getdata()),
            (self.map_im.size[1], self.map_im.size[0])
        )
        up_thresh = self.map_df.occupied_thresh[0] * 255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i, j] > up_thresh:
                    img_array[i, j] = 255   # free
                else:
                    img_array[i, j] = 0     # obstacle

        return img_array


class TreeNode():
    def __init__(self, name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self, node, w=None):
        if w is None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)


class Tree():
    def __init__(self, name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}

    def __call__(self):
        for name, node in self.g.items():
            if self.root == name:
                self.g_visual.node(name, name, color='red')
            elif self.end == name:
                self.g_visual.node(name, name, color='blue')
            else:
                self.g_visual.node(name, name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                if w == 0:
                    self.g_visual.edge(name, c.name)
                else:
                    self.g_visual.edge(name, c.name, label=str(w))
        return self.g_visual

    def add_node(self, node, start=False, end=False):
        self.g[node.name] = node
        if start:
            self.root = node.name
        elif end:
            self.end = node.name

    def set_as_root(self, node):
        self.root = True
        self.end = False

    def set_as_end(self, node):
        self.root = False
        self.end = True


class AStar():
    def __init__(self, in_tree):
        self.in_tree = in_tree
        self.dist = {name: np.inf for name in in_tree.g}
        self.h = {}

        end = tuple(map(int, in_tree.end.split(',')))
        for name in in_tree.g:
            start = tuple(map(int, name.split(',')))
            self.h[name] = np.hypot(end[0] - start[0], end[1] - start[1])

        self.via = {name: None for name in in_tree.g}
        self.open_set = []
        self._counter = 0

    def __get_f_score(self, node):
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        self.dist[sn.name] = 0
        heapq.heappush(self.open_set, (self.__get_f_score(sn), self._counter, sn))
        self._counter += 1

        visited = set()

        while self.open_set:
            _, __, u = heapq.heappop(self.open_set)

            if u.name in visited:
                continue
            visited.add(u.name)

            if u.name == en.name:
                break

            for c, w in zip(u.children, u.weight):
                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name
                    heapq.heappush(
                        self.open_set,
                        (self.__get_f_score(c), self._counter, c)
                    )
                    self._counter += 1

        return self.via

    def reconstruct_path(self, sn, en):
        start_key, end_key = sn.name, en.name
        if self.dist[end_key] == np.inf:
            return [], np.inf

        path = [end_key]
        while path[-1] != start_key:
            prev = self.via[path[-1]]
            if prev is None:
                return [], np.inf
            path.append(prev)
        path.reverse()
        return path, self.dist[end_key]


class MapProcessor():
    def __init__(self, name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self, map_array, i, j, value, absolute):
        if ((0 <= i < map_array.shape[0]) and
                (0 <= j < map_array.shape[1])):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self, kernel, map_array, i, j, absolute):
        dx = int(kernel.shape[0] // 2)
        dy = int(kernel.shape[1] // 2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array, i, j, kernel[0][0], absolute)
        else:
            for k in range(i - dx, i + dx):
                for l in range(j - dy, j + dy):
                    self.__modify_map_pixel(
                        map_array,
                        k,
                        l,
                        kernel[k - i + dx][l - j + dy],
                        absolute
                    )

    def inflate_map(self, kernel, absolute=True, base_map=None):
        """
        Inflate STATIC obstacles:
        - self.map.image_array: 0 = obstacle, 255 = free (base map)
        - self.inf_map_img_array: 0 = far from obstacles, >0 = inflated / unsafe
        """
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        source_map = base_map if base_map is not None else self.map.image_array
        for i in range(source_map.shape[0]):
            for j in range(source_map.shape[1]):
                if source_map[i][j] == 0:   # obstacle pixel in base map
                    self.__inflate_obstacle(kernel, self.inf_map_img_array, i, j, absolute)

        r = np.max(self.inf_map_img_array) - np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (
            self.inf_map_img_array - np.min(self.inf_map_img_array)
        ) / r

    def get_graph_from_map(self):
        """
        Free space = cells where inf_map_img_array == 0
        Any >0 is treated as blocked/unsafe.
        """
        # nodes
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = TreeNode('%d,%d' % (i, j))
                    self.map_graph.add_node(node)

        # edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if i > 0 and self.inf_map_img_array[i-1][j] == 0:
                        child_up = self.map_graph.g['%d,%d' % (i-1, j)]
                        self.map_graph.g['%d,%d' % (i, j)].add_children([child_up], [1])
                    if i < (self.map.image_array.shape[0] - 1) and self.inf_map_img_array[i+1][j] == 0:
                        child_dw = self.map_graph.g['%d,%d' % (i+1, j)]
                        self.map_graph.g['%d,%d' % (i, j)].add_children([child_dw], [1])
                    if j > 0 and self.inf_map_img_array[i][j-1] == 0:
                        child_lf = self.map_graph.g['%d,%d' % (i, j-1)]
                        self.map_graph.g['%d,%d' % (i, j)].add_children([child_lf], [1])
                    if j < (self.map.image_array.shape[1] - 1) and self.inf_map_img_array[i][j+1] == 0:
                        child_rg = self.map_graph.g['%d,%d' % (i, j+1)]
                        self.map_graph.g['%d,%d' % (i, j)].add_children([child_rg], [1])
                    if i > 0 and j > 0 and self.inf_map_img_array[i-1][j-1] == 0:
                        child_up_lf = self.map_graph.g['%d,%d' % (i-1, j-1)]
                        self.map_graph.g['%d,%d' % (i, j)].add_children([child_up_lf], [np.sqrt(2)])
                    if i > 0 and j < (self.map.image_array.shape[1] - 1) and self.inf_map_img_array[i-1][j+1] == 0:
                        child_up_rg = self.map_graph.g['%d,%d' % (i-1, j+1)]
                        self.map_graph.g['%d,%d' % (i, j)].add_children([child_up_rg], [np.sqrt(2)])
                    if i < (self.map.image_array.shape[0] - 1) and j > 0 and self.inf_map_img_array[i+1][j-1] == 0:
                        child_dw_lf = self.map_graph.g['%d,%d' % (i+1, j-1)]
                        self.map_graph.g['%d,%d' % (i, j)].add_children([child_dw_lf], [np.sqrt(2)])
                    if (i < (self.map.image_array.shape[0] - 1) and
                            j < (self.map.image_array.shape[1] - 1) and
                            self.inf_map_img_array[i+1][j+1] == 0):
                        child_dw_rg = self.map_graph.g['%d,%d' % (i+1, j+1)]
                        self.map_graph.g['%d,%d' % (i, j)].add_children([child_dw_rg], [np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g) - np.min(g)
        sm = (g - np.min(g)) * 1 / r
        return sm

    def rect_kernel(self, size, value):
        return np.ones(shape=(size, size)) * value

    def draw_path(self, path):
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_array[tup] = 0.5
        return path_array


# ======================================================================
# TASK 2 NODE
# ======================================================================

class Task2(Node):
    """
    Path planning + following with dynamic obstacle integration.
    """

    def __init__(self, node_name='task2_node'):
        super().__init__(node_name)

        # Path planner/follower related variables
        self.path = None
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

        self.follow_idx = 0
        self.goal_pose_given = False
        self.ttbot_pose_given = False

        # PID state variables (unchanged controller from Version 1)
        self.PID_lin_int = 0.0
        self.PID_lin_prev = 0.0
        self.PID_ang_int = 0.0
        self.PID_ang_prev = 0.0
        self.last_time = self.get_clock().now().nanoseconds * 1e-9

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal',
                                 self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                 self.__ttbot_pose_cbk, 10)
        self.create_subscription(LaserScan, '/scan',
                                 self.__scan_cbk, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time', 10)  # DO NOT MODIFY
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, '/inflated_map', 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/dynamic_obstacle_markers", 10)

        # Node rate
        self.rate = self.create_rate(10)

        # ------------------------------------------------------------------
        # Map generation (STATIC inflation) + hybrid dynamic maps
        # ------------------------------------------------------------------
        # Global planning map: moderate inflation (smaller kernel)
        mp = MapProcessor('sync_classroom_map')
        self.kr_static = 5                       # planner inflation kernel size
        kr = mp.rect_kernel(self.kr_static, 1)
        mp.inflate_map(kr, True)
        mp.get_graph_from_map()

        self.res = mp.map.map_df['resolution'][0]
        self.origin = mp.map.map_df['origin'][0]
        self.shape = mp.map.map_cv2.shape[0]
        self.mp = mp

        # Local running map: more conservative inflation for detection/safety
        self.mp_running = MapProcessor('sync_classroom_map')
        self.running_kr = 11                     # larger local safety bubble
        kr_run = self.mp_running.rect_kernel(self.running_kr, 1)
        self.mp_running.inflate_map(kr_run, True)

        # dynamic obstacle / replanning parameters
        self.block_threshold = 1.2   # meters (max LiDAR range for dynamic obs)
        self.replan_flag = False

        # ------------------------------------------------------------------
        # Dynamic obstacle visualization
        # ------------------------------------------------------------------
        self.dynamic_cells = []  # for marker viz only

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def __goal_pose_cbk(self, data):
        self.goal_pose_given = True
        self.goal_pose = data
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(
                self.goal_pose.pose.position.x,
                self.goal_pose.pose.position.y)
        )

    def __ttbot_pose_cbk(self, data):
        self.ttbot_pose_given = True
        self.ttbot_pose = data.pose
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(
                self.ttbot_pose.pose.position.x,
                self.ttbot_pose.pose.position.y)
        )

    def __scan_cbk(self, scan):
        """
        Dynamic obstacle integration (Version 2-style, Hybrid C):
        - Transform LiDAR range hits into map coordinates (using yaw)
        - Use a conservative local map (mp_running) to detect NEW obstacles
        - Update global planner map (mp) with a smaller inflation window
        - If current path segment is blocked -> set replan_flag
        """

        if not self.ttbot_pose_given:
            return

        # robot pose
        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y
        yaw = self.calc_heading(self.ttbot_pose.pose.orientation)

        ranges = np.array(scan.ranges)
        # consider only reasonable hits
        valid_idx = np.where(
            (ranges < self.block_threshold) &
            (ranges > 0.05)
        )[0]

        if valid_idx.size == 0:
            return

        grid_run = self.mp_running.inf_map_img_array
        grid_pln = self.mp.inf_map_img_array
        h, w = grid_pln.shape

        # kernel radii (cells)
        kr_run_rad = self.running_kr // 2    # larger for safety
        kr_pln_rad = self.kr_static // 2     # smaller for planner

        map_changed = False

        for i in valid_idx:
            r = ranges[i]
            angle = scan.angle_min + i * scan.angle_increment

            # world coordinates of hit (map frame)
            ox = rx + r * np.cos(yaw + angle)
            oy = ry + r * np.sin(yaw + angle)

            # grid coordinates (cols,rows) = (gx,gy)
            gx, gy = self.convert_coordinates(ox, oy, grid=True)

            if not (0 <= gy < h and 0 <= gx < w):
                continue

            # mp_running.inf_map_img_array:
            #   0   = far from obstacles (free)
            #  >0   = near static or previously added obstacle
            if grid_run[gy, gx] > 0.0:
                # already considered occupied / near obstacle
                continue

            # NEW obstacle in truly free space
            map_changed = True

            # --- update local safety map with larger neighborhood ---
            x_min = max(0, gx - kr_run_rad)
            x_max = min(w, gx + kr_run_rad + 1)
            y_min = max(0, gy - kr_run_rad)
            y_max = min(h, gy + kr_run_rad + 1)
            grid_run[y_min:y_max, x_min:x_max] = 1.0

            # --- update global planner map with smaller neighborhood ---
            x_min_p = max(0, gx - kr_pln_rad)
            x_max_p = min(w, gx + kr_pln_rad + 1)
            y_min_p = max(0, gy - kr_pln_rad)
            y_max_p = min(h, gy + kr_pln_rad + 1)
            grid_pln[y_min_p:y_max_p, x_min_p:x_max_p] = 1.0

            # record for visualization (optional)
            self.dynamic_cells.append((gy, gx))
            self.get_logger().warn(f"[DYNAMIC] New obstacle near cell ({gy},{gx})")

        if not map_changed:
            return

        # If we already have a path, see if the next chunk is now blocked
        if self.path is None or len(self.path.poses) == 0:
            return

        start_check = self.follow_idx
        end_check = min(len(self.path.poses), self.follow_idx + 20)
        collision_detected = False

        for k in range(start_check, end_check):
            pose = self.path.poses[k].pose.position
            gx, gy = self.convert_coordinates(
                pose.x,
                pose.y,
                grid=True
            )
            if 0 <= gy < h and 0 <= gx < w:
                if grid_pln[gy, gx] > 0.0:
                    self.get_logger().warn(
                        f"[COLLISION] Dynamic obstacle on path near cell ({gy},{gx})"
                    )
                    collision_detected = True
                    break

        if collision_detected:
            self.replan_flag = True

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def publish_inflated_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # metadata
        meta = MapMetaData()
        meta.resolution = float(self.res)
        meta.width = self.mp.inf_map_img_array.shape[1]
        meta.height = self.mp.inf_map_img_array.shape[0]
        meta.origin.position.x = float(self.origin[0])
        meta.origin.position.y = float(self.origin[1])
        msg.info = meta

        # 0 (free) .. 100 (inflated / occupied)
        grid = (self.mp.inf_map_img_array * 100.0).astype(np.int8)
        msg.data = np.flipud(grid).flatten().tolist()

        self.inflated_map_pub.publish(msg)

    def publish_obstacle_markers(self):
        """Publish RViz markers for each detected dynamic obstacle (planner grid)."""
        markers = MarkerArray()
        t = self.get_clock().now().to_msg()

        for idx, (gy, gx) in enumerate(self.dynamic_cells):
            wx, wy = self.convert_coordinates(gx, gy, grid=False)

            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = t
            m.ns = "dynamic_obstacles"
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.05

            m.scale.x = 0.18
            m.scale.y = 0.18
            m.scale.z = 0.18

            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.85

            m.lifetime.sec = 0
            markers.markers.append(m)

        self.marker_pub.publish(markers)

    # ------------------------------------------------------------------
    # A* planner
    # ------------------------------------------------------------------

    def a_star_path_planner(self, start_pose, end_pose):
        """
        A* path planner using current self.mp.inf_map_img_array,
        which already includes static + dynamic inflation.
        """
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(
                start_pose.pose.position,
                end_pose.pose.position)
        )
        self.start_time = self.get_clock().now().nanoseconds * 1e-9  # DO NOT EDIT

        # Rebuild graph based on current inflated map
        self.mp.map_graph = Tree("dynamic")
        mp = self.mp

        # start position in grid
        x, y = self.convert_coordinates(
            start_pose.pose.position.x,
            start_pose.pose.position.y,
            grid=True
        )
        mp.map_graph.root = f"{y},{x}"
        mp.get_graph_from_map()
        start_node = mp.map_graph.g[mp.map_graph.root]

        # end position in grid
        x, y = self.convert_coordinates(
            end_pose.pose.position.x,
            end_pose.pose.position.y,
            grid=True
        )
        mp.map_graph.end = f"{y},{x}"
        end_node = mp.map_graph.g[mp.map_graph.end]

        # run A*
        as_maze = AStar(mp.map_graph)
        self.get_logger().info('Running A*...')
        t0 = time.time()
        as_maze.solve(start_node, end_node)
        t1 = time.time()
        self.get_logger().info(f"A* Solved in {(t1 - t0):.8f}s")
        path_as, dist_as = as_maze.reconstruct_path(start_node, end_node)

        # convert back to world coordinates
        for pose in path_as:
            temp = PoseStamped()
            temp.header = path.header
            y, x = map(float, pose.split(','))
            cx, cy = self.convert_coordinates(x, y, grid=False)
            temp.pose.position.x = cx
            temp.pose.position.y = cy
            path.poses.append(temp)

        # DO NOT EDIT BELOW
        self.astarTime = Float32()
        self.astarTime.data = float(
            self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        )
        self.calc_time_pub.publish(self.astarTime)

        return path

    # ------------------------------------------------------------------
    # Path following (unchanged Version 1 controller)
    # ------------------------------------------------------------------

    def get_path_idx(self, path, vehicle_pose):
        """
        Choose next waypoint index based on current position.
        """
        ADVANCE_DIST_THRESHOLD = 0.2

        n = len(path.poses)
        if n == 0:
            return 0

        self.follow_idx = min(self.follow_idx, n - 1)

        vehicle = vehicle_pose.pose.position
        while self.follow_idx < n - 1:
            waypoint = path.poses[self.follow_idx].pose.position
            dist = np.hypot(vehicle.x - waypoint.x, vehicle.y - waypoint.y)
            if dist <= ADVANCE_DIST_THRESHOLD:
                self.follow_idx += 1
            else:
                break

        return self.follow_idx

    def path_follower(self, vehicle_pose, current_goal_pose):
        """
        Differential drive path follower using PID control.
        Returns (v, w).  (Same as Version 1.)
        """

        MAX_LIN = 0.3
        MAX_ANG = 0.6

        ROTATE_THRESHOLD = 0.30
        GOAL_TOL = 0.20

        KP_lin = 0.9
        KI_lin = 0.001
        KD_lin = 0.05

        KP_ang = 2.1
        KI_ang = 0.001
        KD_ang = 0.19

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(1e-6, now - self.last_time)
        self.last_time = now

        vx, vy = vehicle_pose.pose.position.x, vehicle_pose.pose.position.y
        tx, ty = current_goal_pose.pose.position.x, current_goal_pose.pose.position.y

        err_dist = np.hypot(tx - vx, ty - vy)

        yaw = self.calc_heading(vehicle_pose.pose.orientation)
        desired_yaw = np.arctan2(ty - vy, tx - vx)
        err_ang = self.normalize(desired_yaw - yaw)

        # rotate in place if heading error is large
        if abs(err_ang) > ROTATE_THRESHOLD:
            d_err = (err_ang - self.PID_ang_prev) / dt
            self.PID_ang_int += err_ang * dt
            w_cmd = KP_ang * err_ang + KI_ang * self.PID_ang_int + KD_ang * d_err

            w_cmd = float(np.clip(w_cmd, -MAX_ANG, MAX_ANG))

            self.PID_ang_prev = err_ang
            return 0.0, w_cmd

        # angular PID
        d_err = (err_ang - self.PID_ang_prev) / dt
        self.PID_ang_int += err_ang * dt
        w_cmd = KP_ang * err_ang + KI_ang * self.PID_ang_int + KD_ang * d_err
        self.PID_ang_prev = err_ang
        w_cmd = float(np.clip(w_cmd, -MAX_ANG, MAX_ANG))

        # linear PID
        d_err_lin = (err_dist - self.PID_lin_prev) / dt
        self.PID_lin_int += err_dist * dt
        v_cmd = KP_lin * err_dist + KI_lin * self.PID_lin_int + KD_lin * d_err_lin
        self.PID_lin_prev = err_dist
        v_cmd = float(np.clip(v_cmd, -MAX_LIN, MAX_LIN))

        # final goal check
        if self.path and self.follow_idx >= len(self.path.poses) - 2:
            final = self.path.poses[-1].pose.position
            final_dist = np.hypot(final.x - vx, final.y - vy)
            if final_dist < GOAL_TOL:
                return 0.0, 0.0

        return v_cmd, w_cmd

    # ------------------------------------------------------------------
    # Main loop + utilities
    # ------------------------------------------------------------------

    def move_ttbot(self, speed, heading):
        cmd_vel = Twist()
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading
        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self)

            # publish current inflated map + markers for RViz
            # self.publish_inflated_map()
            # self.publish_obstacle_markers()

            # wait for goal
            if not self.goal_pose_given:
                self.get_logger().info("Waiting for goal pose to be received...")
                self.rate.sleep()
                continue

            # wait for pose
            if not self.ttbot_pose_given:
                self.rate.sleep()
                continue

            # avoid degenerate goal == start
            if (self.goal_pose.pose.position.x == self.ttbot_pose.pose.position.x and
                    self.goal_pose.pose.position.y == self.ttbot_pose.pose.position.y):
                self.get_logger().warn('Invalid goal pose.')
                self.goal_pose_given = False
                self.rate.sleep()
                continue

            self.ttbot_pose_given = False

            # initial path
            if self.path is None:
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                if path is None or len(path.poses) == 0:
                    self.get_logger().warn("Path planner returned an empty path!")
                    self.rate.sleep()
                    continue

                self.path_pub.publish(path)
                self.get_logger().info('Path has been generated')
                self.get_logger().info(f'Number of poses on path: {len(path.poses)}')
                self.path = path
            else:
                path = self.path

            idx = self.get_path_idx(path, self.ttbot_pose)
            self.get_logger().info(f'Current path index: {idx}')

            # dynamic replan (Version 2-style trigger)
            if self.replan_flag:
                self.get_logger().warn("Replanning due to dynamic obstacle...")
                # stop robot while replanning
                self.move_ttbot(0.0, 0.0)

                # compute new path using updated mp.inf_map_img_array
                new_path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)

                if new_path is not None and len(new_path.poses) > 0:
                    self.path = new_path
                    self.path_pub.publish(self.path)
                    self.follow_idx = 0
                    self.get_logger().warn("New path committed!")
                else:
                    self.get_logger().error("Replan failed — no alternate path found.")

                self.replan_flag = False
                self.rate.sleep()
                continue

            # normal motion
            current_goal = path.poses[idx]
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(speed, heading)

            # goal reached
            if idx >= len(path.poses):
                self.get_logger().info('Goal Pose Reached!')
                self.move_ttbot(0.0, 0.0)
                self.rate.sleep()
                continue

            self.rate.sleep()

    # ------------------------------------------------------------------
    # Coordinate transforms + helpers
    # ------------------------------------------------------------------

    def convert_coordinates(self, x, y, grid=True):
        """
        grid=True : world (x,y) -> grid (cx,cy)
        grid=False: grid (cx,cy) -> world (x,y)
        """
        if grid:
            cx = int((x - self.origin[0]) / self.res)
            cy = int((self.origin[1] + self.shape * self.res - y) / self.res)
        else:
            cx = (x * self.res) + self.origin[0]
            cy = (self.shape * self.res + self.origin[1]) - (y * self.res)
        return cx, cy

    @staticmethod
    def normalize(a):
        return (a + np.pi) % (2.0*np.pi) - np.pi

    def calc_heading(self, q):
        return np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )


def main(args=None):
    rclpy.init(args=args)
    nav = Task2(node_name='task2_node')

    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()