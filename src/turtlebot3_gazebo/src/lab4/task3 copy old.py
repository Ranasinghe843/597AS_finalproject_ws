#!/usr/bin/env python3

import sys
import os
import numpy as np
import math
import heapq
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32
from copy import copy
import yaml
from PIL import Image, ImageOps
from ament_index_python.packages import get_package_share_directory
import pandas as pd
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy
import time

class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)
        
    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        im = Image.open(map_name + '.pgm')
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array
    
class TreeNode():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []
        
    def __repr__(self):
        return self.name
        
    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)
    
class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
    
    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name
            
    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False
    
    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True

class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)
    
    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and 
            (i < map_array.shape[0]) and 
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value 
    
    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)
        
    def inflate_map(self, kernel, absolute=True, base_map=None):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        source_map = base_map if base_map is not None else self.map.image_array
        for i in range(source_map.shape[0]):
            for j in range(source_map.shape[1]):
                if source_map[i][j] == 0:
                    self.__inflate_obstacle(kernel, self.inf_map_img_array, i, j, absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r
                
    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = TreeNode('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:                    
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left 
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left 
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])                    
        
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm
    
    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m
    
    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array

class AStarFast:
    def __init__(self, in_tree):
        self.in_tree = in_tree

        # dist/heuristic/via as before
        self.dist = {name: float('inf') for name, _ in in_tree.g.items()}
        self.via  = {name: 0 for name, _ in in_tree.g.items()}

        # Preserve your heuristic exactly (Euclidean over indices parsed from "iy,ix")
        self.h = {}
        end = tuple(map(int, self.in_tree.end.split(',')))
        for name, _ in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            self.h[name] = math.hypot(dx, dy)

        # For tie-breaking identical to your stable sort of the full list:
        # use the original insertion index of each node in in_tree.g
        self._ord = {}
        for idx, name in enumerate(in_tree.g.keys()):
            self._ord[name] = idx

        # closed set to avoid re-expanding popped nodes (your old code effectively did this)
        self._closed = set()

    def __f(self, name):
        # f = g + h
        return self.dist[name] + self.h[name]

    def solve(self, sn, en):
        self._closed.clear()
        for k in self.dist.keys():
            self.dist[k] = float('inf')
            self.via[k]  = 0

        start = sn.name
        goal  = en.name
        self.dist[start] = 0.0

        # Min-heap of (f, tie, name). tie = original insertion order (stable-sort mimic)
        open_heap = []

        for name in self.in_tree.g.keys():
            f = self.__f(name)
            heapq.heappush(open_heap, (f, self._ord[name], name))

        while open_heap:
            f, tie, u = heapq.heappop(open_heap)

            # Skip stale entries (if a better g came later) or already closed
            if u in self._closed or f != self.__f(u):
                continue

            if u == goal:
                break

            self._closed.add(u)

            u_node = self.in_tree.g[u]
            ug = self.dist[u]

            # Expand in EXACT child order (your graph stored it)
            for i in range(len(u_node.children)):
                c = u_node.children[i]
                v = c.name
                w = u_node.weight[i]

                new_g = ug + w
                if new_g < self.dist[v]:
                    self.dist[v] = new_g
                    self.via[v]  = u
                    # Maintain the same tiebreak (insertion order) as your stable sort
                    heapq.heappush(open_heap, (self.dist[v] + self.h[v], self._ord[v], v))

    def reconstruct_path(self, sn, en):
        start_key = sn.name
        end_key   = en.name
        dist      = self.dist[end_key]
        if math.isinf(dist):
            return [], float('inf')

        u = end_key
        path = [u]
        while u != start_key:
            u = self.via[u]
            path.append(u)
        path.reverse()
        return path, dist

class Task3(Node):
    def __init__(self):
        super().__init__('task3_node')
        
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()

        self.goal_ready = False
        self.pose_ready = True
        self.plan_computed = False
        self.plan_dirty = False
        self.replan_needed = False
        self.goal_reached = True
        self.goal_list = [[7.57, -5.42], 
                          [2.61, 0.49],
                          [-6.52, -5.53]]
        self.current_goal = -1

        self.follow_idx = 0
        self.wp_reached_thresh = 0.25
        self.lookahead_dist = 0.6

        self.max_lin = 0.35                 # [m/s] cap linear speed
        self.max_ang = 1.2                  # [rad/s] cap angular speed
        self.k_heading = 1.8                # P-gain for heading correction
        self.slow_radius = 0.8              # [m] start slowing down when within this to target
        self.stop_radius = 0.12             # [m] consider goal reached; stop

        self.clearance_radius_cells = 4     # window (in cells) to probe for nearby obstacles
        self.narrow_proximity_thresh = 0.15 # > this in the inflated map window => narrow
        self.narrow_lookahead = 0.25        # [m] smaller lookahead in narrow spaces
        self.narrow_lin = 0.15              # [m/s] slower linear speed in narrow spaces
        self.rotate_only_err = 1.0          # [rad] if heading error > this, rotate in place

        qos_sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)
        self.create_subscription(LaserScan, '/scan', self.__scan_cb, qos_sensor)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(OccupancyGrid, '/debug_map', 10)

        # Map File Load
        self.map_file_path = os.path.join(get_package_share_directory('turtlebot3_gazebo'), "maps", "map")
        self.get_logger().info(self.map_file_path)
        self.mp_running = MapProcessor(self.map_file_path)
        self.running_kr = 6
        self.mp_running.inflate_map(self.mp_running.rect_kernel(self.running_kr,1), True)
        self.mp = MapProcessor(self.map_file_path)
        self.kr = 11
        self.mp.inflate_map(self.mp.rect_kernel(self.kr,1), True)
        self.mp.get_graph_from_map()
        self.graph_built = True
        self.map_ready = True
        h, w = self.mp.map.image_array.shape
        self.get_logger().info(
            f"Local map loaded: {w}x{h}, nodes={len(self.mp.map_graph.g)}"
        )

        self.rate = self.create_rate(10)

    def set_next_goal_pose(self):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_reached = False
        self.current_goal += 1
        self.goal_pose.header.frame_id = "map"
        self.goal_pose.header.stamp = self.get_clock().now().to_msg()
        self.goal_pose.pose.position.x = self.goal_list[self.current_goal][0]
        self.goal_pose.pose.position.y = self.goal_list[self.current_goal][1]
        self.goal_pose.pose.orientation.w = 1.0

        self.mp_running = MapProcessor(self.map_file_path)
        self.mp_running.inflate_map(self.mp_running.rect_kernel(self.running_kr,1), True)
        self.mp = MapProcessor(self.map_file_path)
        self.mp.inflate_map(self.mp.rect_kernel(self.kr,1), True)
        self.graph_built = False
        self.goal_ready = True
        self.plan_dirty = True
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y)
        )

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose.header = data.header
        self.ttbot_pose.pose = data.pose.pose
        self.ttbot_pose.header.frame_id = "map"
        self.pose_ready = True
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y)
        )
    
    def world_to_grid(self, x, y):
        
        res = self.mp.map.map_df.resolution[0]
        ox, oy, _ = self.mp.map.map_df.origin[0]
        H, W = self.mp.map.image_array.shape

        cx = (x - ox) / res
        cy = (y - oy) / res

        ix = int(math.floor(cx))
        iy_from_bottom = int(math.floor(cy))
        iy = (H - 1) - iy_from_bottom

        return ix, iy


    def grid_to_world(self, ix, iy):
        
        res = self.mp.map.map_df.resolution[0]
        ox, oy, _ = self.mp.map.map_df.origin[0]
        H, W = self.mp.map.image_array.shape

        iy_from_bottom = (H - 1) - iy

        x = ox + (ix + 0.5) * res
        y = oy + (iy_from_bottom + 0.5) * res
        return x, y
    
    def publish_debug_map(self):
        """Publishes the internal inflated map to Rviz."""
        if self.mp.inf_map_img_array is None:
            return

        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        # Copy metadata from the original map configuration
        msg.info.resolution = float(self.mp.map.map_df.resolution[0])
        msg.info.width = int(self.mp.map.image_array.shape[1])
        msg.info.height = int(self.mp.map.image_array.shape[0])
        
        # Origin
        msg.info.origin.position.x = float(self.mp.map.map_df.origin[0][0])
        msg.info.origin.position.y = float(self.mp.map.map_df.origin[0][1])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # --- THE FIX ---
        # 1. Flip the array upside down (Vertical Flip)
        # Because PIL/NumPy (0,0 is Top-Left) vs ROS (0,0 is Bottom-Left)
        corrected_array = np.flipud(self.mp.inf_map_img_array)

        # 2. Flatten and Cast
        # Map 255 (Obstacle) -> 100 for Rviz
        flat_data = corrected_array.flatten()
        flat_data = np.where(flat_data > 0, 100, 0).astype(np.int8)
        
        msg.data = flat_data.tolist()
        self.debug_pub.publish(msg)
    
    def __scan_cb(self, msg):
        """! Callback to process Lidar data, update map, and trigger replanning."""
        if not self.pose_ready or not self.map_ready:
            return

        # 1. Get current robot pose
        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y
        yaw = self._yaw_from_quat(self.ttbot_pose.pose.orientation)

        ranges = np.array(msg.ranges)
        
        # Optimization: Only process obstacles within 1.0 meter. 
        valid_indices = np.where((ranges < 1.0) & (ranges > 0.05))[0]
        
        map_changed = False
        H, W = self.mp.inf_map_img_array.shape
        
        # Kernel Radius: You used rect_kernel(11,1), so radius is 5 (11 // 2)
        #kr_inf_radius = self.kr // 2
        kr_inf_radius = 2
        kr_running_inf_radius = self.running_kr // 2

        for i in valid_indices:
            r = ranges[i]
            theta = msg.angle_min + i * msg.angle_increment
            
            # 2. Global Coordinates
            ox = rx + r * math.cos(yaw + theta)
            oy = ry + r * math.sin(yaw + theta)
            
            # 3. Grid Coordinates
            ix, iy = self.world_to_grid(ox, oy)

            # 4. Update Map LOCALLY
            if 0 <= ix < W and 0 <= iy < H:
                # Check center pixel. If it's 0 (free), this is a NEW obstacle.
                if self.mp_running.inf_map_img_array[iy, ix] == 0:
                    map_changed = True

                    x_min = max(0, ix - kr_running_inf_radius)
                    x_max = min(W, ix + kr_running_inf_radius + 1)
                    y_min = max(0, iy - kr_running_inf_radius)
                    y_max = min(H, iy + kr_running_inf_radius + 1)
                    self.mp_running.inf_map_img_array[y_min:y_max, x_min:x_max] = 100
                    
                    x_min = max(0, ix - kr_inf_radius)
                    x_max = min(W, ix + kr_inf_radius + 1)
                    y_min = max(0, iy - kr_inf_radius)
                    y_max = min(H, iy + kr_inf_radius + 1)
                    self.mp.inf_map_img_array[y_min:y_max, x_min:x_max] = 100

        # 5. Collision Check & Re-planning
        # Only runs if we actually added new obstacles
        if map_changed and self.path.poses:
            # Check the next 20 waypoints to see if the new obstacle blocks our path
            start_check = self.follow_idx
            end_check = min(len(self.path.poses), self.follow_idx + 20)
            
            collision_detected = False
            for k in range(start_check, end_check):
                wp = self.path.poses[k].pose.position
                wx, wy = self.world_to_grid(wp.x, wp.y)
                
                if 0 <= wx < W and 0 <= wy < H:
                    # If path waypoint is now inside an obstacle (value > 0)
                    if self.mp.inf_map_img_array[wy, wx] > 0:
                        collision_detected = True
                        break
            
            if collision_detected:
                self.get_logger().warn("Path Blocked by Dynamic Obstacle! Replanning...")
                self.move_ttbot(0.0, 0.0) # Stop immediately
                self.path = Path()        # Clear current path
                self.plan_dirty = True    # Request A*
                self.graph_built = False  # Force graph rebuild

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        self.get_logger().info(
            f'A* planner.\n> start: {start_pose.pose.position}\n> end:   {end_pose.pose.position}')

        # Build graph once (after inflation)
        if not self.graph_built:
            self.mp.get_graph_from_map()
            self.graph_built = True
            self.get_logger().info(f"Graph built: {len(self.mp.map_graph.g)} free nodes.")

        # Convert start/goal to grid node keys "iy,ix"
        s_ix, s_iy = self.world_to_grid(start_pose.pose.position.x, start_pose.pose.position.y)
        g_ix, g_iy = self.world_to_grid(end_pose.pose.position.x,  end_pose.pose.position.y)
        s_key = f"{s_iy},{s_ix}"
        g_key = f"{g_iy},{g_ix}"

        self.get_logger().info(s_key)
        self.get_logger().info(g_key)

        if s_key not in self.mp.map_graph.g or g_key not in self.mp.map_graph.g:
            self.get_logger().warn("Start or goal in obstacle/outside free space")
            # path.poses.append(start_pose); path.poses.append(end_pose)
            # path.poses = None
            return path
        
        self.mp.map_graph.root = s_key
        self.mp.map_graph.end = g_key
        solver = AStarFast(self.mp.map_graph)

        solver.solve(self.mp.map_graph.g[self.mp.map_graph.root], self.mp.map_graph.g[self.mp.map_graph.end])
        
        path_as, dist_as = solver.reconstruct_path(self.mp.map_graph.g[self.mp.map_graph.root], self.mp.map_graph.g[self.mp.map_graph.end])

        # Fill Path (grid → world)
        for nm in path_as:
            iy, ix = map(int, nm.split(','))
            wx, wy = self.grid_to_world(ix, iy)
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self.get_logger().info(f"A* done")
        
        return path

    def get_path_idx(self, path, vehicle_pose):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        n = len(path.poses)
        if n == 0:
            return 0

        self.follow_idx = min(self.follow_idx, n - 1)

        robot_p = vehicle_pose.pose.position

        # advance when close to current waypoint
        while self.follow_idx < (n - 1):
            wp_p = path.poses[self.follow_idx].pose.position
            if math.hypot(robot_p.x - wp_p.x, robot_p.y - wp_p.y) <= self.wp_reached_thresh:
                self.follow_idx += 1
            else:
                break

        return self.follow_idx
    
    def _yaw_from_quat(self, q):
        # q is geometry_msgs/Quaternion
        # yaw from quaternion (Z rotation)
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def _proximity_inflated(self, ix, iy, R):
        """
        Check the maximum inflated-map value in a (2R+1)x(2R+1) window.
        0 = far from obstacles; higher -> closer to inflated obstacles.
        """
        a = self.mp.inf_map_img_array
        H, W = a.shape
        x0 = max(0, ix - R); x1 = min(W - 1, ix + R)
        y0 = max(0, iy - R); y1 = min(H - 1, iy + R)
        return float(np.max(a[y0:y1+1, x0:x1+1]))


    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        rx = vehicle_pose.pose.position.x
        ry = vehicle_pose.pose.position.y
        gx = current_goal_pose.pose.position.x
        gy = current_goal_pose.pose.position.y

        # yaw & heading error
        yaw = self._yaw_from_quat(vehicle_pose.pose.orientation)
        desired = math.atan2(gy - ry, gx - rx)
        err = (desired - yaw + math.pi) % (2.0*math.pi) - math.pi

        d = math.hypot(gx - rx, gy - ry)

        # defaults
        max_lin = self.max_lin
        max_ang = self.max_ang

        # narrow passage detection
        narrow = False
        if self.map_ready:
            ix, iy = self.world_to_grid(rx, ry)
            prox = self._proximity_inflated(ix, iy, self.clearance_radius_cells)
            narrow = (prox > self.narrow_proximity_thresh)

        # rotate-in-place if facing far away from target direction
        if abs(err) > self.rotate_only_err:
            v = 0.0
            w = max(-max_ang, min(max_ang, self.k_heading * err))
            return v, w

        # speed profile
        speed_scale_heading = max(0.0, 0.5 + 0.5 * math.cos(err))  # softer than pure cos(err)
        speed_scale_dist    = min(1.0, d / max(self.slow_radius, 1e-6))
        v_cmd = max_lin * speed_scale_heading * speed_scale_dist

        # narrow corridor: slow down and allow more angular authority
        if narrow:
            v_cmd = min(v_cmd, self.narrow_lin)
            max_ang = max_ang  # keep same cap; you can increase if needed

        w_cmd = self.k_heading * err
        w_cmd = max(-max_ang, min(max_ang, w_cmd))

        # stop near final goal
        if self.path and self.follow_idx >= len(self.path.poses) - 2:
            final_p = self.path.poses[-1].pose.position
            d_goal = math.hypot(final_p.x - rx, final_p.y - ry)
            if d_goal < self.stop_radius:
                self.goal_reached = True
                return 0.0, 0.0

        return v_cmd, w_cmd

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)
    
    def _tick(self):
        self.publish_debug_map()
        if self.goal_reached and self.current_goal < (len(self.goal_list) - 1):
            self.set_next_goal_pose()
        if self.map_ready and self.pose_ready and self.goal_ready and (self.plan_dirty or not self.plan_computed):
            self.follow_idx = 0
            self.path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            self.path_pub.publish(self.path)
            self.plan_computed = True
            self.plan_dirty = False
        if self.path.poses:
            idx = self.get_path_idx(self.path, self.ttbot_pose)
            self.get_logger().info(f"Index: {idx}")
            current_goal = self.path.poses[idx]
            self.get_logger().info(f"Current: {current_goal.pose.position.x}, Goal: {current_goal.pose.position.y}")
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.get_logger().info(f"Speed: {speed}, Heading: {heading}")
            self.move_ttbot(speed, heading)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """
        self.create_timer(0.1, self._tick)
        self.get_logger().info("Timer started; entering rclpy.spin()")
        rclpy.spin(self)

def main(args=None):
    rclpy.init(args=args)
    task3 = Task3()

    try:
        task3.run()
    except KeyboardInterrupt:
        pass
    finally:
        task3.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()