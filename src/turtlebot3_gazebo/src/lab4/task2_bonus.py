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
import random
import time

class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)
        
    def __open_map(self,map_name):
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        im = Image.open(map_name + '.pgm')
        im = ImageOps.grayscale(im)
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
    
class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
    
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
        
    def inflate_map(self,kernel,absolute=True):
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r
        
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
    
class NodeRRT:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, map_processor, 
                 expand_dis=0.3, path_resolution=0.05, goal_sample_rate=10, max_iter=500):
        self.start = NodeRRT(start[0], start[1])
        self.goal = NodeRRT(goal[0], goal[1])
        self.mp = map_processor
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.connect_circle_dist = 1.0 

    def plan(self):
        """Main RRT* Loop"""
        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd, self.expand_dis)

            if self.check_collision(new_node):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

        last_index = self.search_best_goal_node()
        if last_index is None:
            return None
        
        return self.generate_final_course(last_index)

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node, math.hypot(new_node.x - near_node.x, new_node.y - near_node.y))
            if t_node and self.check_collision(t_node):
                costs.append(near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y))
            else:
                costs.append(float("inf"))

        min_cost = min(costs)
        if min_cost == float("inf"):
            return None
        
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node, math.hypot(new_node.x - self.node_list[min_ind].x, new_node.y - self.node_list[min_ind].y))
        new_node.cost = min_cost
        new_node.parent = self.node_list[min_ind]
        
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node: continue

            edge_node.cost = new_node.cost + math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)

            if self.check_collision(edge_node) and near_node.cost > edge_node.cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.parent = new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [math.hypot(n.x - self.goal.x, n.y - self.goal.y) for n in self.node_list]
        goal_inds = [i for i, d in enumerate(dist_to_goal_list) if d <= self.expand_dis]

        if not goal_inds: return None

        min_cost = min([self.node_list[i].cost for i in goal_inds])
        for i in goal_inds:
            if self.node_list[i].cost == min_cost:
                return i
        return None

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = NodeRRT(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(int(n_expand)):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            h, w = self.mp.inf_map_img_array.shape
            res = self.mp.map.map_df.resolution[0]
            ox = self.mp.map.map_df.origin[0][0]
            oy = self.mp.map.map_df.origin[0][1]
            
            rand_x = random.uniform(ox, ox + w*res)
            rand_y = random.uniform(oy, oy + h*res)
        else:
            rand_x = self.goal.x
            rand_y = self.goal.y
            
        return NodeRRT(rand_x, rand_y)

    def check_collision(self, node):
        if node is None: return False

        for ix, iy in zip(node.path_x, node.path_y):
            # Use map processor data for conversion
            res = self.mp.map.map_df.resolution[0]
            ox = self.mp.map.map_df.origin[0][0]
            oy = self.mp.map.map_df.origin[0][1]
            h, w = self.mp.inf_map_img_array.shape
            
            # World to Grid Conversion
            cx = (ix - ox) / res
            cy = (iy - oy) / res

            c = int(math.floor(cx))
            r_from_bottom = int(math.floor(cy))
            r = (h - 1) - r_from_bottom
            
            # Bounds Check
            if c < 0 or c >= w or r < 0 or r >= h:
                return False # Out of bounds
            
            # Obstacle Check
            if self.mp.inf_map_img_array[r, c] > 0:
                return False # Collision
                
        return True # Safe

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        r = min(r, self.connect_circle_dist)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_inds = [i for i, d in enumerate(dist_list) if d <= r**2]
        return near_inds

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

class Task2(Node):
    def __init__(self):
        super().__init__('task2_node')
        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

        self.goal_ready = False
        self.pose_ready = False
        self.plan_computed = False
        self.plan_dirty = False
        self.replan_needed = False

        # --- TUNED CONTROL PARAMETERS (From Working A*) ---
        self.follow_idx = 0
        self.lookahead_dist = 0.35  # Tighter lookahead for accuracy
        self.max_lin = 0.22         # Moderate speed
        self.max_ang = 1.0          # Cap angular speed
        self.k_heading = 1.0        # Smooth turning gain
        self.stop_radius = 0.10     # Stop proximity

        qos_sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)
        self.create_subscription(LaserScan, '/scan', self.__scan_cb, qos_sensor)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time',10) 
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
        self.map_ready = True

        self.rate = self.create_rate(10)

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.goal_pose.header.frame_id = "map"
        self.mp_running = MapProcessor(self.map_file_path)
        self.mp_running.inflate_map(self.mp_running.rect_kernel(self.running_kr,1), True)
        self.mp = MapProcessor(self.map_file_path)
        self.mp.inflate_map(self.mp.rect_kernel(self.kr,1), True)
        self.graph_built = False
        self.goal_ready = True
        self.plan_dirty = True
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))

    def __ttbot_pose_cbk(self, data):
        self.ttbot_pose.header = data.header
        self.ttbot_pose.pose = data.pose.pose
        self.ttbot_pose.header.frame_id = "map"
        self.pose_ready = True
        # self.get_logger().info('ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))
    
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

    def publish_debug_map(self):
        if self.mp.inf_map_img_array is None: return
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(self.mp.map.map_df.resolution[0])
        msg.info.width = int(self.mp.map.image_array.shape[1])
        msg.info.height = int(self.mp.map.image_array.shape[0])
        msg.info.origin.position.x = float(self.mp.map.map_df.origin[0][0])
        msg.info.origin.position.y = float(self.mp.map.map_df.origin[0][1])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        corrected_array = np.flipud(self.mp.inf_map_img_array)
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

    def plan_path_rrt(self):
        if not self.goal_ready or not self.pose_ready: return Path()
        
        self.get_logger().info("Planning Path (RRT*)...")
        self.move_ttbot(0.0, 0.0)
        
        start = [self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y]
        goal = [self.goal_pose.pose.position.x, self.goal_pose.pose.position.y]
        
        self.start_time = self.get_clock().now().nanoseconds*1e-9
        
        # Instantiate RRT*
        rrt_star = RRTStar(start, goal, self.mp, 
                           expand_dis=0.3, 
                           path_resolution=0.05, 
                           max_iter=3000)
        
        path_points = rrt_star.plan()

        # Publish Time
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        
        if path_points is None:
            self.get_logger().error("RRT* Failed to find path!")
            return Path()

        current_path = Path()
        current_path.header.frame_id = 'map'
        current_path.header.stamp = self.get_clock().now().to_msg()
        
        # RRT* returns Goal -> Start. Reverse it.
        for point in reversed(path_points):
            p = PoseStamped()
            p.pose.position.x = point[0]
            p.pose.position.y = point[1]
            p.pose.orientation.w = 1.0 
            current_path.poses.append(p)

        self.get_logger().info(f"RRT* Path Found with {len(current_path.poses)} waypoints.")
        return current_path

    def get_path_idx(self, path, vehicle_pose):
        """! Pure Pursuit Lookahead Logic."""
        if not path.poses:
            return 0
        n = len(path.poses)
        rx = vehicle_pose.pose.position.x
        ry = vehicle_pose.pose.position.y

        # Find closest point
        closest_dist = float('inf')
        closest_idx = self.follow_idx
        search_end = min(self.follow_idx + 50, n)

        for i in range(self.follow_idx, search_end):
            wx = path.poses[i].pose.position.x
            wy = path.poses[i].pose.position.y
            dist = math.hypot(wx - rx, wy - ry)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        self.follow_idx = closest_idx

        # Scan forward for lookahead point
        target_idx = closest_idx
        for i in range(closest_idx, n):
            wx = path.poses[i].pose.position.x
            wy = path.poses[i].pose.position.y
            dist = math.hypot(wx - rx, wy - ry)
            if dist > self.lookahead_dist:
                target_idx = i
                break
        
        if target_idx >= n:
            target_idx = n - 1
            
        return target_idx
    
    def _yaw_from_quat(self, q):
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def path_follower(self, vehicle_pose, current_goal_pose):
        """! P-Controller + Stop-And-Turn Logic."""
        rx = vehicle_pose.pose.position.x
        ry = vehicle_pose.pose.position.y
        gx = current_goal_pose.pose.position.x
        gy = current_goal_pose.pose.position.y

        yaw = self._yaw_from_quat(vehicle_pose.pose.orientation)
        desired_yaw = math.atan2(gy - ry, gx - rx)
        err = (desired_yaw - yaw + math.pi) % (2.0*math.pi) - math.pi

        # Angular Control
        w_cmd = self.k_heading * err
        w_cmd = max(-self.max_ang, min(self.max_ang, w_cmd))

        # Linear Control
        v_cmd = self.max_lin

        # Stop and Turn if error > 90 degrees
        if abs(err) > 1.57: 
            v_cmd = 0.0

        # Stop at goal
        if self.path and self.follow_idx >= len(self.path.poses) - 5:
            final_p = self.path.poses[-1].pose.position
            dist_to_end = math.hypot(final_p.x - rx, final_p.y - ry)
            if dist_to_end < self.stop_radius:
                return 0.0, 0.0

        return v_cmd, w_cmd

    def move_ttbot(self, speed, heading):
        cmd_vel = Twist()
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading
        self.cmd_vel_pub.publish(cmd_vel)
    
    def _tick(self):
        self.publish_debug_map()
        
        # Planning State
        if self.map_ready and self.pose_ready and self.goal_ready and (self.plan_dirty or not self.plan_computed):
            self.follow_idx = 0
            # Run RRT* Logic
            self.path = self.plan_path_rrt()
            self.path_pub.publish(self.path)
            
            if self.path.poses:
                self.plan_computed = True
                self.plan_dirty = False
            else:
                self.get_logger().warn("Retrying RRT*...")
        
        # Following State
        if self.path.poses:
            start_t = time.time()
            idx = self.get_path_idx(self.path, self.ttbot_pose)
            
            # self.get_logger().info(f"Index: {idx}")
            current_goal = self.path.poses[idx]
            
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(speed, heading)
            
            end_t = time.time()
            duration = end_t - start_t
            if duration > 0.1:
                self.get_logger().warn(f"Tick took too long! {duration:.4f} seconds")

    def run(self):
        self.create_timer(0.1, self._tick)
        self.get_logger().info("Timer started; entering rclpy.spin()")
        rclpy.spin(self)

def main(args=None):
    rclpy.init(args=args)
    task2 = Task2()
    try:
        task2.run()
    except KeyboardInterrupt:
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()