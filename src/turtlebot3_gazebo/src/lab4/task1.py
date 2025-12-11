#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Float32, ColorRGBA, Header
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import numpy as np
import math
import heapq
import time

# =========================================
# HELPER CLASS: Grid-Based A*
# =========================================
class GridAStar:
    def __init__(self, resolution, origin_x, origin_y, width, height):
        self.res = resolution
        self.ox = origin_x
        self.oy = origin_y
        self.w = width
        self.h = height

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.ox) / self.res)
        gy = int((wy - self.oy) / self.res)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = (gx * self.res) + self.ox + (self.res / 2.0)
        wy = (gy * self.res) + self.oy + (self.res / 2.0)
        return wx, wy

    def solve(self, grid, start_world, goal_world):
        sx, sy = self.world_to_grid(start_world[0], start_world[1])
        gx, gy = self.world_to_grid(goal_world[0], goal_world[1])

        if not (0 <= sx < self.w and 0 <= sy < self.h): return []
        if not (0 <= gx < self.w and 0 <= gy < self.h): return []

        open_set = []
        heapq.heappush(open_set, (0, (sx, sy)))
        came_from = {}
        g_score = { (sx, sy): 0 }
        
        # 8-Connected Neighbors
        motions = [
            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]

        found = False
        
        while open_set:
            _, current = heapq.heappop(open_set)

            if current == (gx, gy):
                found = True
                break

            cx, cy = current
            
            for dx, dy, cost in motions:
                nx, ny = cx + dx, cy + dy
                
                # Check Bounds
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    
                    # Strict check: Can only move through KNOWN FREE space (0)
                    # We treat Unknown (-1) as obstacle for path planning safety
                    if grid[ny, nx] != 0: 
                        continue

                    new_g = g_score[current] + cost
                    if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_g
                        priority = new_g + math.hypot(nx-gx, ny-gy)
                        heapq.heappush(open_set, (priority, (nx, ny)))
                        came_from[(nx, ny)] = current

        if not found: return []

        path = []
        curr = (gx, gy)
        while curr in came_from:
            wx, wy = self.grid_to_world(curr[0], curr[1])
            path.append((wx, wy))
            curr = came_from[curr]
        path.reverse()
        return path

# =========================================
# MAIN NODE
# =========================================
class Task1(Node):
    def __init__(self):
        super().__init__('task1_node')
        
        self.state = "START" 
        self.path = []
        
        # Maps
        self.raw_map = None      # The raw data from SLAM
        self.inflated_map = None # The processed map with safety buffers
        self.map_data = None 
        self.map_has_changed = False # New flag to track updates
        
        self.current_goal = None
        self.follow_idx = 0
        self.blacklist = []
        
        # Parameters
        self.lookahead_dist = 0.4
        self.inflation_radius = 5 # Cells (approx 20cm)
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Communication
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Vis
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.frontier_pub = self.create_publisher(Marker, 'frontiers', 10)
        self.debug_map_pub = self.create_publisher(OccupancyGrid, '/debug_map', 10)
        
        self.create_timer(0.1, self.control_loop)
    
    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return x, y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def map_cb(self, msg):
        self.map_data = msg.info
        # ROS data is row-major, convert to 2D numpy
        raw = np.array(msg.data, dtype=np.int8)
        self.raw_map = raw.reshape((msg.info.height, msg.info.width))
        self.map_has_changed = True # Trigger update logic

    def generate_inflated_map(self):
        """
        Creates a new map where walls are thicker.
        Returns the 2D numpy array of the inflated map.
        """
        if self.raw_map is None: return None
        
        # Copy raw map
        inflated = self.raw_map.copy()
        
        # Find walls (100)
        wall_indices = np.where(self.raw_map > 0)
        
        # If map is empty, return
        if len(wall_indices[0]) == 0: return inflated

        # Create a mask for walls to speed up dilation
        wall_mask = (self.raw_map > 0)
        
        # Dilate using array shifting (Faster than scipy for simple kernels)
        r = self.inflation_radius
        expanded_walls = wall_mask.copy()
        for y_shift in range(-r, r + 1):
            for x_shift in range(-r, r + 1):
                if x_shift == 0 and y_shift == 0: continue
                # Roll/Shift the array
                shifted = np.roll(wall_mask, shift=(y_shift, x_shift), axis=(0, 1))
                expanded_walls |= shifted
        
        # Apply inflated walls to the map (Mark as 100)
        inflated[expanded_walls] = 100
        
        return inflated

    def publish_debug_map(self, grid_2d):
        """Publishes the Inflated Map to Rviz for debugging."""
        if grid_2d is None: return
        
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.map_data 
        
        # Flatten and cast
        flat = grid_2d.flatten().astype(np.int8)
        msg.data = flat.tolist()
        
        self.debug_map_pub.publish(msg)

    def find_nearest_safe_point(self, cx, cy, grid, max_radius=10):
        """
        Spiral search to find the nearest grid cell with value 0 (Safe).
        Used when a frontier centroid falls inside an inflated buffer.
        """
        h, w = grid.shape
        if grid[cy, cx] == 0:
            return cx, cy

        # Spiral search
        x, y = 0, 0
        dx, dy = -1, 0 # Directions
        
        # We search up to max_radius squared cells roughly
        for _ in range(max_radius * max_radius * 4):
            if (-max_radius <= x <= max_radius) and (-max_radius <= y <= max_radius):
                nx, ny = cx + x, cy + y
                if 0 <= nx < w and 0 <= ny < h:
                    if grid[ny, nx] == 0:
                        return nx, ny
            
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
                dx, dy = -dy, dx # Turn 90 degrees
            x, y = x + dx, y + dy
            
        return None # Could not find safe point nearby

    def get_frontiers(self):
        """
        Modified Frontier Detection:
        1. Find frontiers on RAW map (Free touching Unknown).
        2. Cluster them.
        3. Check if the cluster centroid is in Safe Space on INFLATED map.
        4. If not, snap to nearest Safe Space pixel.
        """
        if self.raw_map is None or self.inflated_map is None: return None

        grid = self.raw_map # <--- CHANGE 1: Use RAW map for detection
        
        # Masks
        free_mask = (grid == 0)
        unknown_mask = (grid == -1)
        
        # Edge Detection
        up = np.roll(unknown_mask, -1, axis=0)
        down = np.roll(unknown_mask, 1, axis=0)
        left = np.roll(unknown_mask, -1, axis=1)
        right = np.roll(unknown_mask, 1, axis=1)
        
        has_unknown_neighbor = up | down | left | right
        
        # Frontier = Free Space (Safe) AND Touching Unknown
        is_frontier = free_mask & has_unknown_neighbor
        
        ys, xs = np.where(is_frontier)
        points = list(zip(xs, ys)) 

        if not points: return None

        # Clustering
        clusters = []
        visited = set()
        
        for point in points:
            if point in visited: continue
            
            current_cluster = []
            queue = [point]
            visited.add(point)
            
            while queue:
                curr = queue.pop(0)
                current_cluster.append(curr)
                cx, cy = curr
                
                neighbors = [
                    (cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1),
                    (cx+1, cy+1), (cx-1, cy-1), (cx-1, cy+1), (cx+1, cy-1)
                ]
                
                for n in neighbors:
                    if n in points and n not in visited:
                        if abs(n[0] - cx) <= 1 and abs(n[1] - cy) <= 1:
                            visited.add(n)
                            queue.append(n)
            
            if len(current_cluster) > 5:
                clusters.append(current_cluster)

        # Centroid Calculation (Snap to Valid Pixel)
        centroids = []
        res = self.map_data.resolution
        ox = self.map_data.origin.position.x
        oy = self.map_data.origin.position.y
        
        for cluster in clusters:
            # 1. Geometric Centroid
            sum_x = sum(p[0] for p in cluster)
            sum_y = sum(p[1] for p in cluster)
            avg_x = int(sum_x / len(cluster))
            avg_y = int(sum_y / len(cluster))
            
            # 2. Check Safety on INFLATED MAP
            # If the calculated centroid is inside a buffer zone, find nearest safe point
            safe_point = self.find_nearest_safe_point(avg_x, avg_y, self.inflated_map)
            
            if safe_point:
                cx, cy = safe_point
                # Convert to World
                wx = (cx * res) + ox + (res/2)
                wy = (cy * res) + oy + (res/2)
                centroids.append((wx, wy))
            
        return centroids

    def get_closest_frontier(self, centroids, rx, ry):
        if not centroids: return None
        best_dist = float('inf')
        best_point = None
        for (fx, fy) in centroids:
            
            # Blacklist check
            is_bad = False
            for (bx, by) in self.blacklist:
                if math.hypot(fx - bx, fy - by) < 0.5:
                    is_bad = True
                    break
            if is_bad: continue

            dist = math.hypot(fx - rx, fy - ry)
            if dist > 0.5 and dist < best_dist:
                best_dist = dist
                best_point = (fx, fy)
        return best_point

    def publish_frontiers(self, centroids, selected=None):
        if not centroids: return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "frontiers"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.2; marker.scale.y = 0.2
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        for (x, y) in centroids:
            if selected and x == selected[0] and y == selected[1]: continue
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.0
            marker.points.append(p)
        self.frontier_pub.publish(marker)

        if selected:
            sel = Marker()
            sel.header.frame_id = "map"
            sel.header.stamp = self.get_clock().now().to_msg()
            sel.ns = "frontiers"
            sel.id = 1
            sel.type = Marker.SPHERE
            sel.action = Marker.ADD
            sel.scale.x = 0.4; sel.scale.y = 0.4; sel.scale.z = 0.4
            sel.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            sel.pose.position.x = float(selected[0])
            sel.pose.position.y = float(selected[1])
            sel.pose.orientation.w = 1.0
            self.frontier_pub.publish(sel)

    def publish_plan(self, path_points):
        if not path_points: return
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in path_points:
            pose = PoseStamped()
            pose.pose.position.x = x; pose.pose.position.y = y
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    # Helper function to encapsulate map processing
    def process_map_and_find_frontier(self, rx, ry):
        self.inflated_map = self.generate_inflated_map()
        self.publish_debug_map(self.inflated_map)
        
        if self.inflated_map is not None:
            frontiers = self.get_frontiers()
            target = self.get_closest_frontier(frontiers, rx, ry)
            self.publish_frontiers(frontiers, selected=target)
            return target
        return None

    def control_loop(self):
        pose = self.get_robot_pose()
        if not pose: return
        rx, ry, yaw = pose

        # START State: Preserved exactly as requested
        if self.state == "START":
            self.move_ttbot(0.15, 0.5) 
            
            # But we must update the inflated map if the raw map changes
            if self.map_has_changed:
                self.map_has_changed = False
                target = self.process_map_and_find_frontier(rx, ry)

                if target:
                    self.current_goal = target
                    self.get_logger().info(f"Target Found: {target}")
                    self.state = "PLANNING"
                else:
                    self.get_logger().info("No valid frontiers found. Spinning...")

        elif self.state == "WAITING":
            self.move_ttbot(0.0, 0.0) 
            
            # Same update logic as START, but speed is 0
            if self.map_has_changed:
                self.map_has_changed = False
                target = self.process_map_and_find_frontier(rx, ry)
                
                if target:
                    self.current_goal = target
                    self.get_logger().info(f"Target Found: {target}")
                    self.state = "PLANNING"
                else:
                    self.get_logger().info("No valid frontiers found. Spinning...")

        elif self.state == "PLANNING":
            self.move_ttbot(0.0, 0.0)
            
            solver = GridAStar(self.map_data.resolution, 
                               self.map_data.origin.position.x, 
                               self.map_data.origin.position.y,
                               self.map_data.width, 
                               self.map_data.height)
            
            start = (rx, ry)
            # Use INFLATED MAP for planning
            path_points = solver.solve(self.inflated_map, start, self.current_goal)
            
            if path_points:
                self.path = path_points
                self.publish_plan(self.path)
                self.follow_idx = 0
                self.state = "MOVING"
                self.get_logger().info(f"Path planned: {len(self.path)} steps")
            else:
                self.get_logger().warn("A* Failed. Blacklisting...")
                self.blacklist.append(self.current_goal)
                self.state = "WAITING"

        elif self.state == "MOVING":
            self.publish_plan(self.path)
            if not self.path: return
            
            # --- DYNAMIC REPLANNING (CHANGE 1) ---
            if self.map_has_changed:
                self.map_has_changed = False
                
                # Check for frontiers based on new map
                new_target = self.process_map_and_find_frontier(rx, ry)
                
                if new_target:
                    # If the new goal is significantly different, replan
                    dist_to_old = math.hypot(new_target[0] - self.current_goal[0], 
                                             new_target[1] - self.current_goal[1])
                    if dist_to_old > 1.0:
                        self.get_logger().info("New better frontier found! Replanning.")
                        self.current_goal = new_target
                        self.state = "PLANNING"
                        return

            target = self.path[-1] 
            for i in range(self.follow_idx, len(self.path)):
                px, py = self.path[i]
                dist = math.hypot(px - rx, py - ry)
                if dist > self.lookahead_dist:
                    target = (px, py)
                    self.follow_idx = i
                    break
            
            tx, ty = target
            desired_yaw = math.atan2(ty - ry, tx - rx)
            err = (desired_yaw - yaw + math.pi) % (2.0*math.pi) - math.pi
            
            ang = max(-1.0, min(1.0, 1.5 * err))
            lin = 0.22
            if abs(err) > 1.0: lin = 0.0 
            
            self.move_ttbot(lin, ang)
            
            dist_to_goal = math.hypot(self.current_goal[0] - rx, self.current_goal[1] - ry)
            if dist_to_goal < 0.05:
                self.get_logger().info("Frontier Reached!")
                self.state = "WAITING"

    def move_ttbot(self, speed, heading):
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(heading)
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    task1 = Task1()
    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()