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
                    
                    # RELAXED CHECK: Allow planning through noise < 50
                    if grid[ny, nx] > 50: 
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
        self.raw_map = None      
        self.inflated_map = None 
        self.map_data = None 
        self.map_has_changed = False 
        
        self.current_goal = None
        self.follow_idx = 0
        self.blacklist = []
        self.visited_frontiers = [] 
        
        # Parameters
        self.lookahead_dist = 0.4
        self.inflation_radius = 4
        
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
        raw = np.array(msg.data, dtype=np.int8)
        self.raw_map = raw.reshape((msg.info.height, msg.info.width))
        self.map_has_changed = True 

    def generate_inflated_map(self):
        if self.raw_map is None: return None
        inflated = self.raw_map.copy()
        
        wall_indices = np.where(self.raw_map > 50)
        if len(wall_indices[0]) == 0: return inflated
        
        wall_mask = (self.raw_map > 50)
        r = self.inflation_radius
        expanded_walls = wall_mask.copy()
        for y_shift in range(-r, r + 1):
            for x_shift in range(-r, r + 1):
                if x_shift == 0 and y_shift == 0: continue
                shifted = np.roll(wall_mask, shift=(y_shift, x_shift), axis=(0, 1))
                expanded_walls |= shifted
        inflated[expanded_walls] = 100
        return inflated

    def publish_debug_map(self, grid_2d):
        if grid_2d is None: return
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.map_data 
        flat = grid_2d.flatten().astype(np.int8)
        msg.data = flat.tolist()
        self.debug_map_pub.publish(msg)

    # --- HELPER: Check proximity to walls ---
    def get_proximity(self, rx, ry):
        """Returns max inflated map value around the robot."""
        if self.inflated_map is None: return 0
        res = self.map_data.resolution
        ox = self.map_data.origin.position.x
        oy = self.map_data.origin.position.y
        gx = int((rx - ox) / res)
        gy = int((ry - oy) / res)
        
        h, w = self.inflated_map.shape
        max_val = 0
        # Check 3x3 window around robot
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = gy + dy, gx + dx
                if 0 <= nx < w and 0 <= ny < h:
                    val = self.inflated_map[ny, nx]
                    if val > max_val: max_val = val
        return max_val

    def find_nearest_safe_point(self, cx, cy, grid, max_radius=30):
        h, w = grid.shape
        if grid[cy, cx] < 50 and grid[cy, cx] >= 0:
            return cx, cy

        best_x, best_y = None, None
        min_cost = float('inf')

        x, y = 0, 0
        dx, dy = -1, 0 
        for _ in range(max_radius * max_radius * 4):
            if (-max_radius <= x <= max_radius) and (-max_radius <= y <= max_radius):
                nx, ny = cx + x, cy + y
                if 0 <= nx < w and 0 <= ny < h:
                    val = grid[ny, nx]
                    if val != -1:
                        if val == 0:
                            return nx, ny
                        if val < min_cost:
                            min_cost = val
                            best_x, best_y = nx, ny

            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
                dx, dy = -dy, dx 
            x, y = x + dx, y + dy
        
        if best_x is not None and min_cost < 100:
            return best_x, best_y
        return None 

    def get_frontiers(self):
        if self.raw_map is None or self.inflated_map is None: return None
        grid = self.raw_map
        
        # Relaxed detection
        free_mask = (grid >= 0) & (grid < 25)
        unknown_mask = (grid == -1)
        
        up = np.roll(unknown_mask, -1, axis=0)
        down = np.roll(unknown_mask, 1, axis=0)
        left = np.roll(unknown_mask, -1, axis=1)
        right = np.roll(unknown_mask, 1, axis=1)
        
        has_unknown_neighbor = up | down | left | right
        is_frontier = free_mask & has_unknown_neighbor
        
        ys, xs = np.where(is_frontier)
        points = list(zip(xs, ys)) 

        if not points: return None

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
            
            if len(current_cluster) > 2:
                clusters.append(current_cluster)

        centroids = []
        res = self.map_data.resolution
        ox = self.map_data.origin.position.x
        oy = self.map_data.origin.position.y
        
        for cluster in clusters:
            sum_x = sum(p[0] for p in cluster)
            sum_y = sum(p[1] for p in cluster)
            avg_x = int(sum_x / len(cluster))
            avg_y = int(sum_y / len(cluster))
            
            safe_point = self.find_nearest_safe_point(avg_x, avg_y, self.inflated_map)
            
            if safe_point:
                cx, cy = safe_point
                wx = (cx * res) + ox + (res/2)
                wy = (cy * res) + oy + (res/2)
                centroids.append((wx, wy))
            
        return centroids

    def get_closest_frontier(self, centroids, rx, ry):
        if not centroids: return None
        best_dist = float('inf')
        best_point = None
        for (fx, fy) in centroids:
            
            is_bad = False
            for (bx, by) in self.blacklist:
                if math.hypot(fx - bx, fy - by) < 0.3:
                    is_bad = True
                    break
            if is_bad: continue

            is_visited = False
            for (vx, vy) in self.visited_frontiers:
                if math.hypot(fx - vx, fy - vy) < 0.3: 
                    is_visited = True
                    break
            if is_visited: continue

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

        # START State
        if self.state == "START":
            self.move_ttbot(0.15, 0.5) 
            
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
            
            # Dynamic Replanning
            dist_to_goal = math.hypot(self.current_goal[0] - rx, self.current_goal[1] - ry)
            if self.map_has_changed:
                self.map_has_changed = False
                new_target = self.process_map_and_find_frontier(rx, ry)
                if new_target:
                    dist_to_old = math.hypot(new_target[0] - self.current_goal[0], 
                                             new_target[1] - self.current_goal[1])
                    if dist_to_old > 1.0 and dist_to_goal > 0.5:
                        self.get_logger().info("New better frontier found! Replanning.")
                        self.current_goal = new_target
                        self.state = "PLANNING"
                        return

            # --- IMPROVED WALL-AWARE CONTROLLER ---
            prox = self.get_proximity(rx, ry)
            
            # Default Params
            v_target = 0.22
            l_dist = self.lookahead_dist # 0.4
            
            # 1. Slow Down Near Walls
            # If prox > 0, we are touching buffer/wall
            if prox > 0: 
                v_target = 0.1
                l_dist = 0.25 # Tighten lookahead to avoid corner cutting
            
            # 2. Lookahead Logic
            target = self.path[-1] 
            for i in range(self.follow_idx, len(self.path)):
                px, py = self.path[i]
                dist = math.hypot(px - rx, py - ry)
                if dist > l_dist:
                    target = (px, py)
                    self.follow_idx = i
                    break
            
            tx, ty = target
            desired_yaw = math.atan2(ty - ry, tx - rx)
            err = (desired_yaw - yaw + math.pi) % (2.0*math.pi) - math.pi
            
            # 3. Aggressive Turning
            # Increased gain to 2.0 to correct heading faster
            ang = max(-1.0, min(1.0, 2.0 * err))
            
            # 4. Strict Stop-and-Turn
            # If error is > ~30 deg, stop moving to rotate
            if abs(err) > 0.5: 
                v_target = 0.0 
            
            self.move_ttbot(v_target, ang)
            
            if dist_to_goal < 0.1:
                self.get_logger().info("Frontier Reached!")
                self.visited_frontiers.append(self.current_goal)
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