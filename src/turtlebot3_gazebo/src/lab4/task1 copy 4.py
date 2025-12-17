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
                
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    # Allow planning through small noise < 50
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
        
        # State Machine: WAITING -> MOVING -> RECOVERY -> DONE
        self.state = "WAITING" 
        
        self.path = []
        self.raw_map = None      
        self.inflated_map = None 
        self.map_data = None 
        self.map_has_changed = False 
        
        self.current_goal = None
        self.follow_idx = 0
        self.blacklist = []
        self.visited_frontiers = []
        self.was_recovery = 0
        
        # Timers for State Logic
        self.waiting_start_time = 0.0
        self.recovery_start_time = 0.0
        
        # Parameters
        self.lookahead_dist = 0.4
        self.inflation_radius = 6
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
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

    def get_proximity(self, rx, ry):
        if self.inflated_map is None: return 0
        res = self.map_data.resolution
        ox = self.map_data.origin.position.x
        oy = self.map_data.origin.position.y
        gx = int((rx - ox) / res)
        gy = int((ry - oy) / res)
        h, w = self.inflated_map.shape
        max_val = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = gy + dy, gx + dx
                if 0 <= nx < w and 0 <= ny < h:
                    val = self.inflated_map[ny, nx]
                    if val > max_val: max_val = val
        return max_val

    def find_nearest_safe_point(self, cx, cy, grid, max_radius=35):
        h, w = grid.shape
        # Accept < 50 as safe enough
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
        """
        Returns: 
        1. List of centroids (tuples)
        2. Boolean is_complete (True if NO raw frontier cells exist)
        """
        if self.raw_map is None or self.inflated_map is None: return [], False
        grid = self.raw_map
        
        # Relaxed detection < 25
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

        # --- DONE CONDITION CHECK ---
        # If there are 0 raw frontier pixels, the map is fully closed/explored.
        if len(points) == 0:
            return [], True

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
            
            # --- SINGLE CELL LOGIC ---
            # Changed from > 2 to > 0 to capture single unknown cells
            if len(current_cluster) > 0:
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
            
        return centroids, False

    def get_closest_frontier(self, centroids, rx, ry):
        if not centroids: return None
        best_dist = float('inf')
        best_point = None
        for (fx, fy) in centroids:
            
            # Blacklist
            is_bad = False
            for (bx, by) in self.blacklist:
                if math.hypot(fx - bx, fy - by) < 0.3:
                    is_bad = True
                    break
            if is_bad: continue

            # Visited
            is_visited = False
            for (vx, vy) in self.visited_frontiers:
                if math.hypot(fx - vx, fy - vy) < 0.1: 
                    is_visited = True
                    break
            if is_visited: continue

            dist = math.hypot(fx - rx, fy - ry)
            if dist > 0.1 and dist < best_dist:
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
            sel.type = Marker.POINTS
            sel.action = Marker.ADD
            sel.scale.x = 0.2; sel.scale.y = 0.2
            sel.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            p = Point(); p.x = float(selected[0]); p.y = float(selected[1]); p.z = 0.0
            sel.points.append(p)
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

    def move_ttbot(self, speed, heading):
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(heading)
        self.cmd_vel_pub.publish(cmd)

    def control_loop(self):
        pose = self.get_robot_pose()
        if not pose: return
        rx, ry, yaw = pose
        now = time.time()

        # =========================================================
        # STATE 1: WAITING (Calculate & Plan)
        # =========================================================
        if self.state == "WAITING":
            self.move_ttbot(0.0, 0.0) # Stop
            
            # If map changes, update inflations
            if self.map_has_changed:
                self.inflated_map = self.generate_inflated_map()
                self.publish_debug_map(self.inflated_map)
                self.map_has_changed = False
            
            # 1. Get Frontiers & Check DONE
            frontiers, map_is_complete = self.get_frontiers()
            
            if map_is_complete:
                self.get_logger().info("NO FRONTIERS LEFT. MAPPING COMPLETE.")
                self.state = "DONE"
                return

            # 2. Pick Best Goal
            target = self.get_closest_frontier(frontiers, rx, ry)
            self.publish_frontiers(frontiers, selected=target)

            if target:
                # 3. Plan Path
                solver = GridAStar(self.map_data.resolution, 
                                   self.map_data.origin.position.x, 
                                   self.map_data.origin.position.y,
                                   self.map_data.width, 
                                   self.map_data.height)
                
                path_points = solver.solve(self.inflated_map, (rx, ry), target)
                
                if path_points:
                    self.path = path_points
                    self.current_goal = target
                    self.follow_idx = 0
                    self.state = "MOVING"
                    self.get_logger().info("Path found! Switching to MOVING.")
                else:
                    self.get_logger().warn("Path planning failed. Blacklisting frontier.")
                    self.blacklist.append(target)
                    # Stay in WAITING to pick next best
            else:
                self.get_logger().info("No reachable/valid frontiers found.")
            
            # 4. Timeout -> Recovery
            # If we've been waiting too long with no success
            if self.waiting_start_time == 0:
                self.waiting_start_time = now
            
            if (now - self.waiting_start_time) > 5.0: # 5 seconds timeout
                self.get_logger().warn("WAITING timeout. Switching to RECOVERY.")
                self.state = "RECOVERY"
                self.recovery_start_time = now
                self.waiting_start_time = 0 # Reset

        # =========================================================
        # STATE 2: MOVING (Follow Path & Replan)
        # =========================================================
        elif self.state == "MOVING":
            self.publish_plan(self.path)
            self.waiting_start_time = 0 # Reset wait timer
            
            # 1. Dynamic Update
            if self.map_has_changed:
                self.map_has_changed = False
                self.inflated_map = self.generate_inflated_map()
                self.publish_debug_map(self.inflated_map)
                
                # Check for better frontier
                frontiers, _ = self.get_frontiers()
                new_target = self.get_closest_frontier(frontiers, rx, ry)
                
                if new_target:
                    # dist_current = math.hypot(self.current_goal[0] - rx, self.current_goal[1] - ry)
                    # dist_new = math.hypot(new_target[0] - rx, new_target[1] - ry)
                    
                    # # If new target is significantly closer (by 1.0m)
                    # if dist_new < (dist_current - 1.0):
                    #     self.get_logger().info("Significantly closer frontier found. Switching to WAITING to replan.")
                    self.state = "WAITING"
                    return

            # 2. Controller
            prox = self.get_proximity(rx, ry)
            v_target = 0.22
            l_dist = self.lookahead_dist 
            
            if prox > 0: 
                # v_target = 0.1
                l_dist = 0.25 
            
            # Find Lookahead
            target_point = self.path[-1] 
            for i in range(self.follow_idx, len(self.path)):
                px, py = self.path[i]
                dist = math.hypot(px - rx, py - ry)
                if dist > l_dist:
                    target_point = (px, py)
                    self.follow_idx = i
                    break
            
            tx, ty = target_point
            desired_yaw = math.atan2(ty - ry, tx - rx)
            err = (desired_yaw - yaw + math.pi) % (2.0*math.pi) - math.pi
            
            ang = max(-1.0, min(1.0, 2.0 * err))
            if abs(err) > 0.5: v_target = 0.0 
            
            self.move_ttbot(v_target, ang)
            
            # 3. Check Reached
            dist_to_goal = math.hypot(self.current_goal[0] - rx, self.current_goal[1] - ry)
            if dist_to_goal < 0.05:
                self.get_logger().info("Goal Reached!")
                self.visited_frontiers.append(self.current_goal)
                self.state = "WAITING"

        # =========================================================
        # STATE 3: RECOVERY
        # =========================================================
        elif self.state == "RECOVERY":
            # FIX: Clear memory immediately so we see old frontiers again
            if len(self.blacklist) > 0 or len(self.visited_frontiers) > 0:
                self.get_logger().info("Clearing memory for recovery.")
                self.blacklist = []
                self.visited_frontiers = []

            if self.map_has_changed:
                self.inflated_map = self.generate_inflated_map()
                self.map_has_changed = False

            frontiers, _ = self.get_frontiers()
            target = self.get_closest_frontier(frontiers, rx, ry)
            
            # 
            
            path_found = False
            
            if target:
                target_angle = math.atan2(target[1] - ry, target[0] - rx)
                for _ in range(10):
                    angle_noise = np.random.uniform(-math.pi/3, math.pi/3) 
                    check_angle = target_angle + angle_noise
                    check_dist = np.random.uniform(0.5, 1.2) 
                    wx = rx + check_dist * math.cos(check_angle)
                    wy = ry + check_dist * math.sin(check_angle)
                    gx, gy = self.world_to_grid_safe(wx, wy)
                    
                    if 0 <= gx < self.inflated_map.shape[1] and 0 <= gy < self.inflated_map.shape[0]:
                        if self.inflated_map[gy, gx] < 100:
                            solver = GridAStar(self.map_data.resolution, self.map_data.origin.position.x, self.map_data.origin.position.y, self.map_data.width, self.map_data.height)
                            path = solver.solve(self.inflated_map, (rx, ry), (wx, wy))
                            if path:
                                self.path = path
                                self.current_goal = (wx, wy)
                                self.follow_idx = 0
                                self.state = "MOVING"
                                path_found = True
                                self.get_logger().info("Recovery Path Found!")
                                break
            
            if not path_found:
                self.move_ttbot(0.0, 0.6)
                if (now - self.recovery_start_time) > 3.0:
                    self.state = "WAITING"
                    self.waiting_start_time = now

        # =========================================================
        # STATE 4: DONE
        # =========================================================
        elif self.state == "DONE":
            self.move_ttbot(0.0, 0.0)

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