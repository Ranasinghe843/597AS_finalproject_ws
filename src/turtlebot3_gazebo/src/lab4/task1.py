#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Float32, ColorRGBA
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import numpy as np
import math
import heapq

# =========================================
# HELPER CLASS: Grid-Based A* with Costs
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
        
        motions = [
            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]

        # --- SAFETY PARAMETERS ---
        # 4 cells * 0.05m = 20cm. Robot radius is ~15cm.
        # This keeps the CENTER of the robot 20cm away from walls.
        wall_padding = 4 
        
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
                    
                    # 1. CRITICAL CHECK: Center pixel must be free
                    center_val = grid[ny, nx]
                    if center_val > 50 or center_val == -1: 
                        continue

                    # 2. SAFETY CHECK
                    is_safe = True
                    proximity_cost = 0.0
                    
                    # Scan the local box
                    for r in range(ny - wall_padding, ny + wall_padding + 1):
                        for c in range(nx - wall_padding, nx + wall_padding + 1):
                            if 0 <= r < self.h and 0 <= c < self.w:
                                val = grid[r, c]
                                
                                # A. WALLS: Strict avoidance
                                if val > 50:
                                    is_safe = False
                                    break
                                
                                # B. UNKNOWN: Loose avoidance
                                # Only fail if we are RIGHT next to unknown (radius 1)
                                # This allows passing through doorways where unknown is close
                                if val == -1:
                                    dist = max(abs(r-ny), abs(c-nx))
                                    if dist <= 1: 
                                        # Only unsafe if VERY close to unknown
                                        # (prevents driving off map)
                                        # But we allow it near goal (frontier)
                                        if math.hypot(nx-gx, ny-gy) > 2: # Ignore near goal
                                            is_safe = False
                                            break
                                            
                                # C. COST INFLATION (The "Middle of Hallway" Logic)
                                # If we are safe but close to a wall, add a penalty cost.
                                # This makes the planner prefer the center of open spaces.
                                if val > 50: # If there is a wall nearby (within padding)
                                    dist = math.hypot(r-ny, c-nx)
                                    if dist < wall_padding:
                                        proximity_cost += (wall_padding - dist) * 2.0 # Penalty

                        if not is_safe: break
                    
                    if not is_safe: continue

                    # 3. COST CALCULATION
                    # Base Cost + Proximity Penalty
                    new_g = g_score[current] + cost + proximity_cost
                    
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
        
        self.state = "WAITING" 
        self.path = []
        self.current_map = None
        self.map_data = None 
        self.current_goal = None
        self.follow_idx = 0
        self.blacklist = []
        
        self.lookahead_dist = 0.4
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/vis/global_plan', 10)
        self.frontier_pub = self.create_publisher(Marker, '/vis/frontiers', 10)
        
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
        self.current_map = raw.reshape((msg.info.height, msg.info.width))

    def get_frontiers(self):
        """Finds centroids of frontiers, filtering those too close to walls."""
        if self.current_map is None: return None

        h, w = self.current_map.shape
        grid = self.current_map
        
        free_mask = (grid == 0)
        unknown_mask = (grid == -1)
        wall_mask = (grid > 50) # Strict walls
        
        # 1. Expand Walls (Dilation)
        # We want to ignore any frontier that is within 3 pixels of a wall
        # to prevent "Wall Hugging" goals.
        d_range = 3
        expanded_wall = wall_mask.copy()
        for r in range(-d_range, d_range+1):
            for c in range(-d_range, d_range+1):
                shift = np.roll(wall_mask, shift=(r, c), axis=(0, 1))
                expanded_wall |= shift

        # 2. Find Edges
        up = np.roll(unknown_mask, -1, axis=0)
        down = np.roll(unknown_mask, 1, axis=0)
        left = np.roll(unknown_mask, -1, axis=1)
        right = np.roll(unknown_mask, 1, axis=1)
        
        has_unknown_neighbor = up | down | left | right
        
        # Frontier = Free AND Touching Unknown AND NOT near Wall
        is_frontier = free_mask & has_unknown_neighbor & (~expanded_wall)
        
        ys, xs = np.where(is_frontier)
        points = list(zip(xs, ys)) 

        if not points: return None

        # 3. Cluster
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

        # 4. Centroids
        centroids = []
        res = self.map_data.resolution
        ox = self.map_data.origin.position.x
        oy = self.map_data.origin.position.y
        
        for cluster in clusters:
            sum_x = sum(p[0] for p in cluster)
            sum_y = sum(p[1] for p in cluster)
            avg_x = sum_x / len(cluster)
            avg_y = sum_y / len(cluster)
            
            wx = (avg_x * res) + ox + (res/2)
            wy = (avg_y * res) + oy + (res/2)
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
        """
        Publishes:
        1. Red dots for all candidate frontiers.
        2. A large Green sphere for the SELECTED frontier.
        """
        if not centroids: return
        
        timestamp = self.get_clock().now().to_msg()

        # --- 1. CANDIDATES (RED DOTS) ---
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = timestamp
        marker.ns = "frontiers"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # Red
        
        for (x, y) in centroids:
            # Optional: Don't draw a red dot under the green sphere
            if selected and x == selected[0] and y == selected[1]:
                continue
                
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)
            
        self.frontier_pub.publish(marker)

        # --- 2. SELECTED TARGET (GREEN SPHERE) ---
        if selected:
            marker_sel = Marker()
            marker_sel.header.frame_id = "map"
            marker_sel.header.stamp = timestamp
            marker_sel.ns = "frontiers"
            marker_sel.id = 1 # Different ID to separate from the list
            marker_sel.type = Marker.SPHERE
            marker_sel.action = Marker.ADD
            
            # Make it bigger so it stands out
            marker_sel.scale.x = 0.4 
            marker_sel.scale.y = 0.4
            marker_sel.scale.z = 0.4
            marker_sel.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # Green
            
            marker_sel.pose.position.x = float(selected[0])
            marker_sel.pose.position.y = float(selected[1])
            marker_sel.pose.position.z = 0.0
            marker_sel.pose.orientation.w = 1.0
            
            self.frontier_pub.publish(marker_sel)

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

    def control_loop(self):
        pose = self.get_robot_pose()
        if not pose: return
        rx, ry, yaw = pose

        if self.state == "WAITING":
            self.move_ttbot(0.15, 0.5) # Spin to improve map
            
            if self.current_map is not None:
                self.get_logger().info("Detecting Frontiers...")
                frontiers = self.get_frontiers()
                
                # 1. First, pick the target
                target = self.get_closest_frontier(frontiers, rx, ry)
                
                # 2. Then visualize (Pass 'target' so it turns Green)
                if frontiers:
                    self.publish_frontiers(frontiers, selected=target)

                # 3. Then Process
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
            path_points = solver.solve(self.current_map, start, self.current_goal)
            
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
            
            # Stop if reached goal OR if we run out of path
            dist_to_goal = math.hypot(self.current_goal[0] - rx, self.current_goal[1] - ry)
            if dist_to_goal < 0.3:
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