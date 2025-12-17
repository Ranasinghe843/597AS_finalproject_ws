#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class Task1(Node):
    """
    Task 1: Autonomous Mapping
    Strategy: Drive Straight -> Align -> Tight-Cornering Wall Follower
    """
    def __init__(self):
        super().__init__('task1_node')

        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_policy
        )
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_cb)
        self.scan_data = None
        
        # --- State Management ---
        self.state = 'FIND_WALL'
        
        # --- Tuning Parameters ---
        self.target_dist = 0.50   
        self.kp = 1.0
        self.kd = 5.0             
        
        self.cruising_speed = 0.35     # Normal speed
        self.cornering_speed = 0.10    # Slow speed for tight turns (The Fix)
        
        self.safe_front_dist = 0.60    
        self.door_front_dist = 0.30    # Distance allowed when cornering

        self.prev_error = 0.0

        self.get_logger().info('Task1 Node Started: Tight Cornering Logic')

    def scan_callback(self, msg):
        self.scan_data = msg

    def get_min_range(self, ranges):
        valid_ranges = [r for r in ranges if not (r == float('inf') or r == 0.0)]
        if not valid_ranges:
            return 10.0 
        return min(valid_ranges)

    def timer_cb(self):
        if self.scan_data is None:
            return

        twist = Twist()
        ranges = self.scan_data.ranges
        
        # --- 1. SENSOR PROCESSING ---
        # Front Cone: +/- 10 degrees
        front_cone = ranges[-15:] + ranges[:15]
        d_front = self.get_min_range(front_cone)

        # Right Cone: 280-320 (Look-Ahead)
        right_cone = ranges[280:320]
        d_right = self.get_min_range(right_cone)
        
        # Front-Right Corner Check (Indices 320-350)
        # This checks for the specific "corner right after doorway"
        corner_cone = ranges[320:350]
        d_corner = self.get_min_range(corner_cone)

        # --- 2. STATE MACHINE ---

        # STATE 1: FIND WALL
        if self.state == 'FIND_WALL':
            if d_front < self.safe_front_dist:
                self.state = 'ALIGN_LEFT'
                self.get_logger().info('Wall Found! Aligning Left...')
            else:
                twist.linear.x = self.cruising_speed
                self.cmd_vel_pub.publish(twist)
                return 

        # STATE 2: ALIGN LEFT
        if self.state == 'ALIGN_LEFT':
            if d_front > self.safe_front_dist + 0.1:
                self.state = 'FOLLOW_WALL'
                self.get_logger().info('Aligned! Switching to Wall Follower.')
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.8
                self.cmd_vel_pub.publish(twist)
                return

        # STATE 3: FOLLOW WALL
        if self.state == 'FOLLOW_WALL':
            
            # --- DOORWAY / CORNER LOGIC ---
            # If Right is open, we are at a door.
            if d_right > 1.0:
                active_thresh = self.door_front_dist
                # THE FIX: Slow down to wrap tightly around the corner
                current_speed = self.cornering_speed
            else:
                active_thresh = self.safe_front_dist
                current_speed = self.cruising_speed

            # --- CONTROL LOGIC ---

            # Priority 0: IMMEDIATE CORNER CRASH (The Z-Turn Protection)
            # If we are turning right but a corner is RIGHT in our face (d_corner), 
            # we must panic turn LEFT briefly to clear it.
            if d_corner < 0.25:
                 twist.linear.x = 0.0
                 twist.angular.z = 1.0 # Quick bump left
                 self.get_logger().info('Tight Corner! Bumping Left.', throttle_duration_sec=1)

            # Priority 1: Front Blocked
            elif d_front < active_thresh:
                twist.linear.x = 0.0
                twist.angular.z = 1.2  # Turn Left fast
                self.prev_error = 0.0  
                self.get_logger().info('Front Blocked. Turning Left.', throttle_duration_sec=1)

            # Priority 2: PD Control
            else:
                twist.linear.x = current_speed
                
                # Calculate Error
                error = d_right - self.target_dist
                
                # Cap error for doorways
                error = max(min(error, 0.8), -0.5)

                derivative = error - self.prev_error
                
                # PD Output
                turn_cmd = -(self.kp * error + self.kd * derivative)
                
                twist.angular.z = max(min(turn_cmd, 1.2), -1.2)
                self.prev_error = error

            self.cmd_vel_pub.publish(twist)

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