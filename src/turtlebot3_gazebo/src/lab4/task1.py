#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class Task1(Node):
    """
    Task 1: Autonomous Mapping
    Strategy: Drive Straight -> Align -> PD Wall Follower (with Doorway Logic)
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
        # FIND_WALL: Drive straight until impact
        # ALIGN_LEFT: Turn in place to face the open path
        # FOLLOW_WALL: The main mapping logic
        self.state = 'FIND_WALL'
        
        # --- Tuning Parameters ---
        self.target_dist = 0.50   
        self.kp = 1.0
        self.kd = 5.0             
        self.cruising_speed = 0.35
        
        self.safe_front_dist = 0.60    # Standard turning distance
        self.door_front_dist = 0.30    # "Brave" distance for doorways

        # PD Memory
        self.prev_error = 0.0

        self.get_logger().info('Task1 Node Started: 3-Stage Logic')

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
        front_cone = ranges[-10:] + ranges[:10]
        d_front = self.get_min_range(front_cone)

        right_cone = ranges[280:320]
        d_right = self.get_min_range(right_cone)

        # --- 2. STATE MACHINE ---

        # STATE 1: FIND WALL
        # Drive straight until we hit the first wall.
        if self.state == 'FIND_WALL':
            if d_front < self.safe_front_dist:
                self.state = 'ALIGN_LEFT' # Transition to Align Mode
                self.get_logger().info('Wall Found! Aligning Left...')
            else:
                twist.linear.x = self.cruising_speed
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                return 

        # STATE 2: ALIGN LEFT (The Fix)
        # We are staring at a wall. Turn Left until we see open space.
        # We ignore "Doorway Logic" here to prevent crashing.
        if self.state == 'ALIGN_LEFT':
            if d_front > self.safe_front_dist + 0.1: # Add buffer (0.7m) so we don't switch back too soon
                self.state = 'FOLLOW_WALL'
                self.get_logger().info('Aligned! Switching to Wall Follower.')
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.8 # Consistent turn speed
                self.cmd_vel_pub.publish(twist)
                return

        # STATE 3: FOLLOW WALL
        # Now we are safely parallel, we can use the advanced logic.
        if self.state == 'FOLLOW_WALL':
            
            # DOORWAY CHECK:
            # Only if we are in this state do we allow the threshold to drop.
            if d_right > 1.0:
                active_thresh = self.door_front_dist # 0.30m
            else:
                active_thresh = self.safe_front_dist # 0.60m

            # Priority 1: Emergency Front Blocked
            if d_front < active_thresh:
                twist.linear.x = 0.0
                twist.angular.z = 1.2  # Turn Left fast
                self.prev_error = 0.0  
                self.get_logger().info('Corner! Turning Left.', throttle_duration_sec=1)

            # Priority 2: PD Control
            else:
                twist.linear.x = self.cruising_speed
                
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