#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class Task1(Node):
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
        
        self.state = 'FIND_WALL'
        
        self.target_dist = 0.50   
        self.kp = 1.0
        self.kd = 5.0             
        
        self.cruising_speed = 0.35
        self.cornering_speed = 0.05
        
        self.safe_front_dist = 0.60
        self.door_front_dist = 0.30

        self.prev_error = 0.0

        self.get_logger().info('Task1 Node Started')

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
        
        front_cone = ranges[-15:] + ranges[:15]
        d_front = self.get_min_range(front_cone)

        right_cone = ranges[280:320]
        d_right = self.get_min_range(right_cone)
        
        corner_cone = ranges[340:300]
        d_corner = self.get_min_range(corner_cone)

        if self.state == 'FIND_WALL':
            if d_front < self.safe_front_dist:
                self.state = 'ALIGN_LEFT'
                self.get_logger().info('Wall Found!')
            else:
                twist.linear.x = self.cruising_speed
                self.cmd_vel_pub.publish(twist)
                return 

        if self.state == 'ALIGN_LEFT':
            if d_front > self.safe_front_dist + 0.1:
                self.state = 'FOLLOW_WALL'
                self.get_logger().info('Aligned!')
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.8
                self.cmd_vel_pub.publish(twist)
                return

        if self.state == 'FOLLOW_WALL':
            
            if d_right > 1.0:
                active_thresh = self.door_front_dist
                current_speed = self.cornering_speed
            else:
                active_thresh = self.safe_front_dist
                current_speed = self.cruising_speed

            if d_corner > 1.0 and d_front < active_thresh:
                 twist.linear.x = 0.0
                 twist.angular.z = -1.0
                 self.get_logger().info('Tight Corner', throttle_duration_sec=1)

            elif d_front < active_thresh:
            # if d_front < active_thresh:
                twist.linear.x = 0.0
                twist.angular.z = 1.2
                self.prev_error = 0.0  
                self.get_logger().info('Front Blocked; Turning Left', throttle_duration_sec=1)

            else:
                twist.linear.x = current_speed
                
                error = d_right - self.target_dist
                
                error = max(min(error, 0.8), -0.5)

                derivative = error - self.prev_error
                
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