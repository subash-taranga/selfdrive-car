#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy


class ObjectDetection(Node):

    def __init__(self):
        super().__init__('camara_publisher')

        # Create a publisher for the Twist message
        self.twist_pub = self.create_publisher(String, '/camara_img', 10)

        self.object_detection_string = String()
        
        # Create a timer to publish the Twist message
        timer_period = 4.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)


    def chatter_callback(self, msg: String):
        self.get_logger().info(str(msg))

    # demo program to move the steering
    def timer_callback(self):
        self.object_detection_string.data = "test"
        self.twist_pub.publish(self.object_detection_string)
        self.get_logger().info(f'Published...')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetection()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()