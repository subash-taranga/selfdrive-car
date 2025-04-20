#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy
import torch
from pathlib import Path
from sensor_msgs.msg import Image

class ObjectDetection(Node):

    def __init__(self):
        super().__init__('object_detection')
        topic = "/camara_img"
        self.get_logger().info('object_detection is listening to Topic -> ' + topic)
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(Image, topic, self.camara_frame_callback, qos_profile)
        
        # Create a publisher for the Twist message
        self.twist_pub = self.create_publisher(String, '/vel_cmd', 10)

        self.object_detection_string = String()
        
        self.result = ""
        
        # Create a timer to publish the Twist message
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        BASE = Path(__file__).parent.resolve()
        WEIGHTS = BASE / "best.pt"
        IMG     = BASE / "santa.JPG"  

        if not WEIGHTS.exists():
            raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
        if not IMG.exists():
            raise FileNotFoundError(f"Image not found:   {IMG}")

        self.model = torch.hub.load(
            "ultralytics/yolov5",   
            "custom",               
            path=str(WEIGHTS),      
            force_reload=True       
        )
        

    def camara_frame_callback(self, imageFrame: Image):
        self.object_detection_string.data = self.model(str(imageFrame), augment=True)

    # demo program to move the steering
    def timer_callback(self):
        self.twist_pub.publish(self.object_detection_string)

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