#!/usr/bin/env python3
"""
테스트용 Adaptation Module Publisher 노드
A-RMA Adaptation Module 테스트용으로 8차원 extrinsics_obs 데이터를 발행합니다.
- 8차원 데이터를 redshow/extrinsics_obs 토픽에 발행
- 50Hz로 발행

사용법:
ros2 run redshow_control test_adaptation_module_publisher
"""

import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TestAdaptationModulePublisher(Node):
    def __init__(self):
        super().__init__('test_adaptation_module_publisher')
        
        # 테스트용 extrinsics_obs publisher
        self.extrinsics_pub = self.create_publisher(
            Float64MultiArray, 
            'redshow/extrinsics_obs', 
            10
        )
        
        # 50Hz로 발행
        self.timer = self.create_timer(1.0 / 50.0, self.publish_extrinsics)
        
        self.get_logger().info("✓ Test Adaptation Module Publisher started (50Hz)")
        self.get_logger().info("  Publishing to: redshow/extrinsics_obs")
        self.get_logger().info("  Data: 8-dimensional extrinsics_obs")
    
    def publish_extrinsics(self):
        """테스트용 extrinsics_obs 데이터 발행 (8차원)"""
        # 테스트용 extrinsics_obs 데이터 생성 (8차원)
        # 간단한 테스트 데이터 (예: 사인파 패턴)
        import math
        current_time = time.time()
        
        test_extrinsics = [
            math.sin(current_time * 0.5) * 0.5,  # 0
            math.cos(current_time * 0.5) * 0.5,  # 1
            math.sin(current_time * 0.7) * 0.3,  # 2
            math.cos(current_time * 0.7) * 0.3,  # 3
            math.sin(current_time * 0.3) * 0.2,  # 4
            math.cos(current_time * 0.3) * 0.2,  # 5
            math.sin(current_time * 0.9) * 0.1,  # 6
            math.cos(current_time * 0.9) * 0.1,  # 7
        ]
        
        msg = Float64MultiArray()
        msg.data = test_extrinsics
        self.extrinsics_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestAdaptationModulePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

