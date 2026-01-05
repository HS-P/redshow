#!/usr/bin/env python3
"""
테스트용 피드백 Publisher 노드
PT 파일 테스트용으로 가짜 센서 데이터를 발행합니다.
- 쿼터니언: [1, 0, 0, 0]
- 다리와 다 영도: 0으로 설정
- 50Hz로 redshow/test_feedback 토픽에 발행
"""

import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TestFeedbackPublisher(Node):
    def __init__(self):
        super().__init__('test_feedback_publisher')
        
        # 테스트용 피드백 publisher (GUI와 동일한 토픽 사용)
        self.test_feedback_pub = self.create_publisher(
            Float64MultiArray, 
            'redshow/feedback', 
            10
        )
        
        # 50Hz로 발행
        self.timer = self.create_timer(1.0 / 50.0, self.publish_test_feedback)
        
        self.get_logger().info("✓ Test Feedback Publisher started (50Hz)")
        self.get_logger().info("  Publishing to: redshow/feedback")
        self.get_logger().info("  Use this node instead of redshow_control_node for GUI testing")
    
    def publish_test_feedback(self):
        """테스트용 피드백 데이터 발행
        env.yaml의 observations.policy 순서에 맞춰서:
        - leg_position: 4개 (0-3)
        - wheel_velocity: 2개 (4-5)
        - base_ang_vel: 3개 (6-8)
        - velocity_commands: 4개 (9-12)
        - base_quat: 4개 (13-16) - quaternion (w, x, y, z)
        - actions: 6개 (17-22) - previous actions
        """
        # 테스트용 피드백 데이터 생성 (PT 파일 테스트용)
        # 모든 값을 0으로 설정하되, quat[0] (w)만 1.0으로 설정하여 유효한 데이터로 인식되도록
        test_feedback = [
            0.0, 0.0, 0.0, 0.0,  # leg_position (4개)
            0.0, 0.0,              # wheel_velocity (2개)
            0.0, 0.0, 0.0,          # base_ang_vel (3개)
            0.0, 0.0, 0.0, 0.0,     # velocity_commands (4개)
            1.0, 0.0, 0.0, 0.0,     # base_quat (4개) - quaternion (w, x, y, z), w=1.0으로 설정하여 유효한 데이터로 인식
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # actions (6개) - previous actions
        ]
        
        msg = Float64MultiArray()
        msg.data = test_feedback
        self.test_feedback_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestFeedbackPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

