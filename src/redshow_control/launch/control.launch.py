#!/usr/bin/env python3
"""
라즈베리파이용 Control Launch 파일
BNO085 노드와 Control 노드를 함께 실행합니다.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # BNO085 IMU 센서 노드
        Node(
            package='redshow_control',
            executable='bno085_node',
            name='bno085_node',
            output='screen',
            parameters=[],
        ),
        
        # Control 노드 (OpenCR Firmware와 통신)
        Node(
            package='redshow_control',
            executable='redshow_control_node',
            name='berkeley_control_node',
            output='screen',
            parameters=[
                {'policy_name': 'model_4800.pt'},  # 기본 모델 파일명
            ],
        ),
    ])

