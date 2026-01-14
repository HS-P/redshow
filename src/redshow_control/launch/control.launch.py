#!/usr/bin/env python3
"""
라즈베리파이용 Control Launch 파일
BNO085 노드와 Control 노드를 함께 실행합니다.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 1. BNO085 IMU 센서 노드
        Node(
            package='redshow_control',
            executable='bno085_node',
            name='bno085_node',
            output='screen',
            parameters=[],
        ),
        
        # # 2. Adaptation Module 노드 (A-RMA: 70개 observation history 수집 및 EX_OBS 생성)
        # Node(
        #     package='redshow_control',
        #     executable='adaptation_module_node',
        #     name='adaptation_module_node',
        #     output='screen',
        # ),
        
        # 3. Control 노드 (OpenCR Firmware와 통신, Serial Observation + IMU + EX_OBS 통합)
        Node(
            package='redshow_control',
            executable='redshow_control_node',
            name='berkeley_control_node',
            output='screen',
            parameters=[
                {'policy_name': 'model_4800.pt'},  # 기본 모델 파일명
            ],
        ),
        
        # Record 노드 (SYSID 데이터 기록) - 필요시 주석 해제
        Node(
            package='redshow_record',
            executable='record_node',
            name='record_node',
            output='screen',
        ),
    ])

