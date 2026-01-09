#!/usr/bin/env python3
"""
노트북 PC용 GUI Launch 파일
GUI 노드를 실행합니다.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # GUI 노드
        Node(
            package='redshow_gui',
            executable='redshow_gui',
            name='redshow_gui',
            output='screen',
        ),
    ])

