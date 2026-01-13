#!/usr/bin/env python3
"""
SYSID를 위한 데이터 기록 노드
GUI에서 RECORD 명령을 받으면 모든 Observation과 Action 데이터를 CSV로 저장
"""
import os
import csv
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray


class RecordNode(Node):
    def __init__(self):
        super().__init__('record_node')
        
        # 기록 상태
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.record_start_time = None
        
        # 데이터 버퍼 (최신 값 저장)
        self.data_buffer = {
            'timestamp': None,
            'leg_position': None,
            'wheel_velocity': None,
            'base_ang_vel': None,
            'velocity_commands': None,
            'base_quat': None,
            'base_rpy': None,
            'actions': None,
        }
        
        # RECORD 명령 구독
        self.cmd_sub = self.create_subscription(
            String,
            'redshow/cmd',
            self.cmd_callback,
            10
        )
        
        # Observation 토픽 구독
        self.leg_position_sub = self.create_subscription(
            Float64MultiArray,
            '/Redshow/Observation/leg_position',
            self.leg_position_callback,
            10
        )
        self.wheel_velocity_sub = self.create_subscription(
            Float64MultiArray,
            '/Redshow/Observation/wheel_velocity',
            self.wheel_velocity_callback,
            10
        )
        self.base_ang_vel_sub = self.create_subscription(
            Float64MultiArray,
            '/Redshow/Observation/base_ang_vel',
            self.base_ang_vel_callback,
            10
        )
        self.velocity_commands_sub = self.create_subscription(
            Float64MultiArray,
            '/Redshow/Observation/velocity_commands',
            self.velocity_commands_callback,
            10
        )
        self.base_quat_sub = self.create_subscription(
            Float64MultiArray,
            '/Redshow/Observation/base_quat',
            self.base_quat_callback,
            10
        )
        self.base_rpy_sub = self.create_subscription(
            Float64MultiArray,
            '/Redshow/Observation/base_rpy',
            self.base_rpy_callback,
            10
        )
        self.actions_sub = self.create_subscription(
            Float64MultiArray,
            '/Redshow/Observation/actions',
            self.actions_callback,
            10
        )
        
        # CSV 저장 디렉토리
        self.data_dir = Path.home() / 'redshow_data'
        self.data_dir.mkdir(exist_ok=True)
        
        self.get_logger().info(f"[RECORD] Node initialized. Data will be saved to: {self.data_dir}")
    
    def cmd_callback(self, msg):
        """RECORD 명령 처리"""
        cmd = msg.data.strip().upper()
        
        if cmd == 'RECORD':
            self.start_recording()
        elif cmd == 'STOP_RECORD':
            self.stop_recording()
    
    def start_recording(self):
        """CSV 파일 생성 및 기록 시작"""
        if self.is_recording:
            self.get_logger().warn("[RECORD] Already recording!")
            return
        
        # 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.data_dir / f'record_{timestamp}.csv'
        
        # CSV 파일 열기
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # CSV 헤더 작성
        header = [
            'timestamp',
            'leg_position_0', 'leg_position_1', 'leg_position_2', 'leg_position_3',
            'wheel_velocity_0', 'wheel_velocity_1',
            'base_ang_vel_0', 'base_ang_vel_1', 'base_ang_vel_2',
            'velocity_commands_0', 'velocity_commands_1', 'velocity_commands_2', 'velocity_commands_3',
            'base_quat_0', 'base_quat_1', 'base_quat_2', 'base_quat_3',
            'base_rpy_0', 'base_rpy_1', 'base_rpy_2',
            'actions_0', 'actions_1', 'actions_2', 'actions_3', 'actions_4', 'actions_5',
        ]
        self.csv_writer.writerow(header)
        
        self.is_recording = True
        self.record_start_time = time.time()
        self.get_logger().info(f"[RECORD] Started recording to: {filename}")
    
    def stop_recording(self):
        """기록 중지 및 파일 닫기"""
        if not self.is_recording:
            self.get_logger().warn("[RECORD] Not recording!")
            return
        
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        record_duration = time.time() - self.record_start_time if self.record_start_time else 0
        self.is_recording = False
        self.record_start_time = None
        
        self.get_logger().info(f"[RECORD] Stopped recording. Duration: {record_duration:.2f}s")
    
    def write_data_row(self):
        """현재 버퍼의 데이터를 CSV에 기록"""
        if not self.is_recording or not self.csv_writer:
            return
        
        # 모든 데이터가 준비되었는지 확인
        if any(self.data_buffer[k] is None for k in ['leg_position', 'wheel_velocity', 
                                                      'base_ang_vel', 'velocity_commands',
                                                      'base_quat', 'base_rpy', 'actions']):
            return
        
        # 타임스탬프 (기록 시작 시간 기준 상대 시간)
        rel_time = time.time() - self.record_start_time if self.record_start_time else 0
        
        # 데이터 행 작성
        row = [
            rel_time,
            *self.data_buffer['leg_position'],
            *self.data_buffer['wheel_velocity'],
            *self.data_buffer['base_ang_vel'],
            *self.data_buffer['velocity_commands'],
            *self.data_buffer['base_quat'],
            *self.data_buffer['base_rpy'],
            *self.data_buffer['actions'],
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # 즉시 디스크에 쓰기
    
    def leg_position_callback(self, msg):
        self.data_buffer['leg_position'] = list(msg.data)
        self.write_data_row()
    
    def wheel_velocity_callback(self, msg):
        self.data_buffer['wheel_velocity'] = list(msg.data)
        self.write_data_row()
    
    def base_ang_vel_callback(self, msg):
        self.data_buffer['base_ang_vel'] = list(msg.data)
        self.write_data_row()
    
    def velocity_commands_callback(self, msg):
        self.data_buffer['velocity_commands'] = list(msg.data)
        self.write_data_row()
    
    def base_quat_callback(self, msg):
        self.data_buffer['base_quat'] = list(msg.data)
        self.write_data_row()
    
    def base_rpy_callback(self, msg):
        self.data_buffer['base_rpy'] = list(msg.data)
        self.write_data_row()
    
    def actions_callback(self, msg):
        self.data_buffer['actions'] = list(msg.data)
        self.write_data_row()
    
    def __del__(self):
        """소멸자: 파일 닫기"""
        if self.is_recording:
            self.stop_recording()


def main(args=None):
    rclpy.init(args=args)
    node = RecordNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_recording()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


