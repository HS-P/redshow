"""
실시간 모니터링 GUI - PySide6 + ROS2
200Hz 업데이트, Manual/Auto Control, Graph 표시
"""
import sys
import os
import time
import threading
from collections import deque
from pathlib import Path
import warnings

# matplotlib 3D projection 경고 억제 (3D 기능을 사용하지 않으므로)
warnings.filterwarnings('ignore', message='.*Axes3D.*')

import numpy as np
import torch
import yaml

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QGridLayout,
    QDoubleSpinBox, QScrollArea, QStackedWidget, QTabWidget
)
from PySide6.QtCore import QTimer, Qt, Signal, QObject
from PySide6.QtGui import QFont, QColor, QPalette

import matplotlib
# PySide6와 호환되는 backend 사용
# matplotlib 3.5.1은 PySide6를 완전히 지원하지 않으므로
# 직접 Qt5 backend를 사용하되, 호환성 문제를 우회
# matplotlib.use()는 import 전에 호출해야 함

# matplotlib 3.5.1과 PySide6 호환성 문제 해결
# backend_qtagg가 내부적으로 Qt6를 사용하려고 하므로, 
# 직접 Qt5Agg를 사용하되 import 에러를 처리
import sys

# PySide6와 호환되는 백엔드 사용
# matplotlib 3.5.1은 PySide6를 완전히 지원하지 않으므로
# PyQt5를 설치하거나 matplotlib을 3.6.0 이상으로 업그레이드 필요
USE_QT_BACKEND = False
try:
    # 먼저 QtAgg 시도 (PySide6 지원, matplotlib 3.6.0+ 필요)
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    USE_QT_BACKEND = True
    print("Using QtAgg backend (PySide6 compatible)")
except (ImportError, TypeError, AttributeError) as e1:
    try:
        # QtAgg 실패 시 Qt5Agg 시도 (PyQt5 필요)
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        USE_QT_BACKEND = True
        print("Using Qt5Agg backend")
    except (ImportError, TypeError, AttributeError) as e2:
        # Qt 백엔드 실패 시 TkAgg 사용 (비상용, 그래프 표시 불가)
        print(f"Warning: Qt backends failed (QtAgg: {e1}, Qt5Agg: {e2})")
        print("To enable graphs, please:")
        print("  1. Upgrade matplotlib: pip install --upgrade matplotlib>=3.6.0")
        print("  2. Or install PyQt5: pip install PyQt5")
        print("Using TkAgg backend as fallback (graphs will not be displayed)")
        matplotlib.use('TkAgg')
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
        except ImportError:
            # 최후의 수단: Agg (non-interactive)
            matplotlib.use('Agg')
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            print("Warning: Using Agg backend (non-interactive plots)")

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
import json


# =========================
# ROS2 NODE
# =========================
class ROS2Node(Node):
    def __init__(self):
        super().__init__('gui_node')
        self.cmd_pub = self.create_publisher(String, 'redshow/cmd', 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, 'redshow/joint_cmd', 10)
        self.velocity_command_pub = self.create_publisher(Float64MultiArray, 'redshow/velocity_command', 10)
        self.model_path_pub = self.create_publisher(String, 'redshow/model_path', 10)
        # 개별 Observation 토픽 구독자
        self.leg_position_sub = None
        self.wheel_velocity_sub = None
        self.base_ang_vel_sub = None
        self.velocity_commands_sub = None
        self.base_quat_sub = None
        self.base_rpy_sub = None
        self.actions_sub = None
        
        self.policy_hz_sub = None
        self.shutdown_sub = None
        self.obs_config_sub = None
        self.extrinsics_obs_sub = None
        self.feedback_data = None
        self.last_feedback_time = None
        self.feedback_count = 0
        self.feedback_hz_start_time = time.time()

    def publish_cmd(self, cmd):
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)

    def publish_joint_cmd(self, joints):
        msg = Float64MultiArray()
        msg.data = joints
        self.joint_pub.publish(msg)
    
    def publish_velocity_command(self, vcmd):
        msg = Float64MultiArray()
        msg.data = vcmd
        self.velocity_command_pub.publish(msg)
    
    def publish_model_path(self, model_path):
        msg = String()
        msg.data = model_path
        self.model_path_pub.publish(msg)
    
    def setup_feedback_subscribers(self, callbacks):
        """개별 Observation 토픽 구독 설정"""
        self.leg_position_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/leg_position', callbacks['leg_position'], 10
        )
        self.wheel_velocity_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/wheel_velocity', callbacks['wheel_velocity'], 10
        )
        self.base_ang_vel_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/base_ang_vel', callbacks['base_ang_vel'], 10
        )
        self.velocity_commands_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/velocity_commands', callbacks['velocity_commands'], 10
        )
        self.base_quat_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/base_quat', callbacks['base_quat'], 10
        )
        self.base_rpy_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/base_rpy', callbacks['base_rpy'], 10
        )
        self.actions_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/actions', callbacks['actions'], 10
        )
        self.get_logger().info("[ROS2] Subscribed to individual observation topics")
    
    def setup_policy_hz_subscriber(self, callback):
        self.policy_hz_sub = self.create_subscription(
            Float64MultiArray, 'redshow/policy_hz', callback, 10
        )
        self.get_logger().info("[ROS2] Subscribed to redshow/policy_hz")
    
    def setup_shutdown_subscriber(self, callback):
        self.shutdown_sub = self.create_subscription(
            String, 'redshow/shutdown', callback, 10
        )
    
    def setup_extrinsics_obs_subscriber(self, callback):
        self.extrinsics_obs_sub = self.create_subscription(
            Float64MultiArray, 'redshow/extrinsics_obs', callback, 10
        )
        self.get_logger().info("[ROS2] Subscribed to redshow/extrinsics_obs")

    def get_feedback_hz(self):
        """Observation Hz 계산"""
        current_time = time.time()
        elapsed = current_time - self.feedback_hz_start_time
        if elapsed >= 1.0:
            hz = self.feedback_count / elapsed
            self.feedback_count = 0
            self.feedback_hz_start_time = current_time
            return hz
        return None


# =========================
# SIGNAL EMITTER
# =========================
class SignalEmitter(QObject):
    feedback_received = Signal(object)


# =========================
# MAIN GUI
# =========================
class MonitorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOL Monitor - PySide6")
        self.setGeometry(100, 100, 1800, 1000)

        # ---- state ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_mode = None  # "MANUAL" or "AUTO"
        self.is_running = False
        self.is_recording = False  # RECORD 상태 초기화
        self.selected_mode_btn = None
        self.current_file_path = None
        
        # ---- Observation 상태 ----
        self.obs_ready = False
        self.obs_hz = 0.0
        self.last_obs_time = None
        self.obs_hz_buffer = deque(maxlen=100)  # Hz 계산용
        
        # 각 Observation 인덱스별 상태 추적
        self.obs_index_status = {}  # {index: {'ready': bool, 'last_time': float, 'hz': float}}
        self.obs_index_hz_buffers = {}  # {index: deque} - 각 인덱스별 Hz 버퍼
        # 초기에는 23개로 시작, PT 파일 로드 후 num_actor_obs로 업데이트됨
        for idx in range(23):
            self.obs_index_status[idx] = {'ready': False, 'last_time': None, 'hz': 0.0}
            self.obs_index_hz_buffers[idx] = deque(maxlen=50)
        
        # ---- Policy Hz ----
        self.policy_hz = 0.0
        self.policy_hz_buffer = deque(maxlen=100)
        
        # ---- buffers ----
        self.max_history = 500
        self.time_buffer = deque(maxlen=self.max_history)
        self.obs_buffer = {}  # 각 observation별 버퍼
        self.act_buffer = deque(maxlen=self.max_history)
        
        # Velocity commands (GUI에서 설정)
        self.velocity_commands = [0.0, 0.0, 0.0, 0.0]
        
        # A-RMA 관련
        self.arma_file_path = None
        self.is_arma_model = False
        
        # Observation 그룹 정의 - PT 파일에서 로드할 때까지 빈 상태
        # 사전 정의하지 않음, PT 파일에서 정보를 받아서 설정
        self.obs_groups = {}
        self.num_actor_obs = 23  # 기본값 (PT 파일에서 업데이트됨)
        
        # 일단 기본 인덱스만 버퍼 초기화 (나중에 obs_groups가 설정되면 업데이트됨)
        for idx in range(23):
            self.obs_buffer[idx] = deque(maxlen=self.max_history)

        # ---- ROS2 ----
        rclpy.init()
        self.ros2_node = ROS2Node()
        
        # 개별 Observation 토픽 콜백 설정
        observation_callbacks = {
            'leg_position': lambda msg: self.observation_callback('leg_position', msg),
            'wheel_velocity': lambda msg: self.observation_callback('wheel_velocity', msg),
            'base_ang_vel': lambda msg: self.observation_callback('base_ang_vel', msg),
            'velocity_commands': lambda msg: self.observation_callback('velocity_commands', msg),
            'base_quat': lambda msg: self.observation_callback('base_quat', msg),
            'base_rpy': lambda msg: self.observation_callback('base_rpy', msg),
            'actions': lambda msg: self.observation_callback('actions', msg),
        }
        self.ros2_node.setup_feedback_subscribers(observation_callbacks)
        self.ros2_node.setup_policy_hz_subscriber(self.policy_hz_callback)
        self.ros2_node.setup_shutdown_subscriber(self.shutdown_callback)
        self.ros2_node.setup_extrinsics_obs_subscriber(self.extrinsics_obs_callback)
        threading.Thread(target=self.ros_spin, daemon=True).start()

        # ---- timers ----
        self.manual_timer = QTimer()
        self.manual_timer.timeout.connect(self.send_manual_joint_cmd)

        # GUI 업데이트 타이머 (50Hz로 최적화 - 그래프는 더 낮은 주기로)
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(20)  # 20ms = 50Hz
        
        # 그래프 업데이트는 별도 타이머로 (10Hz)
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.update_graphs)
        # 초기에는 시작하지 않음 (Graph 탭이 활성화될 때 시작)

        # Hz 계산용 타이머 (1초마다)
        self.hz_timer = QTimer()
        self.hz_timer.timeout.connect(self.update_hz)
        self.hz_timer.start(1000)  # 1초마다

        self.init_ui()

    # =========================
    # ROS
    # =========================
    def ros_spin(self):
        while rclpy.ok():
            rclpy.spin_once(self.ros2_node, timeout_sec=0.01)
    
    def observation_callback(self, group_name: str, msg: Float64MultiArray):
        """개별 Observation 토픽 콜백"""
        if not self.obs_groups or group_name not in self.obs_groups:
            return
        
        current_time = time.time()
        group_info = self.obs_groups[group_name]
        indices = group_info['indices']
        data = msg.data
        
        # 데이터 길이 확인
        expected_len = len(indices)
        if len(data) != expected_len:
            if not hasattr(self, '_obs_length_warned'):
                self._obs_length_warned = {}
            if group_name not in self._obs_length_warned:
                self.get_logger().warn(
                    f"[GUI] {group_name} data length mismatch: "
                    f"expected {expected_len}, got {len(data)}"
                )
                self._obs_length_warned[group_name] = True
            return
        
        any_data_ready = False
        
        # 각 인덱스별로 데이터 처리
        for i, env_idx in enumerate(indices):
            if i < len(data):
                value = data[i]
                
                # 인덱스 상태 초기화 (없으면 생성)
                if env_idx not in self.obs_index_status:
                    self.obs_index_status[env_idx] = {'ready': False, 'last_time': None, 'hz': 0.0}
                    self.obs_index_hz_buffers[env_idx] = deque(maxlen=50)
                    if env_idx not in self.obs_buffer:
                        self.obs_buffer[env_idx] = deque(maxlen=self.max_history)
                
                # Hz 계산
                if self.obs_index_status[env_idx]['last_time'] is not None:
                    dt = current_time - self.obs_index_status[env_idx]['last_time']
                    if dt > 0 and dt < 1.0:
                        hz = 1.0 / dt
                        if 0 < hz < 1000:
                            self.obs_index_hz_buffers[env_idx].append(hz)
                            if len(self.obs_index_hz_buffers[env_idx]) > 0:
                                self.obs_index_status[env_idx]['hz'] = np.mean(self.obs_index_hz_buffers[env_idx])
                
                # 상태 업데이트
                self.obs_index_status[env_idx]['ready'] = True
                self.obs_index_status[env_idx]['last_time'] = current_time
                any_data_ready = True
                
                # 버퍼에 추가
                if env_idx in self.obs_buffer:
                    self.obs_buffer[env_idx].append(value)
        
        # 전체 상태 업데이트
        if any_data_ready:
            self.obs_ready = True
            self.last_obs_time = current_time
            
            # 전체 Hz 계산
            if hasattr(self, '_prev_obs_time') and self._prev_obs_time is not None:
                dt = current_time - self._prev_obs_time
                if dt > 0 and dt < 1.0:
                    hz = 1.0 / dt
                    if 0 < hz < 1000:
                        self.obs_hz_buffer.append(hz)
            self._prev_obs_time = current_time
        
        # time_buffer 업데이트
        self.time_buffer.append(len(self.time_buffer))
        self.ros2_node.feedback_count += 1
    
    def show_warning(self, message: str, throttle_sec: float = 5.0):
        """시각적 경고 메시지 표시 (throttle로 중복 방지)"""
        current_time = time.time()
        
        # 같은 메시지가 최근에 표시되었는지 확인
        if not hasattr(self, '_last_warning_time'):
            self._last_warning_time = {}
        
        warning_key = message[:50]  # 메시지의 처음 50자로 키 생성
        last_time = self._last_warning_time.get(warning_key, 0)
        
        if current_time - last_time < throttle_sec:
            return  # 최근에 표시했으면 건너뜀
        
        self._last_warning_time[warning_key] = current_time
        self.warning_label.setText(message)
        self.warning_label.show()
        self.get_logger().warn(f"[GUI WARNING] {message}")
    
    def policy_hz_callback(self, msg):
        """Policy Hz 콜백"""
        if len(msg.data) > 0:
            self.policy_hz = msg.data[0]
            # 첫 번째 메시지 수신 시 로그 출력 (한 번만)
            if not hasattr(self, '_policy_hz_received'):
                self.get_logger().info(f"[GUI] Policy Hz received: {self.policy_hz:.1f} Hz")
                self._policy_hz_received = True
    
    def extrinsics_obs_callback(self, msg):
        """Extrinsics Observation 콜백 (A-RMA Adaptation Module에서 발행)"""
        if not self.obs_groups or 'extrinsics_obs' not in self.obs_groups:
            return
        
        if len(msg.data) != 8:
            self.get_logger().warn(f"[GUI] Invalid extrinsics_obs data length: {len(msg.data)}, expected 8")
            return
        
        current_time = time.time()
        extrinsics_indices = self.obs_groups['extrinsics_obs']['indices']
        
        # extrinsics_obs 데이터를 마지막 8개 인덱스에 매핑
        for i, env_idx in enumerate(extrinsics_indices):
            if i < len(msg.data):
                value = msg.data[i]
                
                # Hz 계산
                if env_idx in self.obs_index_status and self.obs_index_status[env_idx]['last_time'] is not None:
                    dt = current_time - self.obs_index_status[env_idx]['last_time']
                    if dt > 0 and dt < 1.0:
                        hz = 1.0 / dt
                        if 0 < hz < 1000:
                            self.obs_index_hz_buffers[env_idx].append(hz)
                            if len(self.obs_index_hz_buffers[env_idx]) > 0:
                                self.obs_index_status[env_idx]['hz'] = np.mean(self.obs_index_hz_buffers[env_idx])
                
                # 상태 업데이트
                self.obs_index_status[env_idx]['ready'] = True
                self.obs_index_status[env_idx]['last_time'] = current_time
                
                # 버퍼에 추가
                if env_idx in self.obs_buffer:
                    self.obs_buffer[env_idx].append(value)
        
        # 첫 번째 메시지 수신 시 로그 출력
        if not hasattr(self, '_extrinsics_obs_received'):
            self.get_logger().info(f"[GUI] Extrinsics obs received: {len(msg.data)} values")
            self._extrinsics_obs_received = True
    
    def shutdown_callback(self, msg):
        """Control node 종료 신호 콜백"""
        if msg.data.strip() == "shutdown":
            self.get_logger().info("[GUI] Received shutdown signal from control node. Closing GUI...")
            # GUI 종료
            QTimer.singleShot(100, self.close)  # 100ms 후 종료 (메시지 처리 완료 대기)

    def update_hz(self):
        """Hz 업데이트 (1초마다 호출)"""
        # Observation Hz 계산 (최근 1초간의 평균)
        if len(self.obs_hz_buffer) > 0:
            # 이상치 제거 (너무 큰 값이나 작은 값 제거)
            hz_values = list(self.obs_hz_buffer)
            if len(hz_values) > 10:
                # 상위/하위 10% 제거 후 평균
                hz_values.sort()
                trim_size = max(1, len(hz_values) // 10)
                trimmed = hz_values[trim_size:-trim_size]
                self.obs_hz = np.mean(trimmed) if len(trimmed) > 0 else np.mean(hz_values)
            else:
                self.obs_hz = np.mean(hz_values)
        else:
            self.obs_hz = 0.0
        
        # Observation 상태 확인 (1초 이상 데이터가 없으면 None)
        if self.last_obs_time is not None:
            elapsed = time.time() - self.last_obs_time
            if elapsed > 1.0:
                self.obs_ready = False
                self.obs_hz = 0.0
        
        # Policy Hz는 policy_hz_callback에서 업데이트됨
        # AUTO 모드가 아니면 0으로 설정
        if self.current_mode != "AUTO":
            self.policy_hz = 0.0

    # =========================
    # UI SETUP
    # =========================
    def init_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        layout = QHBoxLayout(cw)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 좌측 패널
        layout.addWidget(self.make_left_panel(), 1)
        # 오른쪽 패널
        layout.addWidget(self.make_right_panel(), 2)

    def make_left_panel(self):
        """좌측 패널: 파일 열기, 상태 표시"""
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(10)
        
        # 파일 열기 버튼
        file_group = QGroupBox("Model File")
        file_layout = QVBoxLayout()
        
        # Policy 파일 선택
        policy_file_layout = QHBoxLayout()
        self.open_file_btn = QPushButton("Policy 파일 선택")
        self.open_file_btn.clicked.connect(self.open_file_dialog)
        policy_file_layout.addWidget(self.open_file_btn)
        
        # A-RMA 파일 선택
        self.open_arma_file_btn = QPushButton("A-RMA 파일 선택")
        self.open_arma_file_btn.clicked.connect(self.open_arma_file_dialog)
        policy_file_layout.addWidget(self.open_arma_file_btn)
        
        file_layout.addLayout(policy_file_layout)
        
        # Policy 파일 표시
        self.current_file_label = QLabel("현재 선택된 파일이 없습니다.")
        self.current_file_label.setWordWrap(True)
        self.current_file_label.setAlignment(Qt.AlignCenter)
        self.current_file_label.setMinimumHeight(50)
        self.current_file_label.setStyleSheet(
            "padding: 10px; background-color: #ff4444; color: white; "
            "font-weight: bold; font-size: 11pt; border-radius: 5px;"
        )
        file_layout.addWidget(self.current_file_label)
        
        # A-RMA 파일 표시
        self.arma_file_label = QLabel("A-RMA 파일이 선택되지 않았습니다.")
        self.arma_file_label.setWordWrap(True)
        self.arma_file_label.setAlignment(Qt.AlignCenter)
        self.arma_file_label.setMinimumHeight(50)
        self.arma_file_label.setStyleSheet(
            "padding: 10px; background-color: #666; color: white; "
            "font-weight: bold; font-size: 11pt; border-radius: 5px;"
        )
        file_layout.addWidget(self.arma_file_label)
        
        # A-RMA 상태 표시
        self.arma_status_label = QLabel("A-RMA: OFF")
        self.arma_status_label.setAlignment(Qt.AlignCenter)
        self.arma_status_label.setStyleSheet(
            "padding: 10px; background-color: #666; color: white; "
            "font-weight: bold; font-size: 12pt; border-radius: 5px;"
        )
        file_layout.addWidget(self.arma_status_label)
        
        # 경고 메시지 표시 영역
        self.warning_label = QLabel("")
        self.warning_label.setAlignment(Qt.AlignCenter)
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "padding: 15px; background-color: #ff4444; color: white; "
            "font-weight: bold; font-size: 11pt; border-radius: 5px;"
        )
        self.warning_label.hide()
        file_layout.addWidget(self.warning_label)
        
        file_group.setLayout(file_layout)
        l.addWidget(file_group)
        
        # Observation 상태 (각 인덱스별)
        obs_status_group = QGroupBox("Observation Status")
        obs_status_layout = QVBoxLayout()
        
        # 토픽명 표시
        obs_topic_label = QLabel("Topics: /Redshow/Observation/*")
        obs_topic_label.setStyleSheet("font-size: 9pt; color: #666; padding: 2px;")
        obs_status_layout.addWidget(obs_topic_label)
        
        # 스크롤 가능한 영역 생성
        obs_scroll = QScrollArea()
        obs_scroll.setWidgetResizable(True)
        obs_scroll.setMaximumHeight(400)
        obs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        obs_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        obs_content = QWidget()
        obs_content_layout = QVBoxLayout(obs_content)
        obs_content_layout.setSpacing(5)
        
        # Observation 그룹별 상태 표시 (PT 파일에서 로드될 때까지 빈 상태)
        self.obs_status_labels = {}  # {group_name: {index: label}}
        self.obs_status_container = obs_content  # 나중에 업데이트하기 위해 저장
        self.obs_status_layout = obs_content_layout
        
        # 초기 메시지 표시
        self.obs_status_placeholder = QLabel("PT 파일을 선택하면 Observation Group 정보가 표시됩니다.")
        self.obs_status_placeholder.setAlignment(Qt.AlignCenter)
        self.obs_status_placeholder.setStyleSheet("padding: 20px; color: #666; font-size: 10pt;")
        obs_content_layout.addWidget(self.obs_status_placeholder)
        
        obs_content_layout.addStretch()
        obs_scroll.setWidget(obs_content)
        obs_status_layout.addWidget(obs_scroll)
        
        # 전체 Observation Hz 표시
        self.obs_hz_label = QLabel("Overall Observation Hz: 0.0 Hz")
        self.obs_hz_label.setAlignment(Qt.AlignCenter)
        self.obs_hz_label.setStyleSheet("font-size: 11pt; padding: 5px; font-weight: bold;")
        obs_status_layout.addWidget(self.obs_hz_label)
        
        obs_status_group.setLayout(obs_status_layout)
        l.addWidget(obs_status_group)
        
        # Policy Hz 표시
        policy_hz_group = QGroupBox("Policy Hz")
        policy_hz_layout = QVBoxLayout()
        
        # 토픽명 표시
        policy_topic_label = QLabel("Topic: redshow/policy_hz")
        policy_topic_label.setStyleSheet("font-size: 9pt; color: #666; padding: 2px;")
        policy_hz_layout.addWidget(policy_topic_label)
        
        self.policy_hz_label = QLabel("Policy Hz: 0.0 Hz")
        self.policy_hz_label.setAlignment(Qt.AlignCenter)
        self.policy_hz_label.setStyleSheet("font-size: 12pt; padding: 5px;")
        policy_hz_layout.addWidget(self.policy_hz_label)
        
        policy_hz_group.setLayout(policy_hz_layout)
        l.addWidget(policy_hz_group)
        
        l.addStretch()
        return w

    def make_right_panel(self):
        """오른쪽 패널: Manual/Auto/Graph 탭"""
        w = QWidget()
        l = QVBoxLayout(w)
        
        # 탭 위젯
        self.tab_widget = QTabWidget()
        
        # Manual Control 탭
        manual_tab = self.make_manual_control_tab()
        self.tab_widget.addTab(manual_tab, "Manual Control")
        
        # Auto Control 탭
        auto_tab = self.make_auto_control_tab()
        self.tab_widget.addTab(auto_tab, "Auto Control")
        
        # Graph 탭
        graph_tab = self.make_graph_tab()
        self.tab_widget.addTab(graph_tab, "Graph")
        
        # 탭 변경 시 그래프 타이머 시작/중지
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        l.addWidget(self.tab_widget)
        
        # 하단에 모드 선택 버튼과 RUN 버튼
        control_btn_group = QGroupBox("Control")
        control_btn_layout = QHBoxLayout()
        
        self.manual_mode_btn = QPushButton("Manual")
        self.auto_mode_btn = QPushButton("Auto")
        self.run_btn = QPushButton("RUN")
        self.record_btn = QPushButton("RECORD")
        
        self.manual_mode_btn.setMinimumHeight(50)
        self.auto_mode_btn.setMinimumHeight(50)
        self.run_btn.setMinimumHeight(50)
        self.record_btn.setMinimumHeight(50)
        
        self.manual_mode_btn.clicked.connect(self.on_manual_mode)
        self.auto_mode_btn.clicked.connect(self.on_auto_mode)
        self.run_btn.clicked.connect(self.on_run)
        self.record_btn.clicked.connect(self.on_record)
        
        control_btn_layout.addWidget(self.manual_mode_btn)
        control_btn_layout.addWidget(self.auto_mode_btn)
        control_btn_layout.addWidget(self.run_btn)
        control_btn_layout.addWidget(self.record_btn)
        
        control_btn_group.setLayout(control_btn_layout)
        
        # RECORD 버튼 초기 스타일 설정
        self.update_record_button_style()
        l.addWidget(control_btn_group)
        
        return w

    def make_manual_control_tab(self):
        """Manual Control 탭"""
        w = QWidget()
        l = QVBoxLayout(w)
        
        # Velocity Commands 입력
        vcmd_group = QGroupBox("Velocity Commands")
        vcmd_layout = QGridLayout()
        
        self.velocity_command_inputs = []
        vcmd_names = ["Velocity X", "Velocity Y", "Velocity Z", "Heading"]
        
        for i, name in enumerate(vcmd_names):
            label = QLabel(name)
            label.setMinimumWidth(100)
            sb = QDoubleSpinBox()
            sb.setRange(-10.0, 10.0)
            sb.setDecimals(4)
            sb.setValue(0.0)  # 초기값 0
            sb.setSingleStep(0.1)
            sb.valueChanged.connect(self.send_manual_joint_cmd)
            
            vcmd_layout.addWidget(label, i, 0)
            vcmd_layout.addWidget(sb, i, 1)
            self.velocity_command_inputs.append(sb)
        
        vcmd_group.setLayout(vcmd_layout)
        l.addWidget(vcmd_group)
        
        # Action 입력
        manual_group = QGroupBox("Manual Action Input")
        manual_layout = QGridLayout()
        
        self.manual_inputs = []
        names = ["Action[0]", "Action[1]", "Action[2]", "Action[3]", "Action[4]", "Action[5]"]
        
        for i, name in enumerate(names):
            label = QLabel(name)
            label.setMinimumWidth(100)
            sb = QDoubleSpinBox()
            sb.setRange(-10.0, 10.0)
            sb.setDecimals(4)
            sb.setValue(0.0)  # 초기값 0
            sb.setSingleStep(0.1)
            sb.valueChanged.connect(self.send_manual_joint_cmd)
            
            manual_layout.addWidget(label, i, 0)
            manual_layout.addWidget(sb, i, 1)
            self.manual_inputs.append(sb)
        
        manual_group.setLayout(manual_layout)
        l.addWidget(manual_group)
        l.addStretch()
        
        return w

    def make_auto_control_tab(self):
        """Auto Control 탭"""
        w = QWidget()
        l = QVBoxLayout(w)
        
        # Velocity Commands 입력
        vcmd_group = QGroupBox("Velocity Commands")
        vcmd_layout = QGridLayout()
        
        self.auto_velocity_command_inputs = []
        vcmd_names = ["Velocity X", "Velocity Y", "Velocity Z", "Heading"]
        
        for i, name in enumerate(vcmd_names):
            label = QLabel(name)
            label.setMinimumWidth(100)
            sb = QDoubleSpinBox()
            sb.setRange(-10.0, 10.0)
            sb.setDecimals(4)
            sb.setValue(0.0)  # 초기값 0
            sb.setSingleStep(0.1)
            # Auto 모드에서는 velocity command 변경 시 control node에 전송 필요
            sb.valueChanged.connect(self.on_velocity_command_changed)
            
            vcmd_layout.addWidget(label, i, 0)
            vcmd_layout.addWidget(sb, i, 1)
            self.auto_velocity_command_inputs.append(sb)
        
        vcmd_group.setLayout(vcmd_layout)
        l.addWidget(vcmd_group)
        
        info_label = QLabel("Auto 모드에서는 Policy가 자동으로 Action을 생성합니다.\nVelocity Commands는 위에서 설정할 수 있습니다.")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 12pt; padding: 20px;")
        l.addWidget(info_label)
        
        l.addStretch()
        return w

    def make_graph_tab(self):
        """Graph 탭: Observation 그래프들 (3개씩 표시, 스크롤/다음 버튼)"""
        # 메인 위젯
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 상단 컨트롤 (이전/다음 버튼)
        control_layout = QHBoxLayout()
        self.graph_prev_btn = QPushButton("◀ 이전")
        self.graph_prev_btn.clicked.connect(self.graph_prev_page)
        self.graph_next_btn = QPushButton("다음 ▶")
        self.graph_next_btn.clicked.connect(self.graph_next_page)
        self.graph_page_label = QLabel("페이지: 1/1")
        control_layout.addWidget(self.graph_prev_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.graph_page_label)
        control_layout.addStretch()
        control_layout.addWidget(self.graph_next_btn)
        main_layout.addLayout(control_layout)
        
        # 그래프 표시 영역 (스크롤 없음)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 스크롤바 제거
        
        content_widget = QWidget()
        self.graph_layout = QVBoxLayout(content_widget)
        self.graph_layout.setSpacing(10)
        self.graph_layout.setContentsMargins(5, 5, 5, 5)
        
        # 각 그룹별로 그래프 생성 (obs_groups가 설정된 후에만 생성됨)
        self.obs_figs = {}
        self.obs_canvases = {}
        self.obs_axes = {}
        self.obs_lines = {}
        self.graph_current_page = 0
        self.graph_items_per_page = 2
        
        # 초기 메시지 표시
        graph_placeholder = QLabel("PT 파일을 선택하면 Observation 그래프가 표시됩니다.")
        graph_placeholder.setAlignment(Qt.AlignCenter)
        graph_placeholder.setStyleSheet("padding: 50px; color: #666; font-size: 12pt;")
        self.graph_layout.addWidget(graph_placeholder)
        
        self.graph_layout.addStretch()
        
        # 컨텐츠 위젯 크기 설정
        content_widget.setMinimumWidth(800)
        content_widget.setMinimumHeight(1100)  # 2개 그래프를 위한 높이
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        return main_widget
    
    def graph_prev_page(self):
        """이전 페이지로 이동"""
        if self.graph_current_page > 0:
            self.graph_current_page -= 1
            self.update_graph_display()
    
    def graph_next_page(self):
        """다음 페이지로 이동"""
        if self.obs_groups:
            total_pages = (len(self.obs_groups) + self.graph_items_per_page - 1) // self.graph_items_per_page
            if self.graph_current_page < total_pages - 1:
                self.graph_current_page += 1
                self.update_graph_display()
    
    def update_graph_display(self):
        """현재 페이지에 해당하는 그래프만 표시"""
        if not self.obs_groups or not hasattr(self, 'graph_layout'):
            return
        
        # 모든 그래프 숨기기 (Qt 위젯인 경우만)
        for group_name in self.obs_groups.keys():
            if group_name in self.obs_canvases:
                canvas = self.obs_canvases[group_name]
                if isinstance(canvas, QWidget):
                    if hasattr(canvas, 'setVisible'):
                        canvas.setVisible(False)
                    elif hasattr(canvas, 'hide'):
                        canvas.hide()
        
        # 현재 페이지의 그래프만 표시
        group_names = list(self.obs_groups.keys())
        start_idx = self.graph_current_page * self.graph_items_per_page
        end_idx = min(start_idx + self.graph_items_per_page, len(group_names))
        
        for i in range(start_idx, end_idx):
            group_name = group_names[i]
            if group_name in self.obs_canvases:
                canvas = self.obs_canvases[group_name]
                if isinstance(canvas, QWidget):
                    if hasattr(canvas, 'setVisible'):
                        canvas.setVisible(True)
                    elif hasattr(canvas, 'show'):
                        canvas.show()
        
        # 페이지 정보 업데이트
        total_pages = (len(group_names) + self.graph_items_per_page - 1) // self.graph_items_per_page
        self.graph_page_label.setText(f"페이지: {self.graph_current_page + 1}/{total_pages}")
        
        # 버튼 상태 업데이트
        self.graph_prev_btn.setEnabled(self.graph_current_page > 0)
        self.graph_next_btn.setEnabled(self.graph_current_page < total_pages - 1)

    # =========================
    # BUTTON CALLBACKS
    # =========================
    def open_file_dialog(self):
        """Policy 파일 열기 다이얼로그"""
        # 기본 경로를 asset_vanilla로 설정
        # __file__은 gui_node.py의 경로이므로, src 디렉토리로 올라가서 asset_vanilla 찾기
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        asset_vanilla_path = os.path.join(src_dir, 'asset_vanilla')
        
        # asset_vanilla가 없으면 상위 디렉토리에서 찾기
        if not os.path.exists(asset_vanilla_path):
            # workspace root에서 찾기
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            asset_vanilla_path = os.path.join(workspace_root, 'src', 'asset_vanilla')
        
        default_path = asset_vanilla_path if os.path.exists(asset_vanilla_path) else ""
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Policy 파일 선택",
            default_path,
            "PyTorch Files (*.pt);;ONNX Files (*.onnx);;All Files (*)"
        )
        
        if file_path:
            self.current_file_path = file_path
            file_name = os.path.basename(file_path)
            self.current_file_label.setText(f"Policy 파일:\n{file_name}")
            self.current_file_label.setStyleSheet(
                "padding: 10px; background-color: #50C878; color: white; "
                "font-weight: bold; font-size: 11pt; border-radius: 5px;"
            )
            self.get_logger().info(f"Policy file selected: {file_path}")
            
            # Control node에 모델 파일 경로를 실시간으로 전달
            self.ros2_node.publish_model_path(file_path)
            self.get_logger().info(f"Model path published to control node: {file_path}")
            
            # 모델 파일 구조 확인 (로컬에서도 확인 가능하도록)
            # 이 함수 내부에서 A-RMA 모델 여부도 확인함
            self.check_model_file_structure(file_path)
        else:
            # 파일 선택 취소 시
            self.current_file_path = None
            self.current_file_label.setText("현재 선택된 파일이 없습니다.")
            self.current_file_label.setStyleSheet(
                "padding: 10px; background-color: #ff4444; color: white; "
                "font-weight: bold; font-size: 11pt; border-radius: 5px;"
            )
    
    def open_arma_file_dialog(self):
        """A-RMA 파일 열기 다이얼로그"""
        # 기본 경로를 control node의 모델 디렉토리로 설정
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'redshow_control', 'redshow_control'
        )
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "A-RMA Adaptation Module 파일 선택",
            default_path if os.path.exists(default_path) else "",
            "PyTorch Files (*.pt);;All Files (*)"
        )
        
        if file_path:
            self.arma_file_path = file_path
            file_name = os.path.basename(file_path)
            self.arma_file_label.setText(f"A-RMA 파일:\n{file_name}")
            self.arma_file_label.setStyleSheet(
                "padding: 10px; background-color: #50C878; color: white; "
                "font-weight: bold; font-size: 11pt; border-radius: 5px;"
            )
            self.arma_status_label.setText("A-RMA: ON")
            self.arma_status_label.setStyleSheet(
                "padding: 10px; background-color: #4A90E2; color: white; "
                "font-weight: bold; font-size: 12pt; border-radius: 5px;"
            )
            self.get_logger().info(f"A-RMA file selected: {file_path}")
            
            # Control node에 A-RMA 파일 경로 전달
            # TODO: A-RMA 파일 경로를 전달하는 토픽 추가 필요
        else:
            # 파일 선택 취소 시
            self.arma_file_path = None
            self.arma_file_label.setText("A-RMA 파일이 선택되지 않았습니다.")
            self.arma_file_label.setStyleSheet(
                "padding: 10px; background-color: #666; color: white; "
                "font-weight: bold; font-size: 11pt; border-radius: 5px;"
            )
            self.arma_status_label.setText("A-RMA: OFF")
            self.arma_status_label.setStyleSheet(
                "padding: 10px; background-color: #666; color: white; "
                "font-weight: bold; font-size: 12pt; border-radius: 5px;"
            )

    def on_manual_mode(self):
        """Manual 모드 선택"""
        if self.selected_mode_btn is not None and self.selected_mode_btn != self.manual_mode_btn:
            self.selected_mode_btn.setStyleSheet("")
            if self.is_running:
                self.is_running = False
                self.manual_timer.stop()
                self.ros2_node.publish_cmd(f"{self.current_mode}::STOP")
                zero_actions = [0.0] * 6
                self.ros2_node.publish_joint_cmd(zero_actions)
        
        self.current_mode = "MANUAL"
        self.selected_mode_btn = self.manual_mode_btn
        self.manual_mode_btn.setStyleSheet(
            "background-color: #4A90E2; color: white; font-weight: bold; font-size: 12pt;"
        )
        self.auto_mode_btn.setStyleSheet("")
        self.ros2_node.publish_cmd("MANUAL::STOP")
        
        if self.is_running:
            self.is_running = False
            self.manual_timer.stop()
            zero_actions = [0.0] * 6
            self.ros2_node.publish_joint_cmd(zero_actions)
        
        self.update_run_button_style()
        self.tab_widget.setCurrentIndex(0)  # Manual 탭으로 전환

    def on_auto_mode(self):
        """Auto 모드 선택"""
        if self.selected_mode_btn is not None and self.selected_mode_btn != self.auto_mode_btn:
            self.selected_mode_btn.setStyleSheet("")
            if self.is_running:
                self.is_running = False
                self.manual_timer.stop()
                self.ros2_node.publish_cmd(f"{self.current_mode}::STOP")
                zero_actions = [0.0] * 6
                self.ros2_node.publish_joint_cmd(zero_actions)
        
        self.current_mode = "AUTO"
        self.selected_mode_btn = self.auto_mode_btn
        self.auto_mode_btn.setStyleSheet(
            "background-color: #4A90E2; color: white; font-weight: bold; font-size: 12pt;"
        )
        self.manual_mode_btn.setStyleSheet("")
        self.ros2_node.publish_cmd("AUTO::STOP")
        
        if self.is_running:
            self.is_running = False
            self.manual_timer.stop()
            zero_actions = [0.0] * 6
            self.ros2_node.publish_joint_cmd(zero_actions)
        
        self.update_run_button_style()
        self.tab_widget.setCurrentIndex(1)  # Auto 탭으로 전환

    def on_run(self):
        """RUN/STOP 버튼"""
        if self.current_mode is None:
            self.show_warning("경고: 모드를 선택해주세요 (Manual 또는 Auto)")
            return
        
        # RUN 버튼을 누를 때 검증
        if not self.is_running:  # RUN 시작 시
            # 1. Policy 파일이 선택되었는지 확인
            if not self.current_file_path:
                self.show_warning("경고: Policy 파일을 선택해주세요.")
                return
            
            # 2. A-RMA 모델인데 A-RMA 파일이 선택되지 않았으면 경고하고 STOP
            if self.is_arma_model and not self.arma_file_path:
                self.show_warning("경고: A-RMA 모델이 선택되었지만 A-RMA 파일이 선택되지 않았습니다.\nA-RMA 파일을 선택해주세요.")
                return
            
            # 3. Observation 데이터가 모두 들어왔는지 엄격하게 확인
            if not hasattr(self, 'obs_index_status') or not self.obs_index_status:
                self.show_warning("경고: Observation 상태가 초기화되지 않았습니다.\nPT 파일을 다시 선택해주세요.")
                return
            
            if not self.is_arma_model:
                # 바닐라 모델: num_actor_obs 개 데이터가 모두 ready 상태여야 함
                # base_lin_vel이 제거되었을 수 있으므로 실제 계산된 num_actor_obs 사용
                required_count = self.num_actor_obs if hasattr(self, 'num_actor_obs') and self.num_actor_obs > 0 else 23
                required_indices = list(range(required_count))  # 0 ~ (required_count-1)
                missing_indices = []
                for idx in required_indices:
                    if idx not in self.obs_index_status or not self.obs_index_status[idx].get('ready', False):
                        missing_indices.append(idx)
                
                if missing_indices:
                    self.show_warning(
                        f"경고: 필수 Observation 데이터가 부족합니다.\n"
                        f"기대: {required_count}개, 부족한 인덱스: {missing_indices[:10]}{'...' if len(missing_indices) > 10 else ''}\n"
                        f"Control Node가 실행 중이고 데이터를 발행하는지 확인해주세요."
                    )
                    return
            else:
                # A-RMA 모델: 기본 num_actor_obs 개 + extrinsics_obs 8개가 모두 ready 상태여야 함
                # base_lin_vel이 제거되었을 수 있으므로 실제 계산된 num_actor_obs 사용
                basic_count = self.num_actor_obs if hasattr(self, 'num_actor_obs') and self.num_actor_obs > 0 else 23
                # extrinsics_obs를 제외한 기본 observation 개수 (extrinsics_obs는 별도 토픽으로 옴)
                if 'extrinsics_obs' in self.obs_groups:
                    extrinsics_dim = len(self.obs_groups['extrinsics_obs']['indices'])
                    basic_count = basic_count - extrinsics_dim
                
                basic_indices = list(range(basic_count))
                missing_basic = []
                for idx in basic_indices:
                    if idx not in self.obs_index_status or not self.obs_index_status[idx].get('ready', False):
                        missing_basic.append(idx)
                
                # extrinsics_obs 8개 확인 (23-30)
                missing_extrinsics = []
                if 'extrinsics_obs' in self.obs_groups:
                    extrinsics_indices = self.obs_groups['extrinsics_obs']['indices']
                    for idx in extrinsics_indices:
                        if idx not in self.obs_index_status or not self.obs_index_status[idx].get('ready', False):
                            missing_extrinsics.append(idx)
                else:
                    missing_extrinsics = list(range(23, 31))  # extrinsics_obs가 없으면 8개 모두 부족
                
                if missing_basic or missing_extrinsics:
                    error_msg = "경고: 필수 Observation 데이터가 부족합니다.\n"
                    if missing_basic:
                        error_msg += f"기본 데이터 부족: {len(missing_basic)}개 인덱스\n"
                    if missing_extrinsics:
                        error_msg += f"Extrinsics Obs 부족: {len(missing_extrinsics)}개 인덱스\n"
                    error_msg += "Control Node와 Adaptation Module이 실행 중인지 확인해주세요."
                    self.show_warning(error_msg)
                    return
        
        self.is_running = not self.is_running
        cmd = f"{self.current_mode}::{'RUN' if self.is_running else 'STOP'}"
        self.ros2_node.publish_cmd(cmd)
        
        # RUN이 성공적으로 시작되면 경고 메시지 숨기기
        if self.is_running:
            self.warning_label.hide()
        
        if self.current_mode == "MANUAL":
            if self.is_running:
                self.manual_timer.start(20)  # 50Hz로 전송
                self.send_manual_joint_cmd()
                QTimer.singleShot(100, self.send_manual_joint_cmd)
            else:
                self.manual_timer.stop()
                zero_actions = [0.0] * 6
                self.ros2_node.publish_joint_cmd(zero_actions)
        elif self.current_mode == "AUTO":
            # Auto 모드에서 RUN 시작 시 velocity_command 전송
            if self.is_running:
                if hasattr(self, 'auto_velocity_command_inputs'):
                    vcmd_vals = [s.value() for s in self.auto_velocity_command_inputs]
                    self.velocity_commands = vcmd_vals.copy()
                    self.ros2_node.publish_velocity_command(vcmd_vals)
        
        if not self.is_running:
            zero_actions = [0.0] * 6
            self.ros2_node.publish_joint_cmd(zero_actions)
        
        self.update_run_button_style()
    
    def on_record(self):
        """RECORD 버튼 클릭 핸들러"""
        self.is_recording = not self.is_recording
        cmd = "RECORD" if self.is_recording else "STOP_RECORD"
        self.ros2_node.publish_cmd(cmd)
        self.update_record_button_style()
    
    def update_record_button_style(self):
        """RECORD 버튼 스타일 업데이트"""
        if self.is_recording:
            self.record_btn.setText("STOP RECORD")
            self.record_btn.setStyleSheet(
                "background-color: #FF6B6B; color: white; font-weight: bold; font-size: 12pt;"
            )
        else:
            self.record_btn.setText("RECORD")
            self.record_btn.setStyleSheet(
                "background-color: #4A90E2; color: white; font-weight: bold; font-size: 12pt;"
            )
    
    def update_run_button_style(self):
        """RUN 버튼 스타일 업데이트"""
        if self.is_running:
            self.run_btn.setText("STOP")
            self.run_btn.setStyleSheet(
                "background-color: #50C878; color: white; font-weight: bold; font-size: 12pt;"
            )
        else:
            self.run_btn.setText("RUN")
            self.run_btn.setStyleSheet("")
    
    def on_tab_changed(self, index):
        """탭 변경 시 그래프 타이머 시작/중지"""
        if index == 2:  # Graph 탭
            # Graph 탭이 열릴 때 그래프 위젯 생성
            if self.obs_groups and len(self.obs_figs) == 0:
                QTimer.singleShot(200, self.update_graph_widgets)  # 200ms 지연 후 업데이트
            
            if not self.graph_timer.isActive():
                self.graph_timer.start(100)  # 10Hz
        else:
            if self.graph_timer.isActive():
                self.graph_timer.stop()

    # =========================
    # DATA FLOW
    # =========================
    def send_manual_joint_cmd(self):
        """Manual 명령 전송"""
        if self.current_mode == "MANUAL" and self.is_running:
            vals = [s.value() for s in self.manual_inputs]
            self.ros2_node.publish_joint_cmd(vals)
            self.act_buffer.append(vals.copy())
            
            # Velocity command도 전송 및 저장
            if hasattr(self, 'velocity_command_inputs'):
                vcmd_vals = [s.value() for s in self.velocity_command_inputs]
                self.velocity_commands = vcmd_vals.copy()
                self.ros2_node.publish_velocity_command(vcmd_vals)
    
    def on_velocity_command_changed(self):
        """Velocity command 변경 시 호출 (Auto 모드)"""
        if hasattr(self, 'auto_velocity_command_inputs'):
            vcmd_vals = [s.value() for s in self.auto_velocity_command_inputs]
            self.velocity_commands = vcmd_vals.copy()
            self.ros2_node.publish_velocity_command(vcmd_vals)

    def update_ui(self):
        """UI 업데이트 (50Hz)"""
        # Observation 그룹이 설정되지 않았으면 업데이트하지 않음
        if not self.obs_groups:
            return
        
        # 각 Observation 인덱스별 상태 업데이트
        for group_name, group_info in self.obs_groups.items():
            for i, idx in enumerate(group_info['indices']):
                name = group_info['names'][i]
                status = self.obs_index_status[idx]
                label = self.obs_status_labels[group_name][idx]
                
                # 현재 값 가져오기
                current_value = None
                if idx in self.obs_buffer and len(self.obs_buffer[idx]) > 0:
                    current_value = self.obs_buffer[idx][-1]
                
                if status['ready']:
                    hz = status['hz']
                    # 값 표시 형식: "name: value READY Hz" 또는 "name: READY Hz" (값이 없으면)
                    if current_value is not None:
                        if hz > 0:
                            label.setText(f"{name}: {current_value:.4f} READY {hz:.1f} Hz")
                        else:
                            label.setText(f"{name}: {current_value:.4f} READY")
                    else:
                        if hz > 0:
                            label.setText(f"{name}: READY {hz:.1f} Hz")
                        else:
                            label.setText(f"{name}: READY")
                    label.setStyleSheet(
                        "padding: 3px 5px; background-color: #50C878; color: white; "
                        "font-size: 9pt; border-radius: 3px;"
                    )
                else:
                    if current_value is not None:
                        label.setText(f"{name}: {current_value:.4f} None")
                    else:
                        label.setText(f"{name}: None")
                    label.setStyleSheet(
                        "padding: 3px 5px; background-color: #ff4444; color: white; "
                        "font-size: 9pt; border-radius: 3px;"
                    )
        
        # 전체 Observation Hz 업데이트
        if self.obs_ready:
            if self.obs_hz > 0:
                self.obs_hz_label.setText(f"Overall Observation Hz: {self.obs_hz:.1f} Hz")
            else:
                self.obs_hz_label.setText("Overall Observation Hz: 계산 중...")
        else:
            self.obs_hz_label.setText("Overall Observation Hz: 0.0 Hz")
        
        # Policy Hz 업데이트
        self.policy_hz_label.setText(f"Policy Hz: {self.policy_hz:.1f} Hz")

    def update_graphs(self):
        """그래프 업데이트 (10Hz) - Graph 탭이 활성화되어 있을 때만"""
        if self.tab_widget.currentIndex() != 2:  # Graph 탭이 아니면 업데이트 안 함
            return
        
        if len(self.time_buffer) == 0:
            return
        
        # Observation 그룹이 설정되지 않았으면 업데이트하지 않음
        if not self.obs_groups:
            return
        
        times = np.array(list(self.time_buffer))
        
        for group_name, group_info in self.obs_groups.items():
            # 그래프가 생성되지 않았으면 건너뜀
            if group_name not in self.obs_axes:
                continue
                
            ax = self.obs_axes[group_name]
            lines = self.obs_lines[group_name]
            
            data_exists = False
            for i, idx in enumerate(group_info['indices']):
                if idx in self.obs_buffer and len(self.obs_buffer[idx]) > 0:
                    values = np.array(list(self.obs_buffer[idx]))
                    min_len = min(len(times), len(values))
                    if min_len > 0:
                        # 데이터가 모두 0이 아닌지 확인
                        if not np.allclose(values[:min_len], 0.0, atol=1e-6):
                            lines[i].set_data(times[:min_len], values[:min_len])
                            data_exists = True
                        else:
                            # 데이터가 모두 0이면 빈 라인으로 표시
                            lines[i].set_data([], [])
            
            if data_exists:
                ax.relim()
                ax.autoscale_view()
                self.obs_canvases[group_name].draw()
            else:
                # 데이터가 없으면 그래프를 초기화 상태로 유지
                ax.relim()
                ax.autoscale_view()
                self.obs_canvases[group_name].draw()

    def check_model_file_structure(self, model_path: str):
        """모델 파일 구조 확인 및 env.yaml, agent.yaml에서 Observation 정보 추출
        .pt와 .onnx 파일 모두 지원
        """
        try:
            model_dir = os.path.dirname(model_path)
            input_dim = None
            
            # 1. 파일 확장자에 따라 input_dim 추출
            if model_path.endswith('.pt'):
                # PT 파일에서 actor.0.weight shape 확인하여 input_dim 추출
                ckpt = torch.load(model_path, map_location="cpu")
                actor_weight = ckpt.get("actor.0.weight", None)
                if actor_weight is not None:
                    input_dim = actor_weight.shape[1]  # (hidden_dim, input_dim)
                    self.get_logger().info(f"[GUI CHECKPOINT] Actor input dimension: {input_dim}")
                else:
                    self.get_logger().warn("[GUI CHECKPOINT] Could not find actor.0.weight in checkpoint")
                
                # PT 파일 구조 확인 (디버깅용)
                all_keys = list(ckpt.keys())
                self.get_logger().info(f"[GUI CHECKPOINT] Checkpoint keys ({len(all_keys)}): {all_keys[:10]}...")
            elif model_path.endswith('.onnx'):
                # ONNX 파일의 경우 input_dim을 직접 추출하기 어려우므로
                # env.yaml에서 observation 차원을 합산하여 계산
                self.get_logger().info(f"[GUI CHECKPOINT] ONNX file detected, will infer input_dim from env.yaml")
            else:
                self.get_logger().warn(f"[GUI CHECKPOINT] Unsupported file format: {model_path}")
                return
            
            # 2. 같은 폴더에서 env.yaml, agent.yaml 찾기
            # 먼저 모델 파일과 같은 디렉토리에서 찾기
            env_yaml_path = os.path.join(model_dir, "env.yaml")
            agent_yaml_path = os.path.join(model_dir, "agent.yaml")
            
            # 같은 디렉토리에 없으면 /mnt/data/env.yaml도 확인
            if not os.path.exists(env_yaml_path):
                alt_env_yaml_path = "/mnt/data/env.yaml"
                if os.path.exists(alt_env_yaml_path):
                    env_yaml_path = alt_env_yaml_path
                    self.get_logger().info(f"[GUI CHECKPOINT] Using env.yaml from /mnt/data: {alt_env_yaml_path}")
            
            obs_config = None
            
            # env.yaml 읽기
            if os.path.exists(env_yaml_path):
                self.get_logger().info(f"[GUI CHECKPOINT] Found env.yaml: {env_yaml_path}")
                try:
                    # observations.policy 섹션만 추출하여 파싱 (Python tuple/slice 문제 회피)
                    policy_obs = self.extract_policy_observations(env_yaml_path)
                    
                    if policy_obs:
                        self.get_logger().info(f"[GUI CHECKPOINT] Policy observations: {list(policy_obs.keys())}")
                        
                        # Observation Group 구성
                        obs_config = self.build_obs_groups_from_yaml(policy_obs, input_dim)
                        
                        if obs_config:
                            self.obs_groups = obs_config['obs_groups']
                            self.num_actor_obs = obs_config.get('num_actor_obs', input_dim or 23)
                            self.get_logger().info(f"[GUI CHECKPOINT] ✓ Observation groups loaded: {list(self.obs_groups.keys())}")
                            
                            # A-RMA 모델인지 확인 (extrinsics_obs가 있는지 확인)
                            if 'extrinsics_obs' in self.obs_groups:
                                self.is_arma_model = True
                                self.get_logger().info("[GUI CHECKPOINT] This is an A-RMA model (extrinsics_obs found)")
                            else:
                                self.is_arma_model = False
                                self.get_logger().info("[GUI CHECKPOINT] This is a Vanilla model (no extrinsics_obs)")
                            
                            # UI 업데이트
                            self.update_obs_ui()
                except Exception as e:
                    self.get_logger().error(f"[GUI CHECKPOINT] Failed to parse env.yaml: {e}")
            else:
                self.get_logger().warn(f"[GUI CHECKPOINT] env.yaml not found in {model_dir} or /mnt/data")
            
            # agent.yaml 읽기 (추가 정보가 있을 수 있음)
            if os.path.exists(agent_yaml_path):
                self.get_logger().info(f"[GUI CHECKPOINT] Found agent.yaml: {agent_yaml_path}")
                try:
                    with open(agent_yaml_path, 'r') as f:
                        agent_config = yaml.safe_load(f)
                    # 필요시 agent.yaml에서 추가 정보 추출 가능
                except Exception as e:
                    self.get_logger().warn(f"[GUI CHECKPOINT] Failed to parse agent.yaml: {e}")
            
        except Exception as e:
            self.get_logger().error(f"[GUI CHECKPOINT] Failed to inspect model file: {e}")
    
    
    def extract_policy_observations(self, yaml_path: str):
        """env.yaml에서 observations.policy 섹션만 추출 (Python tuple/slice 문제 회피)"""
        try:
            with open(yaml_path, 'r') as f:
                lines = f.readlines()
            
            # observations.policy 섹션 찾기
            in_observations = False
            in_policy = False
            observations_indent = None
            policy_indent = None
            policy_obs = {}
            current_obs_name = None
            
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                
                # observations: 시작
                if stripped.startswith('observations:'):
                    in_observations = True
                    observations_indent = current_indent
                    continue
                
                # policy: 시작
                if in_observations and stripped.startswith('policy:'):
                    in_policy = True
                    policy_indent = current_indent
                    continue
                
                # policy 섹션 내부의 observation 항목들 추출
                if in_policy:
                    # policy 섹션이 끝났는지 확인 (다른 섹션 시작: student_policy, teacher_policy, critic 등)
                    if stripped and current_indent <= policy_indent:
                        # policy와 같은 레벨의 다른 섹션이 시작되면 policy 섹션 종료
                        if current_indent == policy_indent and ':' in stripped:
                            break
                        # observations 레벨로 돌아가면 종료
                        if current_indent <= observations_indent:
                            break
                    
                    # observation 항목 이름 추출 (policy_indent보다 한 단계 더 들여쓰기된 것들)
                    if current_indent == policy_indent + 2 and ':' in stripped:
                        key = stripped.split(':')[0].strip()
                        # 메타데이터 키가 아닌 실제 observation 이름만
                        # base_lin_vel은 제거 (사용하지 않음)
                        if key not in ['concatenate_terms', 'enable_corruption', 'history_length', 
                                      'flatten_history_dim', 'func', 'params', 'modifiers', 
                                      'noise', 'clip', 'scale', 'base_lin_vel']:
                            # 중복 체크: 이미 추가된 observation은 건너뜀
                            if key and key not in policy_obs:
                                policy_obs[key] = {}
                                current_obs_name = key
                                self.get_logger().info(f"[GUI CHECKPOINT] Found observation: {key}")
            
            if not policy_obs:
                self.get_logger().warn("[GUI CHECKPOINT] Could not find any observations in policy section")
                return None
            
            return policy_obs
                
        except Exception as e:
            self.get_logger().error(f"[GUI CHECKPOINT] Failed to extract policy observations: {e}")
            return None
    
    def build_obs_groups_from_yaml(self, policy_obs: dict, total_dim: int = None):
        """env.yaml의 policy observations에서 Observation Group 구성
        - base_lin_vel은 제거
        - extrinsics_obs는 마지막 인덱스로 이동
        - 기본 23개 observation은 앞에서부터 순차적으로 매핑
        - extrinsics_obs는 마지막 8개 인덱스에 매핑 (A-RMA 모델인 경우)
        """
        # 각 Observation 항목의 기본 차원 정의
        DEFAULT_DIMS = {
            'extrinsics_obs': 8,  # A-RMA 모델의 경우 마지막에 추가
            'leg_position': 4,
            'wheel_velocity': 2,
            'base_ang_vel': 3,
            'velocity_commands': 4,
            'base_quat': 4,
            'actions': 6,
        }
        
        # base_lin_vel 제거 및 extrinsics_obs 분리
        filtered_obs = {}
        extrinsics_obs_config = None
        
        for obs_name, obs_config in policy_obs.items():
            if obs_name == 'base_lin_vel':
                continue  # base_lin_vel 제거
            elif obs_name == 'extrinsics_obs':
                extrinsics_obs_config = (obs_name, obs_config)  # 나중에 처리
            else:
                filtered_obs[obs_name] = obs_config
        
        obs_groups = {}
        current_idx = 0
        unknown_dims = []
        
        # Control Node가 보내는 기본 23개 데이터를 채울 시작 인덱스 (앞에서부터 순차적으로)
        control_data_idx = 0
        
        # 기본 observation들을 먼저 처리 (extrinsics_obs 제외)
        for obs_name, obs_config in filtered_obs.items():
            dim = None
            
            if isinstance(obs_config, dict):
                # 차원 정보가 직접 있는 경우
                dim = obs_config.get('dim', None)
                if dim is None:
                    # shape 정보가 있는 경우
                    shape = obs_config.get('shape', None)
                    if shape:
                        if isinstance(shape, list):
                            dim = int(np.prod(shape))
                        else:
                            dim = int(shape)
            
            # 차원 정보가 없으면 기본값 사용
            if dim is None:
                dim = DEFAULT_DIMS.get(obs_name, None)
                if dim is None:
                    # 차원을 모르는 경우 unknown_dims에 추가
                    unknown_dims.append(obs_name)
                    self.get_logger().info(
                        f"[GUI CHECKPOINT] Unknown dimension for {obs_name}, will infer from remaining dimensions"
                    )
                    continue
                elif dim == 0:
                    # 차원이 0인 경우도 unknown_dims에 추가 (나중에 역추정)
                    unknown_dims.append(obs_name)
                    self.get_logger().info(
                        f"[GUI CHECKPOINT] {obs_name} has dim=0, will infer from remaining dimensions"
                    )
                    continue
            
            dim = int(dim)
            
            # 인덱스 범위 생성
            indices = list(range(current_idx, current_idx + dim))
            names = [f"{obs_name}[{i}]" for i in range(dim)]
            
            # Control Node 데이터 매핑 정보 추가
            # Control Node가 보내는 데이터는 앞에서부터 순차적으로 채움 (어떤 observation이든 상관없이)
            control_start_idx = control_data_idx
            control_data_idx += dim
            
            obs_groups[obs_name] = {
                'names': names,
                'indices': indices,
                'control_start_idx': control_start_idx  # Control Node에서 오는 데이터의 시작 인덱스 (없으면 None)
            }
            
            current_idx += dim
            
            self.get_logger().info(
                f"[GUI CHECKPOINT] {obs_name}: dim={dim}, indices={indices[0]}-{indices[-1]}, "
                f"control_data_idx={control_start_idx}-{control_start_idx+dim-1}"
            )
        
        # extrinsics_obs를 마지막에 추가 (A-RMA 모델인 경우)
        if extrinsics_obs_config:
            obs_name, obs_config = extrinsics_obs_config
            dim = None
            
            if isinstance(obs_config, dict):
                dim = obs_config.get('dim', None)
                if dim is None:
                    shape = obs_config.get('shape', None)
                    if shape:
                        if isinstance(shape, list):
                            dim = int(np.prod(shape))
                        else:
                            dim = int(shape)
            
            if dim is None:
                dim = DEFAULT_DIMS.get(obs_name, 8)
            
            dim = int(dim)
            
            # 마지막 인덱스에 추가
            indices = list(range(current_idx, current_idx + dim))
            names = [f"{obs_name}[{i}]" for i in range(dim)]
            
            # extrinsics_obs는 Control Node 데이터의 마지막 8개에 매핑
            # 기본 23개가 먼저 채워지고, 그 다음에 extrinsics_obs 8개가 옴
            control_start_idx = control_data_idx  # 기본 23개 이후부터 시작
            
            obs_groups[obs_name] = {
                'names': names,
                'indices': indices,
                'control_start_idx': control_start_idx
            }
            
            current_idx += dim
            control_data_idx += dim
            
            self.get_logger().info(
                f"[GUI CHECKPOINT] {obs_name}: dim={dim}, indices={indices[0]}-{indices[-1]}, "
                f"control_data_idx={control_start_idx}-{control_start_idx+dim-1} (마지막에 추가됨)"
            )
        
        # 알 수 없는 차원이 있고 total_dim이 있으면 역추정
        if unknown_dims and total_dim is not None:
            remaining_dim = total_dim - current_idx
            if remaining_dim > 0:
                self.get_logger().info(
                    f"[GUI CHECKPOINT] Inferring dimensions for {unknown_dims}: "
                    f"remaining {remaining_dim} dimensions"
                )
                # 남은 차원을 알 수 없는 항목들에 분배
                if len(unknown_dims) == 1:
                    # 하나만 있으면 모두 할당
                    obs_name = unknown_dims[0]
                    dim = remaining_dim
                    indices = list(range(current_idx, current_idx + dim))
                    names = [f"{obs_name}[{j}]" for j in range(dim)]
                    
                    obs_groups[obs_name] = {
                        'names': names,
                        'indices': indices
                    }
                    
                    current_idx += dim
                    self.get_logger().info(
                        f"[GUI CHECKPOINT] {obs_name}: inferred dim={dim}, "
                        f"indices={indices[0]}-{indices[-1]}"
                    )
                else:
                    # 여러 개면 균등 분배
                    dim_per_item = remaining_dim // len(unknown_dims)
                    for i, obs_name in enumerate(unknown_dims):
                        if i == len(unknown_dims) - 1:
                            # 마지막 항목이 나머지 모두 가져감
                            dim = remaining_dim - (dim_per_item * i)
                        else:
                            dim = dim_per_item
                        
                        indices = list(range(current_idx, current_idx + dim))
                        names = [f"{obs_name}[{j}]" for j in range(dim)]
                        
                        obs_groups[obs_name] = {
                            'names': names,
                            'indices': indices
                        }
                        
                        current_idx += dim
                        self.get_logger().info(
                            f"[GUI CHECKPOINT] {obs_name}: inferred dim={dim}, "
                            f"indices={indices[0]}-{indices[-1]}"
                        )
        
        # 총 차원 확인
        if total_dim is not None:
            if current_idx != total_dim:
                self.get_logger().warn(
                    f"[GUI CHECKPOINT] Dimension mismatch: "
                    f"calculated {current_idx} vs checkpoint {total_dim}"
                )
            else:
                self.get_logger().info(
                    f"[GUI CHECKPOINT] ✓ Total dimensions match: {current_idx}"
                )
        
        if not obs_groups:
            return None
        
        return {
            'obs_groups': obs_groups,
            'num_actor_obs': current_idx if total_dim is None else total_dim
        }
    
    def update_obs_ui(self):
        """Observation Group 정보가 업데이트되면 UI 재구성"""
        # Observation Status 영역 재구성
        if hasattr(self, 'obs_status_container'):
            # 기존 위젯 제거
            for i in reversed(range(self.obs_status_layout.count())):
                item = self.obs_status_layout.itemAt(i)
                if item.widget():
                    item.widget().deleteLater()
            
            # 새로운 Observation Group 표시
            self.obs_status_labels = {}
            for group_name, group_info in self.obs_groups.items():
                group_label = QLabel(f"<b>{group_name.upper()}</b>")
                group_label.setStyleSheet("font-size: 11pt; font-weight: bold; padding: 5px;")
                self.obs_status_layout.addWidget(group_label)
                
                self.obs_status_labels[group_name] = {}
                for i, idx in enumerate(group_info['indices']):
                    name = group_info['names'][i]
                    status_label = QLabel(f"{name}: None")
                    status_label.setStyleSheet(
                        "padding: 3px 5px; background-color: #ff4444; color: white; "
                        "font-size: 9pt; border-radius: 3px;"
                    )
                    self.obs_status_layout.addWidget(status_label)
                    self.obs_status_labels[group_name][idx] = status_label
            
            # Observation 인덱스 상태 초기화
            self.obs_index_status = {}
            self.obs_index_hz_buffers = {}
            for group_name, group_info in self.obs_groups.items():
                for idx in group_info['indices']:
                    self.obs_index_status[idx] = {'ready': False, 'last_time': None, 'hz': 0.0}
                    self.obs_index_hz_buffers[idx] = deque(maxlen=50)
                    if idx not in self.obs_buffer:
                        self.obs_buffer[idx] = deque(maxlen=self.max_history)
        
        # Graph 탭은 나중에 탭이 열릴 때 업데이트 (지금은 건너뜀)
        # QTimer.singleShot(100, self.update_graph_widgets)
        
        self.get_logger().info("[GUI] Observation UI updated")
    
    def update_graph_widgets(self):
        """Graph 탭의 그래프 위젯 생성/업데이트"""
        try:
            if not self.obs_groups:
                return
            
            # tab_widget이 아직 생성되지 않았으면 건너뜀
            if not hasattr(self, 'tab_widget') or self.tab_widget is None:
                return
            
            # Graph 탭의 content_widget 찾기
            if self.tab_widget.count() < 3:
                return
            
            graph_tab = self.tab_widget.widget(2)  # Graph 탭은 인덱스 2
            if graph_tab is None:
                return
            
            # graph_tab은 이제 QWidget (메인 위젯)
            # 내부에 QScrollArea가 있음
            if not isinstance(graph_tab, QWidget):
                return
            
            # ScrollArea 찾기
            scroll_area = None
            for child in graph_tab.findChildren(QScrollArea):
                scroll_area = child
                break
            
            if scroll_area is None:
                return
            
            content_widget = scroll_area.widget()
            if content_widget is None:
                return
            
            layout = content_widget.layout()
            if layout is None:
                return
            
            # graph_layout이 없으면 설정
            if not hasattr(self, 'graph_layout'):
                self.graph_layout = layout
            
            # 기존 그래프 위젯 제거
            for group_name in list(self.obs_figs.keys()):
                if group_name in self.obs_canvases:
                    canvas = self.obs_canvases[group_name]
                    if canvas:
                        canvas.setParent(None)
                        layout.removeWidget(canvas)
                        canvas.deleteLater()
                if group_name in self.obs_figs:
                    del self.obs_figs[group_name]
                if group_name in self.obs_canvases:
                    del self.obs_canvases[group_name]
                if group_name in self.obs_axes:
                    del self.obs_axes[group_name]
                if group_name in self.obs_lines:
                    del self.obs_lines[group_name]
            
            # 기존 placeholder 제거
            widgets_to_remove = []
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    if isinstance(widget, QLabel) and "PT 파일을 선택하면" in widget.text():
                        widgets_to_remove.append((widget, item))
            
            for widget, item in widgets_to_remove:
                try:
                    layout.removeItem(item)
                    widget.setParent(None)
                    widget.deleteLater()
                except Exception as e:
                    self.get_logger().warn(f"[GUI] Error removing placeholder: {e}")
            
            # Qt 백엔드가 아닌 경우 그래프 생성 건너뛰기
            if not USE_QT_BACKEND:
                self.get_logger().warn(
                    "[GUI] Qt backend not available. Graphs will not be displayed. "
                    "Please upgrade matplotlib (pip install --upgrade matplotlib>=3.6.0) "
                    "or install PyQt5 (pip install PyQt5) to enable graphs."
                )
                return
            
            # 각 observation group에 대해 그래프 생성
            for group_name, group_info in self.obs_groups.items():
                try:
                    # Figure 생성
                    fig = Figure(figsize=(10, 4))
                    canvas = FigureCanvas(fig)
                    ax = fig.add_subplot(111)
                    
                    # 라인 생성
                    lines = []
                    colors = plt.cm.tab10(np.linspace(0, 1, len(group_info['indices'])))
                    for i, idx in enumerate(group_info['indices']):
                        line, = ax.plot([], [], label=group_info['names'][i], color=colors[i])
                        lines.append(line)
                    
                    ax.set_title(f"{group_name.upper()}", fontsize=12, fontweight='bold')
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Value")
                    ax.legend(loc='upper right', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # 저장
                    self.obs_figs[group_name] = fig
                    self.obs_canvases[group_name] = canvas
                    self.obs_axes[group_name] = ax
                    self.obs_lines[group_name] = lines
                    
                    # 레이아웃에 추가 (Qt 위젯인 경우만)
                    if isinstance(canvas, QWidget):
                        layout.addWidget(canvas)
                        # 초기에는 숨김 (페이지별로 표시)
                        if hasattr(canvas, 'setVisible'):
                            canvas.setVisible(False)
                        elif hasattr(canvas, 'hide'):
                            canvas.hide()
                    else:
                        self.get_logger().warn(f"[GUI] Canvas for {group_name} is not a Qt widget, skipping")
                        continue
                except Exception as e:
                    self.get_logger().error(f"[GUI] Error creating graph for {group_name}: {e}")
                    continue
            
            layout.addStretch()
            
            # 그래프 표시 업데이트
            self.update_graph_display()
            
        except Exception as e:
            self.get_logger().error(f"[GUI] Error updating graph widgets: {e}")
            import traceback
            self.get_logger().error(f"[GUI] Traceback: {traceback.format_exc()}")
    
    def get_logger(self):
        """로거 반환"""
        return self.ros2_node.get_logger()

    def closeEvent(self, event):
        """종료 시 정리"""
        # 모든 타이머 중지
        if self.is_running:
            if self.current_mode == "MANUAL":
                self.manual_timer.stop()
            self.ros2_node.publish_cmd(f"{self.current_mode}::STOP")
        
        self.ui_timer.stop()
        self.hz_timer.stop()
        if self.graph_timer.isActive():
            self.graph_timer.stop()
        
        # 모든 조인트를 0으로 설정
        zero_actions = [0.0] * 6
        self.ros2_node.publish_joint_cmd(zero_actions)
        
        # ROS2 정리
        try:
            self.ros2_node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        event.accept()


# =========================
# ENTRY
# =========================
def main():
    app = QApplication(sys.argv)
    w = MonitorGUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
