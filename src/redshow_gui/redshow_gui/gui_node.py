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

# 먼저 matplotlib.use()를 호출
matplotlib.use('Qt5Agg')

# backend_qtagg의 호환성 문제를 우회하기 위해 
# 직접 backend_qt5agg를 import 시도
try:
    # matplotlib 3.5.1에서 PySide6 사용 시 발생하는 문제를 우회
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except (ImportError, TypeError, AttributeError) as e:
    # Qt5Agg 실패 시 TkAgg 사용 (비상용)
    print(f"Warning: Qt5Agg backend failed ({e}), trying TkAgg...")
    matplotlib.use('TkAgg')
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
        print("Using TkAgg backend as fallback")
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
        self.model_path_pub = self.create_publisher(String, 'redshow/model_path', 10)
        self.feedback_sub = None
        self.policy_hz_sub = None
        self.shutdown_sub = None
        self.obs_config_sub = None
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
    
    def publish_model_path(self, model_path):
        msg = String()
        msg.data = model_path
        self.model_path_pub.publish(msg)
    
    def setup_feedback_subscriber(self, callback):
        self.feedback_sub = self.create_subscription(
            Float64MultiArray, 'redshow/feedback', callback, 10
        )
    
    def setup_policy_hz_subscriber(self, callback):
        self.policy_hz_sub = self.create_subscription(
            Float64MultiArray, 'redshow/policy_hz', callback, 10
        )
    
    def setup_shutdown_subscriber(self, callback):
        self.shutdown_sub = self.create_subscription(
            String, 'redshow/shutdown', callback, 10
        )

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
        self.ros2_node.setup_feedback_subscriber(self.feedback_callback)
        self.ros2_node.setup_policy_hz_subscriber(self.policy_hz_callback)
        self.ros2_node.setup_shutdown_subscriber(self.shutdown_callback)
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
    
    def feedback_callback(self, msg):
        """피드백 데이터 콜백"""
        if len(msg.data) != 23:
            self.get_logger().warn(f"[FEEDBACK] Invalid data length: {len(msg.data)} (expected 23)")
            return
        
        current_time = time.time()
        
        # 각 인덱스별로 데이터 확인 및 상태 업데이트
        data_array = np.array(msg.data)
        any_data_ready = False
        
        for idx in range(23):
            value = msg.data[idx]
            # 데이터가 0이 아니거나 (센서 데이터), 또는 인덱스가 17-22 (prev_act)인 경우는 항상 업데이트
            is_valid = not np.isclose(value, 0.0, atol=1e-6) or (17 <= idx <= 22)
            
            if is_valid:
                # 이전 시간과의 간격으로 Hz 계산
                if self.obs_index_status[idx]['last_time'] is not None:
                    dt = current_time - self.obs_index_status[idx]['last_time']
                    if dt > 0 and dt < 1.0:
                        hz = 1.0 / dt
                        if 0 < hz < 1000:
                            self.obs_index_hz_buffers[idx].append(hz)
                            # Hz 평균 계산
                            if len(self.obs_index_hz_buffers[idx]) > 0:
                                self.obs_index_status[idx]['hz'] = np.mean(self.obs_index_hz_buffers[idx])
                
                self.obs_index_status[idx]['ready'] = True
                self.obs_index_status[idx]['last_time'] = current_time
                any_data_ready = True
                
                # 피드백 데이터를 obs_buffer에 추가
                if idx in self.obs_buffer:
                    self.obs_buffer[idx].append(value)
            else:
                # 데이터가 0이면 상태는 유지하되, 시간이 오래 지나면 None으로 변경
                if self.obs_index_status[idx]['last_time'] is not None:
                    elapsed = current_time - self.obs_index_status[idx]['last_time']
                    if elapsed > 1.0:  # 1초 이상 데이터가 없으면 None
                        self.obs_index_status[idx]['ready'] = False
                        self.obs_index_status[idx]['hz'] = 0.0
        
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
    
    def policy_hz_callback(self, msg):
        """Policy Hz 콜백"""
        if len(msg.data) > 0:
            self.policy_hz = msg.data[0]
    
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
        
        self.open_file_btn = QPushButton("파일 열기")
        self.open_file_btn.clicked.connect(self.open_file_dialog)
        self.open_file_btn.setMinimumHeight(40)
        file_layout.addWidget(self.open_file_btn)
        
        # 현재 선택된 파일 표시
        self.current_file_label = QLabel("현재 선택된 파일이 없습니다.")
        self.current_file_label.setWordWrap(True)
        self.current_file_label.setAlignment(Qt.AlignCenter)
        self.current_file_label.setMinimumHeight(50)
        self.current_file_label.setStyleSheet(
            "padding: 10px; background-color: #ff4444; color: white; "
            "font-weight: bold; font-size: 11pt; border-radius: 5px;"
        )
        file_layout.addWidget(self.current_file_label)
        
        file_group.setLayout(file_layout)
        l.addWidget(file_group)
        
        # Observation 상태 (각 인덱스별)
        obs_status_group = QGroupBox("Observation Status")
        obs_status_layout = QVBoxLayout()
        
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
        
        self.manual_mode_btn.setMinimumHeight(50)
        self.auto_mode_btn.setMinimumHeight(50)
        self.run_btn.setMinimumHeight(50)
        
        self.manual_mode_btn.clicked.connect(self.on_manual_mode)
        self.auto_mode_btn.clicked.connect(self.on_auto_mode)
        self.run_btn.clicked.connect(self.on_run)
        
        control_btn_layout.addWidget(self.manual_mode_btn)
        control_btn_layout.addWidget(self.auto_mode_btn)
        control_btn_layout.addWidget(self.run_btn)
        
        control_btn_group.setLayout(control_btn_layout)
        l.addWidget(control_btn_group)
        
        return w

    def make_manual_control_tab(self):
        """Manual Control 탭"""
        w = QWidget()
        l = QVBoxLayout(w)
        
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
        
        info_label = QLabel("Auto 모드에서는 Policy가 자동으로 Action을 생성합니다.")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 14pt; padding: 20px;")
        l.addWidget(info_label)
        
        l.addStretch()
        return w

    def make_graph_tab(self):
        """Graph 탭: Observation 그래프들"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        content_widget = QWidget()
        l = QVBoxLayout(content_widget)
        l.setSpacing(10)
        l.setContentsMargins(5, 5, 5, 5)
        
        # 각 그룹별로 그래프 생성 (obs_groups가 설정된 후에만 생성됨)
        self.obs_figs = {}
        self.obs_canvases = {}
        self.obs_axes = {}
        self.obs_lines = {}
        
        # 초기 메시지 표시
        graph_placeholder = QLabel("PT 파일을 선택하면 Observation 그래프가 표시됩니다.")
        graph_placeholder.setAlignment(Qt.AlignCenter)
        graph_placeholder.setStyleSheet("padding: 50px; color: #666; font-size: 12pt;")
        l.addWidget(graph_placeholder)
        
        # obs_groups가 설정되면 그래프 생성 (동적으로 업데이트됨)
        # 초기에는 빈 상태로 시작, PT 파일 로드 후 업데이트
        
        l.addStretch()
        
        # 컨텐츠 위젯 크기 설정
        content_widget.setMinimumWidth(800)
        num_groups = len(self.obs_groups)
        estimated_height = num_groups * (400 + 50)
        content_widget.setMinimumHeight(estimated_height)
        
        scroll.setWidget(content_widget)
        return scroll

    # =========================
    # BUTTON CALLBACKS
    # =========================
    def open_file_dialog(self):
        """파일 열기 다이얼로그"""
        # 기본 경로를 control node의 모델 디렉토리로 설정
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'redshow_control', 'redshow_control'
        )
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "모델 파일 선택",
            default_path if os.path.exists(default_path) else "",
            "PyTorch Files (*.pt);;ONNX Files (*.onnx);;All Files (*)"
        )
        
        if file_path:
            self.current_file_path = file_path
            file_name = os.path.basename(file_path)
            self.current_file_label.setText(f"현재 선택된 파일:\n{file_name}")
            self.current_file_label.setStyleSheet(
                "padding: 10px; background-color: #50C878; color: white; "
                "font-weight: bold; font-size: 11pt; border-radius: 5px;"
            )
            self.get_logger().info(f"Model file selected: {file_path}")
            
            # Control node에 모델 파일 경로를 실시간으로 전달
            self.ros2_node.publish_model_path(file_path)
            self.get_logger().info(f"Model path published to control node: {file_path}")
            
            # PT 파일 구조 확인 (로컬에서도 확인 가능하도록)
            self.check_pt_file_structure(file_path)
        else:
            # 파일 선택 취소 시
            self.current_file_path = None
            self.current_file_label.setText("현재 선택된 파일이 없습니다.")
            self.current_file_label.setStyleSheet(
                "padding: 10px; background-color: #ff4444; color: white; "
                "font-weight: bold; font-size: 11pt; border-radius: 5px;"
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
            return
        
        self.is_running = not self.is_running
        cmd = f"{self.current_mode}::{'RUN' if self.is_running else 'STOP'}"
        self.ros2_node.publish_cmd(cmd)
        
        if self.current_mode == "MANUAL":
            if self.is_running:
                self.manual_timer.start(20)  # 50Hz로 전송
                self.send_manual_joint_cmd()
                QTimer.singleShot(100, self.send_manual_joint_cmd)
            else:
                self.manual_timer.stop()
                zero_actions = [0.0] * 6
                self.ros2_node.publish_joint_cmd(zero_actions)
        
        if not self.is_running:
            zero_actions = [0.0] * 6
            self.ros2_node.publish_joint_cmd(zero_actions)
        
        self.update_run_button_style()
    
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
                
                if status['ready']:
                    hz = status['hz']
                    if hz > 0:
                        label.setText(f"{name}: READY {hz:.1f} Hz")
                        label.setStyleSheet(
                            "padding: 3px 5px; background-color: #50C878; color: white; "
                            "font-size: 9pt; border-radius: 3px;"
                        )
                    else:
                        label.setText(f"{name}: READY")
                        label.setStyleSheet(
                            "padding: 3px 5px; background-color: #50C878; color: white; "
                            "font-size: 9pt; border-radius: 3px;"
                        )
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

    def check_pt_file_structure(self, pt_path: str):
        """PT 파일 구조 확인 (GUI에서 직접 확인)"""
        if not pt_path.endswith('.pt'):
            return
        
        try:
            ckpt = torch.load(pt_path, map_location="cpu")
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"[GUI CHECKPOINT INSPECTION] Analyzing file: {pt_path}")
            self.get_logger().info("=" * 80)
            
            # 모든 최상위 키 출력
            all_keys = list(ckpt.keys())
            self.get_logger().info(f"[GUI CHECKPOINT] Top-level keys ({len(all_keys)}): {all_keys}")
            
            # Observation 관련 키 확인
            obs_keywords = ['obs', 'observation', 'config', 'cfg', 'group', 'space']
            found_obs_keys = []
            for key in all_keys:
                if any(kw in key.lower() for kw in obs_keywords):
                    found_obs_keys.append(key)
            
            if found_obs_keys:
                self.get_logger().info(f"[GUI CHECKPOINT] ⚠️  Found potential observation-related keys: {found_obs_keys}")
                for key in found_obs_keys:
                    value = ckpt[key]
                    if isinstance(value, dict):
                        self.get_logger().info(f"[GUI CHECKPOINT]   {key} content: {list(value.keys())}")
                        # obs_groups가 있는지 확인
                        if 'obs_groups' in value:
                            self.get_logger().info(f"[GUI CHECKPOINT]   ✓ Found obs_groups in {key}!")
                            # 여기서 obs_groups를 추출하여 GUI에 적용할 수 있음
                    else:
                        self.get_logger().info(f"[GUI CHECKPOINT]   {key} type: {type(value)}")
            else:
                self.get_logger().info(f"[GUI CHECKPOINT] ❌ No observation-related keys found")
            
            self.get_logger().info("=" * 80)
        except Exception as e:
            self.get_logger().error(f"[GUI CHECKPOINT] Failed to inspect PT file: {e}")
    
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
