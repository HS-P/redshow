#!/usr/bin/env python3
import os
import time
import threading
from collections import deque

import numpy as np
import torch
import serial

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String, Float64MultiArray
import json

from rsl_rl.modules import ActorCritic

# ONNX Runtime import (optional)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class BerkeleyControlNode(Node):
    def __init__(self):
        super().__init__('berkeley_control_node')

        # ────────────── ROS2 I/O ──────────────
        self.cmd_sub = self.create_subscription(String, 'redshow/cmd', self.cmd_callback, 10)
        self.manual_sub = self.create_subscription(Float64MultiArray, 'redshow/joint_cmd', self.manual_callback, 10)
        self.velocity_command_sub = self.create_subscription(Float64MultiArray, 'redshow/velocity_command', self.velocity_command_callback, 10)
        self.model_path_sub = self.create_subscription(String, 'redshow/model_path', self.model_path_callback, 10)
        
        # BNO085 IMU 개별 토픽 구독
        self.bno085_gyro_sub = self.create_subscription(Float64MultiArray, '/Redshow/Sensor/gyroscope', self.bno085_gyro_callback, 10)
        self.bno085_quat_sub = self.create_subscription(Float64MultiArray, '/Redshow/Sensor/quaternion', self.bno085_quat_callback, 10)
        self.get_logger().info("[INIT] Subscribed to /Redshow/Sensor/gyroscope and /Redshow/Sensor/quaternion for BNO085 IMU data")
        
        # EX_OBS 구독 (A-RMA Adaptation Module 출력)
        self.ex_obs_sub = self.create_subscription(Float64MultiArray, '/Redshow/Observation/ex_obs', self.ex_obs_callback, 10)
        self.get_logger().info("[INIT] Subscribed to /Redshow/Observation/ex_obs for A-RMA EX_OBS")
        
        self.policy_hz_pub = self.create_publisher(Float64MultiArray, 'redshow/policy_hz', 10)
        self.shutdown_pub = self.create_publisher(String, 'redshow/shutdown', 10)
        
        # ────────────── Observation별 개별 Publisher ──────────────
        self.leg_position_pub = self.create_publisher(Float64MultiArray, '/Redshow/Observation/leg_position', 10)
        self.wheel_velocity_pub = self.create_publisher(Float64MultiArray, '/Redshow/Observation/wheel_velocity', 10)
        self.base_ang_vel_pub = self.create_publisher(Float64MultiArray, '/Redshow/Observation/base_ang_vel', 10)
        self.velocity_commands_pub = self.create_publisher(Float64MultiArray, '/Redshow/Observation/velocity_commands', 10)
        self.base_quat_pub = self.create_publisher(Float64MultiArray, '/Redshow/Observation/base_quat', 10)
        self.base_rpy_pub = self.create_publisher(Float64MultiArray, '/Redshow/Observation/base_rpy', 10)
        self.actions_pub = self.create_publisher(Float64MultiArray, '/Redshow/Observation/actions', 10)

        # ────────────── Device ──────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ────────────── Policy load ──────────────
        self.declare_parameter('policy_name', 'model_4800.pt')
        policy_name = self.get_parameter('policy_name').get_parameter_value().string_value
        
        # ────────────── Action Scale (Sim2Real 보정) ──────────────
        # 학습 시 사용된 Action Scale (Policy Space) - IsaacLab Simulation에서 사용된 값
        self.ACTION_SCALE_POLICY = 1.0  # 학습 시 wheel action에 곱해진 스케일
        
        # 실제 적용할 Action Scale (Firmware Space)
        # 이 값이 학습 시와 다르면 Observation에서 보정 필요
        self.WHEEL_SCALE_DRIVE = 1.0  # Driving 모드 실제 적용 스케일

        # Policy-space ↔ Firmware-space 휠 방향 정합 (부호)
        # +1.0: 부호 유지, -1.0: 부호 반전
        # Manual/Auto/Observation/역변환 모두에 동일하게 적용되어 일관성 보장
        self.WHEEL_SIGN = -1.0
        
        # Max wheel speed (rad/s)
        self.MAX_WHEEL_SPEED = 30.0

        # asset_vanilla 경로 찾기 헬퍼 함수
        def find_asset_vanilla_path():
            """asset_vanilla 디렉토리 경로 찾기"""
            # 여러 가능한 경로 확인
            possible_paths = [
                # 현재 파일 기준 상대 경로
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'asset_vanilla'),
                # workspace root 기준
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src', 'asset_vanilla'),
                # 절대 경로 (라즈베리파이에서 사용)
                '/home/pi/ros2_ws/src/redshow/src/asset_vanilla',
                '/home/sol/redshow/src/asset_vanilla',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            return None
        
        asset_vanilla_path = find_asset_vanilla_path()
        
        # 경로 결정: 절대 경로 > 상대 경로 > asset_vanilla > 현재 디렉토리
        if os.path.isabs(policy_name):
            pt_path = policy_name
        elif os.path.exists(policy_name):
            pt_path = policy_name
        elif asset_vanilla_path:
            # asset_vanilla에서 찾기
            pt_path = os.path.join(asset_vanilla_path, policy_name)
            if not os.path.exists(pt_path):
                # asset_vanilla에 없으면 현재 디렉토리에서 찾기
                pt_path = os.path.join(os.path.dirname(__file__), policy_name)
        else:
            pt_path = os.path.join(os.path.dirname(__file__), policy_name)

        # Policy 초기화
        self.policy = ActorCritic(
            num_actor_obs=23, num_critic_obs=23, num_actions=6,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128],
            activation="elu",
            init_noise_std=0.0
        )
        
        # 모델의 예상 observation 차원 (기본값 23, 모델 로드 시 업데이트됨)
        self.expected_obs_dim = 23
        
        # ONNX 모델 관련 변수
        self.driving_onnx_session = None
        self.is_onnx_model = False
        self.current_onnx_session = None  # 현재 사용 중인 세션

        # 초기 모델 로드 (파일이 없어도 에러 없이 진행, GUI에서 선택할 때 로드)
        try:
            self.load_policy(pt_path)
        except FileNotFoundError:
            self.get_logger().warn(f"[INIT] Policy file not found: {pt_path}. Will load when GUI selects a file.")
            pt_path = None

        # 현재 모델 경로 저장
        self.current_policy_path = pt_path

        # ────────────── Control params ──────────────
        self.ENC_PER_RAD = 651.89865
        self.PI = 3.142

        self.ACTOR_DT = 1.0 / 50.0   # 50 Hz inference
        self.CTRL_DT  = 1.0 / 200.0  # 200 Hz motor send / feedback

        self.obs_buffer = deque(maxlen=4)

        self.prev_act = torch.zeros(6, device=self.device)
        
        # ────────────── Action Smoothing ──────────────
        # Exponential smoothing: smoothed_act = alpha * current + (1-alpha) * prev
        # alpha 값이 클수록 현재 값에 더 가중치 (0.0~1.0)
        # 0.8 = 현재 80%, 이전 20% (부드러운 전환)
        # 1.0 = 스무딩 없음 (즉시 반영)
        self.ACTION_SMOOTHING_ALPHA = 0.8  # 조정 가능: 0.7~0.9 권장
        
        # ────────────── Leg Joint Limits (라디안) ──────────────
        # 순서: [upper_L, upper_R, lower_L, lower_R]
        # margin=0.01 rad 적용하여 보수적으로 클램프
        self.LEG_MIN = torch.tensor([-0.8726646, -0.8726646, -1.0471976, -0.6108652], device=self.device) + 0.01
        self.LEG_MAX = torch.tensor([0.6981317, 0.6981317, 0.6108652, 1.0471976], device=self.device) - 0.01
        
        # YAW unwrapping을 위한 이전 yaw 값 저장
        self.prev_yaw = None
        
        # Quaternion double cover 문제 해결을 위한 이전 quaternion 저장
        self.prev_quat = None

        # Manual input (raw 6D from GUI)
        self.manual_act = torch.zeros(6, device=self.device)
        
        # Velocity command (from GUI, Manual/Auto 모두 사용)
        # 4차원 [vx, vy, vz, heading]
        self.velocity_command = torch.zeros(4, device=self.device)
        
        # BNO085 IMU 데이터 (개별 토픽에서 받음)
        self.bno085_base_ang_vel = None  # [gyro_x, gyro_y, gyro_z]
        self.bno085_quat = None  # [quat_i, quat_j, quat_k, quat_real]
        self.bno085_data_lock = threading.Lock()  # 스레드 안전성을 위한 락
        self.bno085_last_update = None
        self._bno085_warn_time = time.time()  # 경고 메시지 제어를 위한 초기화
        
        # EX_OBS 데이터 (A-RMA Adaptation Module 출력)
        self.ex_obs = None  # [8차원]
        self.ex_obs_lock = threading.Lock()  # 스레드 안전성을 위한 락
        
        # Auto mode debugging (1초마다 출력)
        self.auto_debug_count = 0
        self.auto_debug_last_time = time.time()
        self.auto_debug_hz_counter = 0
        self.auto_debug_hz_start_time = time.time()
        
        # Policy Hz tracking
        self.policy_hz_counter = 0
        self.policy_hz_start_time = time.time()
        self.current_policy_hz = 0.0
        
        # Inference 시간 측정
        self.inference_times = deque(maxlen=100)  # 최근 100개 inference 시간 저장
        self.inference_time_sum = 0.0
        self.inference_time_count = 0
        
        # 통합 디버그 출력 (1초마다)
        self.debug_last_time = time.time()
        
        # 타이밍 측정 (200Hz 주기 확인용)
        self.obs_loop_times = deque(maxlen=200)  # 1초간의 타이밍 저장
        self.ctrl_loop_times = deque(maxlen=200)  # 1초간의 타이밍 저장

        # current_act MUST ALWAYS be "converted act" that matches firmware expectations:
        # [wheel_L, wheel_R, upper_L, upper_R, lower_L, lower_R] with PI offsets
        self.current_act = self.convert_act(torch.zeros(6, device=self.device))

        # Mode state
        self.mode = "STOP"  # "MANUAL::RUN", "AUTO::RUN", "MANUAL::STOP", "AUTO::STOP", etc.
        
        # Policy Hz 초기화 (STOP 상태일 때 0으로 publish)
        hz_msg = Float64MultiArray()
        hz_msg.data = [0.0]
        self.policy_hz_pub.publish(hz_msg)

        # ────────────── Serial ──────────────
        try:
            self.serial = serial.Serial('/dev/ttyACM0', 1_000_000, timeout=0.005)  # 5ms timeout (200Hz = 5ms)
            # 시리얼 버퍼 크기 설정 (입력/출력 버퍼 최소화)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            self.get_logger().info(f"✓ Serial port opened: /dev/ttyACM0")
            # 초기화 시 안전을 위해 neutral pose로 설정 (RUN 명령 없이는 움직이지 않음)
            self.to_neutral()
        except Exception as e:
            self.get_logger().error(f"✗ Failed to open serial port: {e}")
            raise

        # ────────────── Threads ──────────────
        self.running = True

        self.obs_thread = threading.Thread(target=self.obs_loop, daemon=True)
        self.policy_thread = threading.Thread(target=self.policy_loop, daemon=True)
        self.ctrl_thread = threading.Thread(target=self.ctrl_loop, daemon=True)
        self.spin_thread = threading.Thread(target=self.ros_spin, daemon=True)

        self.obs_thread.start()
        self.policy_thread.start()
        self.ctrl_thread.start()
        self.spin_thread.start()

        self.get_logger().info("✓ Node started (threads: obs/policy/ctrl/spin)")
        
        # 초기 상태 출력
        time.sleep(0.5)  # 센서 데이터 수신 대기
        self.print_debug_status()

    # ─────────────────────────────
    # ROS Spin Thread
    # ─────────────────────────────
    def ros_spin(self):
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        while self.running and rclpy.ok():
            executor.spin_once(timeout_sec=0.01)

    # ─────────────────────────────
    # ROS Callbacks
    # ─────────────────────────────
    def cmd_callback(self, msg: String):
        new_mode = msg.data.strip()
        old_mode = self.mode
        self.mode = new_mode
        
        # RUN/STOP 명령만 간단하게 출력
        if "RUN" in new_mode:
            self.get_logger().info(f"[CMD] → RUN: {new_mode}")
        elif "STOP" in new_mode:
            self.get_logger().info(f"[CMD] → STOP: {new_mode}")

        # When entering MANUAL::RUN, start from NEUTRAL (converted zero)
        if self.mode == "MANUAL::RUN":
            z = torch.zeros(6, device=self.device)
            self.current_act = self.convert_act(z)

        # When entering AUTO::RUN, also start from neutral (optional, safer)
        if self.mode == "AUTO::RUN":
            z = torch.zeros(6, device=self.device)
            self.current_act = self.convert_act(z)

        # Any STOP -> go neutral immediately (one-shot)
        if "STOP" in self.mode:
            self.to_neutral()
            # Policy Hz를 0으로 설정
            self.current_policy_hz = 0.0
            hz_msg = Float64MultiArray()
            hz_msg.data = [0.0]
            self.policy_hz_pub.publish(hz_msg)

    def manual_callback(self, msg: Float64MultiArray):
        """Manual 명령 콜백: 반드시 6D, finite, then convert to firmware-space."""
        data = list(msg.data)

        # 1) length check
        if len(data) != 6:
            self.get_logger().warn(f"[MANUAL] Invalid length: {len(data)} (expected 6). Ignoring.")
            return

        # 2) finite check
        arr = np.asarray(data, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            self.get_logger().warn(f"[MANUAL] NaN/Inf detected: {data}. Ignoring.")
            return

        # 3) clamp raw manual domain (GUI range -10~10; keep some margin)
        arr = np.clip(arr, -10.0, 10.0)

        # Store raw manual and update converted current_act immediately
        self.manual_act = torch.tensor(arr, dtype=torch.float32, device=self.device)

        # Only apply to output when MANUAL::RUN
        if self.mode == "MANUAL::RUN":
            self.current_act = self.convert_act(self.manual_act)

    def velocity_command_callback(self, msg: Float64MultiArray):
        """Velocity command 콜백: Manual/Auto 모드 모두에서 사용
        4차원 [vx, vy, vz, heading]
        """
        data = list(msg.data)
        
        # 1) length check (4 차원)
        if len(data) != 4:
            self.get_logger().warn(f"[VELOCITY_CMD] Invalid length: {len(data)} (expected 4). Ignoring.")
            return
        
        # 2) finite check
        arr = np.asarray(data, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            self.get_logger().warn(f"[VELOCITY_CMD] NaN/Inf detected: {data}. Ignoring.")
            return
        
        # 3) Store velocity command
        self.velocity_command = torch.tensor(arr, dtype=torch.float32, device=self.device)
        self.get_logger().debug(f"[VELOCITY_CMD] Received: {data}")

    def bno085_gyro_callback(self, msg: Float64MultiArray):
        """BNO085 Gyroscope 데이터 콜백"""
        data = list(msg.data)
        
        # 첫 번째 메시지 수신 시 로그 출력
        if not hasattr(self, '_bno085_gyro_first_received'):
            self.get_logger().info(f"[BNO085] First gyroscope data received! Length: {len(data)}")
            self._bno085_gyro_first_received = True
        
        # 데이터 길이 확인 (3개: gyro_x, gyro_y, gyro_z)
        if len(data) != 3:
            self.get_logger().warn(f"[BNO085] Invalid gyroscope length: {len(data)} (expected 3). Ignoring.")
            return
        
        # finite check
        arr = np.asarray(data, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            self.get_logger().warn(f"[BNO085] NaN/Inf detected in gyroscope. Ignoring.")
            return
        
        # 스레드 안전하게 저장
        with self.bno085_data_lock:
            self.bno085_base_ang_vel = arr.copy()
            self.bno085_last_update = time.time()
    
    def bno085_quat_callback(self, msg: Float64MultiArray):
        """BNO085 Quaternion 데이터 콜백"""
        data = list(msg.data)
        
        # 첫 번째 메시지 수신 시 로그 출력
        if not hasattr(self, '_bno085_quat_first_received'):
            self.get_logger().info(f"[BNO085] First quaternion data received! Length: {len(data)}")
            self._bno085_quat_first_received = True
        
        # 데이터 길이 확인 (4개: quat_i, quat_j, quat_k, quat_real)
        if len(data) != 4:
            self.get_logger().warn(f"[BNO085] Invalid quaternion length: {len(data)} (expected 4). Ignoring.")
            return
        
        # finite check
        arr = np.asarray(data, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            self.get_logger().warn(f"[BNO085] NaN/Inf detected in quaternion. Ignoring.")
            return
        
        # 스레드 안전하게 저장
        with self.bno085_data_lock:
            self.bno085_quat = arr.copy()
            self.bno085_last_update = time.time()
    
    def ex_obs_callback(self, msg: Float64MultiArray):
        """EX_OBS 데이터 콜백 (A-RMA Adaptation Module 출력)"""
        data = list(msg.data)
        
        # 첫 번째 메시지 수신 시 로그 출력
        if not hasattr(self, '_ex_obs_first_received'):
            self.get_logger().info(f"[EX_OBS] First EX_OBS data received! Length: {len(data)}")
            self._ex_obs_first_received = True
        
        # 데이터 길이 확인 (8차원)
        if len(data) != 8:
            self.get_logger().warn(f"[EX_OBS] Invalid EX_OBS length: {len(data)} (expected 8). Ignoring.")
            return
        
        # finite check
        arr = np.asarray(data, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            self.get_logger().warn(f"[EX_OBS] NaN/Inf detected in EX_OBS. Ignoring.")
            return
        
        # 스레드 안전하게 저장
        with self.ex_obs_lock:
            self.ex_obs = arr.copy()

    def model_path_callback(self, msg: String):
        """모델 파일 경로 콜백: 실시간 모델 로딩
        형식: "DRIVING:/path/to/driving.onnx"
        또는 레거시: "/path/to/model.pt" 또는 "/path/to/model.onnx"
        """
        model_path = msg.data.strip()
        
        if not model_path:
            self.get_logger().warn("[MODEL] Empty model path received. Ignoring.")
            return
        
        # 모드 확인 (DRIVING: 접두사)
        if model_path.startswith("DRIVING:"):
            model_type = "DRIVING"
            actual_path = model_path[8:].strip()  # "DRIVING:" 제거
        else:
            # 레거시 모드: 단일 파일
            model_type = "LEGACY"
            actual_path = model_path
        
        # asset_vanilla 경로 찾기 헬퍼 함수
        def find_asset_vanilla_path():
            """asset_vanilla 디렉토리 경로 찾기"""
            possible_paths = [
                '/home/pi/ros2_ws/src/asset_vanilla',  # 라즈베리파이 workspace
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'asset_vanilla'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src', 'asset_vanilla'),
                '/home/pi/ros2_ws/src/redshow/src/asset_vanilla',
                '/home/sol/redshow/src/asset_vanilla',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            return None
        
        # GUI에서 보낸 경로는 다른 머신의 경로일 수 있으므로, 파일명만 추출해서 로컬에서 찾기
        # 경로 해석: 파일명 추출 > 로컬 asset_vanilla에서 찾기 > 상대 경로 > 절대 경로
        resolved_path = None
        
        # 파일명만 추출 (actual_path에서 파일명만 가져오기)
        file_name = os.path.basename(actual_path)
        
        # 먼저 로컬 asset_vanilla에서 찾기
        asset_vanilla_path = find_asset_vanilla_path()
        if asset_vanilla_path:
            candidate = os.path.join(asset_vanilla_path, file_name)
            if os.path.exists(candidate):
                resolved_path = candidate
                self.get_logger().info(f"[MODEL] Found model in asset_vanilla: {resolved_path}")
        
        # asset_vanilla에 없으면 actual_path로 시도
        if not resolved_path and os.path.exists(actual_path):
            resolved_path = os.path.abspath(actual_path)
        
        # 그래도 없으면 절대 경로로 시도 (다른 머신의 경로일 수 있으므로 마지막 시도)
        if not resolved_path and os.path.isabs(actual_path) and os.path.exists(actual_path):
            resolved_path = actual_path
            self.get_logger().warn(f"[MODEL] Using absolute path from GUI (may be from different machine): {resolved_path}")
        
        if not resolved_path:
            self.get_logger().error(f"[MODEL] Model file not found: {actual_path}")
            return
        
        # 파일 존재 확인
        if not os.path.exists(resolved_path):
            self.get_logger().error(f"[MODEL] Model file not found: {resolved_path}")
            return
        
        # 파일 확장자 확인 (.pt 또는 .onnx 지원)
        if not (resolved_path.endswith('.pt') or resolved_path.endswith('.onnx')):
            self.get_logger().warn(f"[MODEL] Only .pt and .onnx files are supported. Received: {resolved_path}")
            return
        
        # ONNX 파일인데 ONNX Runtime이 없으면 경고
        if resolved_path.endswith('.onnx') and not ONNX_AVAILABLE:
            self.get_logger().error(f"[MODEL] ONNX file detected but onnxruntime is not installed. Please install: pip install onnxruntime")
            return
        
        # 모델 로드 처리
        if model_type == "DRIVING":
            # 같은 경로면 무시
            if resolved_path == getattr(self, 'driving_policy_path', None):
                self.get_logger().info(f"[MODEL] Same Driving model path: {resolved_path}. Skipping reload.")
                return
            
            self.get_logger().info(f"[MODEL] Loading Driving ONNX model: {resolved_path}")
            
            # AUTO 모드 실행 중이면 일시 중지
            was_auto_running = (self.mode == "AUTO::RUN")
            if was_auto_running:
                self.get_logger().info("[MODEL] Temporarily stopping AUTO mode for Driving model reload...")
                self.mode = "AUTO::STOP"
                self.to_neutral()
            
            try:
                self.load_onnx_policy(resolved_path, model_type="DRIVING")
                self.driving_policy_path = resolved_path
                self.current_onnx_session = self.driving_onnx_session
                self.get_logger().info(f"✓ Driving ONNX model loaded successfully: {resolved_path}")
                
                if was_auto_running:
                    self.get_logger().info("[MODEL] Driving model reloaded. Please press RUN button to resume AUTO mode.")
            except Exception as e:
                import traceback
                self.get_logger().error(f"[MODEL] Failed to load Driving model: {e}")
                self.get_logger().error(f"[MODEL] Traceback: {traceback.format_exc()}")
                if was_auto_running:
                    self.get_logger().warn("[MODEL] Previous model remains active. AUTO mode stopped.")
        
        else:
            # 레거시 모드: 단일 파일
            # 같은 경로면 무시
            if resolved_path == self.current_policy_path:
                self.get_logger().info(f"[MODEL] Same model path: {resolved_path}. Skipping reload.")
                return
            
            self.get_logger().info(f"[MODEL] Loading legacy model: {resolved_path}")
            
            # AUTO 모드 실행 중이면 일시 중지
            was_auto_running = (self.mode == "AUTO::RUN")
            if was_auto_running:
                self.get_logger().info("[MODEL] Temporarily stopping AUTO mode for model reload...")
                self.mode = "AUTO::STOP"
                self.to_neutral()
            
            try:
                self.load_policy(resolved_path)
                self.current_policy_path = resolved_path
                self.get_logger().info(f"✓ Legacy model loaded successfully: {resolved_path}")
                
                if was_auto_running:
                    self.get_logger().info("[MODEL] Model reloaded. Please press RUN button to resume AUTO mode.")
            except Exception as e:
                import traceback
                self.get_logger().error(f"[MODEL] Failed to load model: {e}")
                self.get_logger().error(f"[MODEL] Traceback: {traceback.format_exc()}")
                if was_auto_running:
                    self.get_logger().warn("[MODEL] Previous model remains active. AUTO mode stopped.")
    
    def load_onnx_policy(self, model_path: str, model_type: str = "DRIVING"):
        """ONNX 모델 로드 함수
        model_type: "DRIVING"
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX file not found: {model_path}")
        
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed. Please install: pip install onnxruntime")
        
        # ONNX Runtime 세션 생성
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        onnx_session = ort.InferenceSession(model_path, providers=providers)
        
        # 입력/출력 이름 확인
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        
        # ONNX 모델의 입력 차원 확인
        input_shape = onnx_session.get_inputs()[0].shape
        if len(input_shape) >= 2:
            obs_dim = input_shape[1] if input_shape[1] != 'batch' else 23
            if isinstance(obs_dim, str):
                obs_dim = 23  # 동적 차원인 경우 기본값 사용
            expected_obs_dim = int(obs_dim)
        else:
            expected_obs_dim = 23
        
        self.get_logger().info(f"[ONNX] {model_type} model loaded: {model_path}")
        self.get_logger().info(f"[ONNX] Input: {input_name}, Output: {output_name}")
        self.get_logger().info(f"[ONNX] Input shape: {input_shape}, Expected obs dim: {expected_obs_dim}")
        self.get_logger().info(f"[ONNX] Providers: {onnx_session.get_providers()}")
        
        # 모델 저장
        self.driving_onnx_session = onnx_session
        self.is_onnx_model = True
        self.current_onnx_session = onnx_session
        
        # 액션 초기화
        self.prev_act = torch.zeros(6, device=self.device)
    
    def load_policy(self, model_path: str):
        """모델 로드 함수 (PT 또는 ONNX 지원) - 레거시 모드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Policy file not found: {model_path}")
        
        # ONNX 파일인 경우
        if model_path.endswith('.onnx'):
            if not ONNX_AVAILABLE:
                raise ImportError("onnxruntime is not installed. Please install: pip install onnxruntime")
            
            # 레거시 ONNX 모드: 단일 세션 사용
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            onnx_session = ort.InferenceSession(model_path, providers=providers)
            self.driving_onnx_session = onnx_session
            self.current_onnx_session = onnx_session
            self.is_onnx_model = True
            
            # 입력/출력 이름 확인
            input_name = onnx_session.get_inputs()[0].name
            output_name = onnx_session.get_outputs()[0].name
            
            # ONNX 모델의 입력 차원 확인
            input_shape = onnx_session.get_inputs()[0].shape
            if len(input_shape) >= 2:
                obs_dim = input_shape[1] if input_shape[1] != 'batch' else 23
                if isinstance(obs_dim, str):
                    obs_dim = 23  # 동적 차원인 경우 기본값 사용
                self.expected_obs_dim = int(obs_dim)
            else:
                self.expected_obs_dim = 23
            
            self.get_logger().info(f"[ONNX] Legacy ONNX model loaded: {model_path}")
            self.get_logger().info(f"[ONNX] Input: {input_name}, Output: {output_name}")
            self.get_logger().info(f"[ONNX] Input shape: {input_shape}, Expected obs dim: {self.expected_obs_dim}")
            self.get_logger().info(f"[ONNX] Providers: {onnx_session.get_providers()}")
            
            # 액션 초기화
            self.prev_act = torch.zeros(6, device=self.device)
            
            return
        
        # PT 파일인 경우
        self.is_onnx_model = False
        self.driving_onnx_session = None
        self.current_onnx_session = None
        
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor")}
        
        # Observation 차원 확인 (A-RMA 모델인지 확인)
        # actor의 첫 번째 레이어 weight의 shape[1]이 observation 차원
        obs_dim = 23  # 기본값
        
        # 여러 가능한 키 패턴 확인
        possible_keys = [
            "actor.mlp.0.weight",  # 일반적인 구조
            "actor.0.weight",      # Sequential 구조
            "actor.module.0.weight", # Module로 감싸진 경우
        ]
        
        for key in possible_keys:
            if key in actor_state:
                obs_dim = actor_state[key].shape[1]
                self.get_logger().info(f"[MODEL] Detected observation dimension: {obs_dim} (from {key})")
                break
        
        if obs_dim == 23:
            # 키를 찾지 못한 경우, 모든 actor 키를 확인
            for key in actor_state.keys():
                if "weight" in key and len(actor_state[key].shape) == 2:
                    # 첫 번째 레이어 찾기 (가장 작은 인덱스)
                    if "0" in key or "mlp" in key:
                        obs_dim = actor_state[key].shape[1]
                        self.get_logger().info(f"[MODEL] Detected observation dimension: {obs_dim} (from {key})")
                        break
        
        # 체크포인트에서 실제 hidden_dims 추출
        actor_hidden_dims = []
        critic_hidden_dims = []
        
        # actor hidden dims 추출 (0, 2, 4 레이어의 출력 차원)
        for i in [0, 2, 4]:
            key = f"actor.{i}.weight"
            if key in actor_state:
                actor_hidden_dims.append(actor_state[key].shape[0])
        
        # critic hidden dims 추출
        critic_state = {k: v for k, v in state_dict.items() if k.startswith("critic")}
        for i in [0, 2, 4]:
            key = f"critic.{i}.weight"
            if key in critic_state:
                critic_hidden_dims.append(critic_state[key].shape[0])
        
        # hidden_dims가 비어있으면 기본값 사용
        if not actor_hidden_dims:
            actor_hidden_dims = [128, 128, 128]
        if not critic_hidden_dims:
            critic_hidden_dims = [128, 128, 128]
        
        self.get_logger().info(f"[MODEL] Detected actor hidden dims: {actor_hidden_dims}")
        self.get_logger().info(f"[MODEL] Detected critic hidden dims: {critic_hidden_dims}")
        
        if obs_dim == 31:
            self.get_logger().info("[MODEL] A-RMA model detected (31-dim observation with EX_OBS)")
        elif obs_dim == 23:
            self.get_logger().info("[MODEL] Standard model detected (23-dim observation)")
        else:
            self.get_logger().warn(f"[MODEL] Unknown observation dimension: {obs_dim}")
        
        # 예상 observation 차원 저장
        self.expected_obs_dim = obs_dim
        
        # Observation 차원이나 hidden_dims가 다르면 policy 재초기화
        if obs_dim != 23 or actor_hidden_dims != [128, 128, 128]:
            self.get_logger().info(f"[MODEL] Reinitializing policy with observation dimension: {obs_dim}, hidden_dims: {actor_hidden_dims}")
            self.policy = ActorCritic(
                num_actor_obs=obs_dim, num_critic_obs=obs_dim, num_actions=6,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation="elu",
                init_noise_std=0.0
            )
        
        # 기존 policy에 로드
        self.policy.load_state_dict(actor_state, strict=False)
        self.policy.eval()
        self.policy.to(self.device)
        
        # 액션 초기화 (새 모델에 맞게)
        self.prev_act = torch.zeros(6, device=self.device)
        
        # PT 파일 구조 확인 (Observation Group 정보가 있는지 확인)
        self.inspect_checkpoint_structure(ckpt, model_path)
        
        self.get_logger().info(f"✓ Policy loaded: {model_path}")
    
    def inspect_checkpoint_structure(self, ckpt: dict, pt_path: str):
        """PT/ONNX 파일 구조 확인 - Observation Group 정보가 있는지 확인"""
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"[CHECKPOINT INSPECTION] Analyzing file: {pt_path}")
        self.get_logger().info("=" * 80)
        
        # 1. 모든 최상위 키 출력
        all_keys = list(ckpt.keys())
        self.get_logger().info(f"[CHECKPOINT] Top-level keys ({len(all_keys)}): {all_keys}")
        
        # 2. 각 키의 타입과 내용 일부 확인
        for key in all_keys:
            value = ckpt[key]
            value_type = type(value).__name__
            
            if isinstance(value, dict):
                sub_keys = list(value.keys())[:20]  # 처음 20개만
                self.get_logger().info(f"[CHECKPOINT]   {key}: dict with {len(value)} keys (showing first 20): {sub_keys}")
                
                # Observation 관련 키가 있는지 확인
                obs_related = [k for k in sub_keys if 'obs' in k.lower() or 'observation' in k.lower()]
                if obs_related:
                    self.get_logger().info(f"[CHECKPOINT]     ⚠️  Found observation-related keys: {obs_related}")
            elif isinstance(value, (list, tuple)):
                self.get_logger().info(f"[CHECKPOINT]   {key}: {value_type} with length {len(value)}")
            elif isinstance(value, torch.Tensor):
                self.get_logger().info(f"[CHECKPOINT]   {key}: Tensor with shape {value.shape}")
            else:
                # 작은 값만 출력 (너무 크면 생략)
                str_value = str(value)
                if len(str_value) > 200:
                    str_value = str_value[:200] + "..."
                self.get_logger().info(f"[CHECKPOINT]   {key}: {value_type} = {str_value}")
        
        # 3. Observation Group 관련 키 직접 확인
        obs_keywords = ['obs', 'observation', 'config', 'cfg', 'group', 'space']
        found_obs_keys = []
        for key in all_keys:
            if any(kw in key.lower() for kw in obs_keywords):
                found_obs_keys.append(key)
        
        if found_obs_keys:
            self.get_logger().info(f"[CHECKPOINT] ⚠️  Found potential observation-related keys: {found_obs_keys}")
            for key in found_obs_keys:
                value = ckpt[key]
                if isinstance(value, dict):
                    self.get_logger().info(f"[CHECKPOINT]   {key} content: {list(value.keys())}")
                else:
                    self.get_logger().info(f"[CHECKPOINT]   {key} content: {str(value)[:500]}")
        else:
            self.get_logger().info(f"[CHECKPOINT] ❌ No observation-related keys found in top-level")
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("[CHECKPOINT INSPECTION] Analysis complete. Check logs above for observation group information.")
        self.get_logger().info("=" * 80)

    # ─────────────────────────────
    # Sensor loop (200 Hz - 정주기)
    # ─────────────────────────────
    def obs_loop(self):
        self.get_logger().info("[OBS_LOOP] Thread started (200Hz 정주기)")
        read_count = 0
        raw_line_count = 0
        next_tick = time.perf_counter()
        OBS_DT = 1.0 / 200.0  # 200Hz = 5ms
        last_loop_time = time.perf_counter()
        
        while self.running:
            # 정주기로 읽기 시도
            current_time = time.perf_counter()
            sleep_time = next_tick - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += OBS_DT
            
            # 타이밍 측정
            loop_dt = current_time - last_loop_time
            self.obs_loop_times.append(loop_dt)
            last_loop_time = current_time
            
            try:
                # 사용 가능한 데이터 확인
                bytes_available = self.serial.in_waiting
                
                # 버퍼가 너무 많이 쌓이면 오래된 데이터 제거 (200Hz * 0.1초 = 20개 이상이면)
                if bytes_available > 2000:  # 대략 20개 라인 이상
                    self.serial.reset_input_buffer()
                    continue
                
                # 한 라인 읽기 시도 (non-blocking)
                line = self.serial.read_until(b"\n", size=200)  # 최대 200바이트
                
                # 빈 라인이면 스킵 (다음 주기에 다시 시도)
                if not line or len(line) == 0:
                    continue
                
                raw_line_count += 1
                line_str = line.decode(errors="ignore").strip()
                
                # "fb,"가 포함되어 있는지 확인 (앞에 다른 텍스트가 올 수 있음)
                fb_idx = line_str.find("fb,")
                if fb_idx == -1:
                    continue
                
                # "fb," 이후 부분만 사용
                fb_part = line_str[fb_idx:]
                parts = fb_part.split(",")
                
                # fb, 다음에 16개의 값이 있어야 함 (총 17개: fb + 16 values)
                if len(parts) != 17:
                    continue
                
                # 안전하게 float 변환 (잘못된 값 필터링)
                try:
                    fb_values = []
                    for x in parts[1:]:
                        x = x.strip()
                        # 빈 문자열, '-', 또는 잘못된 값 건너뛰기
                        if not x or x == '-' or x == '':
                            raise ValueError(f"Invalid value: '{x}'")
                        fb_values.append(float(x))
                    
                    fb = tuple(fb_values)
                    self.obs_buffer.append(fb)
                    read_count += 1
                except (ValueError, IndexError) as parse_error:
                    # 파싱 에러는 조용히 스킵 (잘못된 데이터)
                    continue
            except serial.SerialTimeoutException:
                # timeout은 정상 (읽을 데이터가 없을 때) - 다음 주기에 다시 시도
                continue
            except Exception as e:
                # 예상치 못한 에러만 로깅 (파싱 에러는 위에서 처리)
                self.get_logger().error(f"[OBS_LOOP] Unexpected error: {e}", exc_info=True)
                time.sleep(0.01)

    # ─────────────────────────────
    # Policy loop (50 Hz)
    # ─────────────────────────────
    def policy_loop(self):
        next_tick = time.perf_counter()
        loop_count = 0
        last_warn_time = time.time()
        while self.running:
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += self.ACTOR_DT
            loop_count += 1

            current_time = time.time()

            if len(self.obs_buffer) < 4:
                # AUTO::RUN 모드에서 obs_buffer가 부족하면 경고 (5초마다)
                if self.mode == "AUTO::RUN" and current_time - last_warn_time > 5.0:
                    self.get_logger().warn(f"[POLICY] Obs buffer 부족: {len(self.obs_buffer)}/4")
                    last_warn_time = current_time
                continue
            
            # AUTO::RUN 모드에서 첫 번째 실행 시 로그 출력
            if self.mode == "AUTO::RUN" and not hasattr(self, '_policy_loop_first_run'):
                self.get_logger().info(f"[POLICY] Policy loop running (ONNX: {self.is_onnx_model})")
                self._policy_loop_first_run = True

            # 펌웨어에서 받은 데이터: 위치(4) + 휠 속도(2) = 6개만 사용
            # IMU 데이터는 BNO085에서 받음
            avg = [sum(s[i] for s in self.obs_buffer) / len(self.obs_buffer) for i in range(16)]
            (U_posL, U_posR, L_posL, L_posR, wL, wR,
             _, _, _, _, _, _, _, _, _, _) = avg  # OpenCR IMU 데이터는 사용하지 않음

            # Manual Mode에서 Action 방향과 실제 로봇 방향이 동일하므로,
            # Action에서 부호 반전이 있는 부분은 Observation에서도 동일하게 부호 반전 필요
            # Action: upper_L(변경없음), upper_R(부호반전), lower_L(변경없음), lower_R(변경없음)
            # 펌웨어에서 이미 PI를 빼서 보내므로, 여기서는 PI를 빼지 않음
            leg_pos = torch.tensor([
                U_posL / self.ENC_PER_RAD - self.PI,                              # Left Upper Leg - Action[2]와 일치
                -U_posR / self.ENC_PER_RAD + self.PI,                              # Right Upper Leg (부호 반전) - Action[3]와 일치
                L_posL / self.ENC_PER_RAD - self.PI,                               # Left Lower Leg (변경 없음) - Action[4]와 일치
                L_posR / self.ENC_PER_RAD - self.PI                           # Right Lower Leg (변경 없음) - Action[5]와 일치
            ], dtype=torch.float32, device=self.device)

            # Wheel velocity: 실제로는 Action Scale이 곱해진 값이 들어옴
            # Observation을 policy-space로 만들기 위해 ACTION_SCALE_POLICY / actual_scale을 곱해야 함
            wheel_vel_raw = torch.tensor([self.WHEEL_SIGN * wL, self.WHEEL_SIGN * wR], dtype=torch.float32, device=self.device)
            
            # 현재 적용 중인 Action Scale 확인
            if self.mode == "AUTO::RUN":
                actual_action_scale = self.WHEEL_SCALE_DRIVE
            else:
                actual_action_scale = 1.0  # Manual 모드는 스케일 없음
            
            # Sim2Real 보정: 실제로는 actual_action_scale이 곱해진 값이 들어오므로
            # policy-space로 변환하려면 ACTION_SCALE_POLICY / actual_action_scale을 곱해야 함
            obs_scale_factor = self.ACTION_SCALE_POLICY / actual_action_scale
            wheel_vel = wheel_vel_raw * obs_scale_factor
            
            # BNO085에서 IMU 데이터 가져오기 (개별 토픽에서)
            with self.bno085_data_lock:
                if self.bno085_base_ang_vel is not None and self.bno085_quat is not None:
                    # gyro: [gyro_x, gyro_y, gyro_z]
                    base_ang = torch.tensor(self.bno085_base_ang_vel, dtype=torch.float32, device=self.device)
                    # quat: [quat_i, quat_j, quat_k, quat_real] -> (w, x, y, z) 순서로 변환
                    quat = torch.tensor([self.bno085_quat[3], self.bno085_quat[0], self.bno085_quat[1], self.bno085_quat[2]], dtype=torch.float32, device=self.device)
                else:
                    # BNO085 데이터가 없으면 기본값 사용 (경고는 5초마다만 출력)
                    if not hasattr(self, '_bno085_warn_time'):
                        self._bno085_warn_time = time.time()
                    if current_time - self._bno085_warn_time > 5.0:
                        self.get_logger().warn("[POLICY_LOOP] BNO085 data not available, using zeros. Make sure bno085_node is running.")
                        self._bno085_warn_time = current_time
                    base_ang = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
                    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            
            # Quaternion 정규화
            n = torch.linalg.vector_norm(quat)
            if n.item() > 1e-8:
                quat = quat / n
            else:
                quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            
            # Quaternion double cover 문제 해결: 가장 가까운 표현 선택 (짧은 경로)
            # 이전 quaternion과의 내적을 계산하여 부호 선택
            if self.prev_quat is not None:
                # 현재 quaternion과 이전 quaternion의 내적
                dot_product = torch.dot(quat, self.prev_quat)
                
                # 내적이 음수면 (각도 > 90도) 부호 반전하여 가장 가까운 표현 선택
                # 이렇게 하면 360도 회전 시에도 연속적으로 유지됨
                if dot_product.item() < 0.0:
                    quat = -quat
                    dot_product = -dot_product  # 부호 반전 후 내적도 업데이트
                
                # 추가 검증: 각 성분의 변화량이 너무 크면 부호 반전 고려
                # (이전 값과의 차이가 큰 경우, 반대 부호가 더 가까울 수 있음)
                diff = torch.abs(quat - self.prev_quat)
                diff_neg = torch.abs(-quat - self.prev_quat)
                if torch.sum(diff_neg).item() < torch.sum(diff).item():
                    quat = -quat
            else:
                # 첫 번째 quaternion: w 성분이 양수가 되도록 정규화
                if quat[0].item() < 0.0:
                    quat = -quat
            
            # 이전 quaternion 업데이트
            self.prev_quat = quat.clone()

            # Velocity command 사용 (GUI에서 받은 값)
            # 4차원 [vx, vy, vz, heading]
            vcmd = self.velocity_command.clone()

            # Observation 조립
            # leg_pos(4) + wheel_vel(2) + base_ang(3) + vcmd(4) + quat(4) + prev_act(6) = 23
            obs_base = torch.cat([leg_pos, wheel_vel, base_ang, vcmd, quat, self.prev_act])
            
            # EX_OBS 추가 (모델이 31차원을 기대하는 경우에만)
            obs = obs_base.unsqueeze(0)  # [1, 23]
            
            if self.expected_obs_dim == 31:
                # A-RMA 모델: EX_OBS 추가 필요
                with self.ex_obs_lock:
                    if self.ex_obs is not None:
                        ex_obs_tensor = torch.tensor(self.ex_obs, dtype=torch.float32, device=self.device)
                        obs = torch.cat([obs_base, ex_obs_tensor]).unsqueeze(0)  # [1, 31]
                    else:
                        # A-RMA 모델인데 EX_OBS가 없으면 경고 (5초마다)
                        if not hasattr(self, '_ex_obs_warn_time'):
                            self._ex_obs_warn_time = time.time()
                        if current_time - self._ex_obs_warn_time > 5.0:
                            self.get_logger().warn("[POLICY] A-RMA model expects 31-dim observation but EX_OBS is not available. Make sure adaptation_module_node is running.")
                            self._ex_obs_warn_time = current_time
            elif self.expected_obs_dim == 23:
                # Standard 모델: EX_OBS 제외
                pass  # obs는 이미 23차원
            else:
                # 알 수 없는 차원
                self.get_logger().warn(f"[POLICY] Unknown expected observation dimension: {self.expected_obs_dim}. Using 23-dim observation.")
            
            # Observation shape 확인 (첫 번째 실행 시)
            if self.mode == "AUTO::RUN" and not hasattr(self, '_obs_shape_logged'):
                ex_obs_info = f", ex_obs={self.ex_obs}" if self.ex_obs is not None else ", ex_obs=None"
                self.get_logger().info(f"[POLICY] Observation shape: {obs.shape}, values: leg_pos={leg_pos.tolist()}, wheel_vel={wheel_vel.tolist()}, base_ang={base_ang.tolist()}, vcmd={vcmd.tolist()}, quat={quat.tolist()}, prev_act={self.prev_act.tolist()}{ex_obs_info}")
                self._obs_shape_logged = True

            if self.mode == "AUTO::RUN":
                # Inference 시간 측정 시작
                inference_start = time.perf_counter()
                
                try:
                    if self.is_onnx_model and self.current_onnx_session is not None:
                        # ONNX 모델 inference (점프 모드에 따라 세션 선택)
                        obs_np = obs.cpu().numpy().astype(np.float32)
                        input_name = self.current_onnx_session.get_inputs()[0].name
                        output_name = self.current_onnx_session.get_outputs()[0].name
                        
                        # 첫 번째 inference 시 로그 출력
                        if not hasattr(self, '_onnx_first_inference'):
                            self.get_logger().info(f"[POLICY] ONNX inference: input_shape={obs_np.shape}, input_name={input_name}, output_name={output_name}")
                            self._onnx_first_inference = True
                        
                        outputs = self.current_onnx_session.run([output_name], {input_name: obs_np})
                        act = torch.tensor(outputs[0], device=self.device).squeeze()
                        
                        # 첫 번째 action 생성 시 로그 출력
                        if not hasattr(self, '_onnx_first_action'):
                            self.get_logger().info(f"[POLICY] ONNX action generated: {act.tolist()}")
                            self._onnx_first_action = True
                    elif self.is_onnx_model and self.current_onnx_session is None:
                        # ONNX 모드인데 세션이 없으면 경고
                        if not hasattr(self, '_onnx_session_warn_time'):
                            self._onnx_session_warn_time = time.time()
                        if current_time - self._onnx_session_warn_time > 5.0:
                            self.get_logger().warn(f"[POLICY] ONNX model not loaded. Please load ONNX file.")
                            self._onnx_session_warn_time = current_time
                        act = torch.zeros(6, device=self.device)
                    else:
                        # PyTorch 모델 inference
                        with torch.no_grad():
                            act = self.policy.act_inference(obs).squeeze()
                except Exception as e:
                    import traceback
                    self.get_logger().error(f"[POLICY] Inference error: {e}")
                    self.get_logger().error(f"[POLICY] Traceback: {traceback.format_exc()}")
                    # 에러 발생 시 기본값 사용
                    act = torch.zeros(6, device=self.device)
                
                # Inference 시간 측정 종료 및 저장
                inference_time = (time.perf_counter() - inference_start) * 1000.0  # ms 단위로 변환
                self.inference_times.append(inference_time)
                self.inference_time_sum += inference_time
                self.inference_time_count += 1

                # Action smoothing: 이전 값과 현재 값을 부드럽게 보간
                # Exponential smoothing: smoothed = alpha * current + (1-alpha) * prev
                if self.ACTION_SMOOTHING_ALPHA < 1.0:
                    act_smoothed = self.ACTION_SMOOTHING_ALPHA * act + (1.0 - self.ACTION_SMOOTHING_ALPHA) * self.prev_act
                else:
                    act_smoothed = act  # 스무딩 없음
                
                # 이전 액션 업데이트 (스무딩된 값 저장)
                self.prev_act = act_smoothed.clone()

                # convert to firmware-space (Auto 모드이므로 wheel에 스케일 적용)
                self.current_act = self.convert_act(act_smoothed, is_auto=True)
                
                # Policy Hz 계산 및 publish
                self.policy_hz_counter += 1
                current_time_debug = time.time()
                elapsed = current_time_debug - self.policy_hz_start_time
                if elapsed >= 1.0:
                    self.current_policy_hz = self.policy_hz_counter / elapsed
                    self.policy_hz_counter = 0
                    self.policy_hz_start_time = current_time_debug
                    
                    # Policy Hz publish
                    hz_msg = Float64MultiArray()
                    hz_msg.data = [self.current_policy_hz]
                    self.policy_hz_pub.publish(hz_msg)
                    
                    # 통합 디버그 출력 (policy_loop에서 1초마다)
                    self.print_debug_status(act)

            # MANUAL::RUN: output updated in manual_callback
            # STOP: handled by cmd_callback -> to_neutral()

    # ─────────────────────────────
    # Motor control loop (200 Hz)
    # ─────────────────────────────
    def ctrl_loop(self):
        next_tick = time.perf_counter()
        debug_counter = 0
        last_loop_time = time.perf_counter()

        while self.running:
            current_time = time.perf_counter()
            sleep_time = next_tick - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += self.CTRL_DT
            
            # 타이밍 측정
            loop_dt = current_time - last_loop_time
            self.ctrl_loop_times.append(loop_dt)
            last_loop_time = current_time

            # Send motor command only when RUN
            if self.mode in ("AUTO::RUN", "MANUAL::RUN"):
                # MANUAL 모드일 때만 디버그 출력 (AUTO는 policy_loop에서 출력)
                if self.mode == "MANUAL::RUN":
                    debug_counter += 1
                    if debug_counter >= 200:  # 1초마다 (200Hz)
                        debug_counter = 0
                        self.print_debug_status()
                
                self.send_motor_cmd(self.current_act, log=False)

            # Always publish feedback
            self.publish_feedback()
            
            # STOP 모드일 때도 상태 확인 (5초마다)
            if "STOP" in self.mode:
                debug_counter += 1
                if debug_counter >= 1000:  # 5초마다 (200Hz * 5)
                    debug_counter = 0
                    self.print_debug_status()
            
            # Policy Hz는 policy_loop에서만 publish (중복 방지)

    # ─────────────────────────────
    # Helpers
    # ─────────────────────────────
    def quaternion_to_rpy(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Quaternion (w, x, y, z)을 Roll, Pitch, Yaw (RPY)로 변환
        YAW 각도 wrapping 문제 해결 (unwrapping 적용)
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # Clamp sinp to [-1, 1] to avoid domain error
        sinp_clamped = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp_clamped)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        # YAW unwrapping: 이전 값과의 차이가 π를 넘어가면 2π 보정
        if self.prev_yaw is not None:
            yaw_diff = yaw.item() - self.prev_yaw
            # 차이가 π보다 크면 -2π, -π보다 작으면 +2π
            if yaw_diff > np.pi:
                yaw = yaw - 2 * np.pi
            elif yaw_diff < -np.pi:
                yaw = yaw + 2 * np.pi
        
        # 이전 yaw 값 업데이트
        self.prev_yaw = yaw.item()
        
        return torch.stack([roll, pitch, yaw])
    
    def convert_act(self, act: torch.Tensor, is_auto: bool = False) -> torch.Tensor:
        """
        Convert policy/manual-space action -> firmware-space command.
        Sim2Real 보정: 실제 적용할 Action Scale을 사용하여 변환
        0 input should map to NEUTRAL pose (upper/lower = PI).
        바퀴 속도 -0.3~0.3 범위는 DeadZone으로 처리 (역구동 Jerk 방지).
        Leg joints는 policy-space(PI 더하기 전)에서 하드 클램프 적용.
        """
        converted = torch.zeros(6, device=self.device)
        
        # AUTO에서만 스케일 적용
        if is_auto:
            wheel_scale = self.WHEEL_SCALE_DRIVE
        else:
            wheel_scale = 1.0
        
        # DeadZone 적용: -0.2~0.2 범위는 0으로 처리 (Tensor 연산 사용)
        wheel_L_raw = act[0].clone()
        wheel_R_raw = act[1].clone()
        
        # DeadZone: 절댓값이 0.3 이하이면 0으로 설정
        wheel_L_raw = torch.where(torch.abs(wheel_L_raw) <= 0.1, torch.tensor(0.0, device=self.device), wheel_L_raw)
        wheel_R_raw = torch.where(torch.abs(wheel_R_raw) <= 0.1, torch.tensor(0.0, device=self.device), wheel_R_raw)
        
        # wheel: 방향 부호(WHEEL_SIGN) 반영 후 실제 적용 스케일 곱하기, MAX_WHEEL_SPEED로 클램프
        converted[0] = torch.clamp(self.WHEEL_SIGN * wheel_L_raw * wheel_scale, -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED)  # wheel_L
        converted[1] = torch.clamp(self.WHEEL_SIGN * wheel_R_raw * wheel_scale, -self.MAX_WHEEL_SPEED, self.MAX_WHEEL_SPEED)  # wheel_R
        
        # Leg joints: policy-space에서 하드 클램프 (PI 더하기 전)
        # 순서: [upper_L, upper_R, lower_L, lower_R] = act[2:6]
        leg = act[2:6].clone()
        leg = torch.max(torch.min(leg, self.LEG_MAX), self.LEG_MIN)
        
        # Manual Mode 기준: Action 방향과 실제 로봇 방향이 동일
        # Observation에서 부호 반전이 있는 부분은 Action에서도 동일하게 부호 반전 필요
        converted[2] = leg[0] + self.PI      # upper_L - Observation[0]와 일치
        converted[3] = -leg[1] + self.PI     # upper_R (부호 반전) - Observation[1]와 일치
        converted[4] = leg[2] + self.PI      # lower_L (변경 없음) - Observation[2]와 일치
        converted[5] = leg[3] + self.PI      # lower_R (변경 없음) - Observation[3]와 일치
        return converted

    def send_motor_cmd(self, act: torch.Tensor, log: bool = False):
        pkt = (
            f"act,{act[0].item():.2f},{act[1].item():.2f},"
            f"{act[2].item():.2f},{act[3].item():.2f},"
            f"{act[4].item():.2f},{act[5].item():.2f}\n"
        )
        try:
            # 출력 버퍼가 너무 많이 쌓여있으면 플러시 (지연 방지)
            if self.serial.out_waiting > 100:  # 약 10개 명령 이상 쌓이면
                self.serial.reset_output_buffer()
            
            # 명령 전송
            bytes_written = self.serial.write(pkt.encode())
            
            # 즉시 플러시하여 지연 최소화
            self.serial.flush()
            
            if log:
                self.get_logger().info(f"[ACT] 전송: {pkt.strip()}, 버퍼: {self.serial.out_waiting} bytes")
        except Exception as e:
            self.get_logger().error(f"[ACT] 전송 실패: {e}")
    
    def print_debug_status(self, act: torch.Tensor = None):
        """통합 디버그 상태 출력"""
        # Serial 연결 상태
        serial_ok = self.serial.is_open if hasattr(self.serial, 'is_open') else True
        
        # Obs 버퍼 상태
        obs_status = f"{len(self.obs_buffer)}/4"
        obs_ok = len(self.obs_buffer) >= 4
        
        # BNO085 연결 상태
        with self.bno085_data_lock:
            bno085_ok = (self.bno085_base_ang_vel is not None and self.bno085_quat is not None)
        
        # ACT 명령
        if act is not None:
            act_str = f"{act[0].item():.2f},{act[1].item():.2f},{act[2].item():.2f},{act[3].item():.2f},{act[4].item():.2f},{act[5].item():.2f}"
        else:
            act_str = f"{self.current_act[0].item():.2f},{self.current_act[1].item():.2f},{self.current_act[2].item():.2f},{self.current_act[3].item():.2f},{self.current_act[4].item():.2f},{self.current_act[5].item():.2f}"
        
        # Policy Hz (AUTO 모드일 때만)
        hz_str = f"Hz:{self.current_policy_hz:.1f}" if self.mode == "AUTO::RUN" else ""
        
        # Inference 시간 통계 (AUTO 모드일 때만)
        inference_str = ""
        if self.mode == "AUTO::RUN" and len(self.inference_times) > 0:
            avg_inference = sum(self.inference_times) / len(self.inference_times)
            min_inference = min(self.inference_times)
            max_inference = max(self.inference_times)
            inference_str = f" | Inf: {avg_inference:.2f}ms (min:{min_inference:.2f}, max:{max_inference:.2f})"
        
        # 루프 주기 측정 (실제 Hz)
        obs_hz = 0.0
        ctrl_hz = 0.0
        if len(self.obs_loop_times) > 10:
            avg_dt = sum(self.obs_loop_times) / len(self.obs_loop_times)
            obs_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
        if len(self.ctrl_loop_times) > 10:
            avg_dt = sum(self.ctrl_loop_times) / len(self.ctrl_loop_times)
            ctrl_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
        
        # 상태 표시
        serial_str = "✓" if serial_ok else "✗"
        obs_str = "✓" if obs_ok else "✗"
        bno085_str = "✓" if bno085_ok else "✗"
        
        # 출력
        status_line = f"[DEBUG] Mode: {self.mode:12s} | Serial: {serial_str} | Obs: {obs_status} {obs_str}({obs_hz:.1f}Hz) | BNO085: {bno085_str}"
        if hz_str:
            status_line += f" | {hz_str}"
        if inference_str:
            status_line += inference_str
        status_line += f" | ACT: {act_str} | Ctrl: {ctrl_hz:.1f}Hz"
        
        self.get_logger().info(status_line)

    def publish_observations(self, leg_pos: torch.Tensor, wheel_vel: torch.Tensor, 
                            base_ang: torch.Tensor, vcmd, 
                            quat: torch.Tensor, rpy: torch.Tensor, actions: torch.Tensor):
        """각 observation을 개별 토픽으로 publish
        vcmd는 torch.Tensor 또는 list일 수 있음 (4차원)
        """
        # LEG_POSITION (4개)
        msg_leg = Float64MultiArray()
        msg_leg.data = [leg_pos[i].item() for i in range(4)]
        self.leg_position_pub.publish(msg_leg)
        
        # WHEEL_VELOCITY (2개)
        msg_wheel = Float64MultiArray()
        msg_wheel.data = [wheel_vel[i].item() for i in range(2)]
        self.wheel_velocity_pub.publish(msg_wheel)
        
        # BASE_ANG_VEL (3개)
        msg_ang = Float64MultiArray()
        msg_ang.data = [base_ang[i].item() for i in range(3)]
        self.base_ang_vel_pub.publish(msg_ang)
        
        # VELOCITY_COMMANDS (4개)
        msg_vcmd = Float64MultiArray()
        if isinstance(vcmd, (list, tuple)):
            # 리스트나 튜플인 경우
            msg_vcmd.data = list(vcmd)
        elif isinstance(vcmd, torch.Tensor):
            # Tensor인 경우
            vcmd_len = vcmd.shape[0]
            msg_vcmd.data = [vcmd[i].item() for i in range(vcmd_len)]
        else:
            # 기타 타입 (numpy array 등)
            msg_vcmd.data = list(vcmd)
        self.velocity_commands_pub.publish(msg_vcmd)
        
        # BASE_QUAT (4개: w, x, y, z)
        msg_quat = Float64MultiArray()
        msg_quat.data = [quat[i].item() for i in range(4)]
        self.base_quat_pub.publish(msg_quat)
        
        # BASE_RPY (3개: roll, pitch, yaw)
        msg_rpy = Float64MultiArray()
        msg_rpy.data = [rpy[i].item() for i in range(3)]
        self.base_rpy_pub.publish(msg_rpy)
        
        # ACTIONS (6개)
        msg_act = Float64MultiArray()
        msg_act.data = [actions[i].item() for i in range(6)]
        self.actions_pub.publish(msg_act)
    
    def publish_feedback(self):
        # obs_buffer가 비어있으면 0으로 채워진 데이터 publish
        # (GUI에서 Observation Status가 "None"으로 표시되도록)
        if len(self.obs_buffer) >= 1:
            avg = [sum(s[i] for s in self.obs_buffer) / len(self.obs_buffer) for i in range(16)]
        else:
            # obs_buffer가 비어있으면 0으로 채운 데이터 publish
            # 이렇게 하면 GUI에서 데이터가 없다는 것을 알 수 있음
            avg = [0.0] * 16
            # 디버깅: obs_buffer가 비어있을 때 로그 출력 (5초마다)
            if not hasattr(self, '_last_empty_buffer_log_time'):
                self._last_empty_buffer_log_time = time.time()
            current_time = time.time()
            if current_time - self._last_empty_buffer_log_time > 5.0:
                self.get_logger().warn(
                    f"[FEEDBACK] obs_buffer is empty (size: {len(self.obs_buffer)}). "
                    f"No sensor data received. Check serial port connection (/dev/ttyACM0)."
                )
                self._last_empty_buffer_log_time = current_time

        # 펌웨어에서 받은 데이터: 위치(4) + 휠 속도(2)만 사용
        (U_posL, U_posR, L_posL, L_posR, wL, wR,
         _, _, _, _, _, _, _, _, _, _) = avg  # OpenCR IMU 데이터는 사용하지 않음

        # Manual Mode 기준: Action 방향과 실제 로봇 방향이 동일하므로,
        # Action에서 부호 반전이 있는 부분은 Observation에서도 동일하게 부호 반전 필요
        # Action: upper_L(변경없음), upper_R(부호반전), lower_L(변경없음), lower_R(변경없음)
        # 펌웨어에서 이미 PI를 빼서 보내므로, 여기서는 PI를 빼지 않음
        leg_pos = [
            U_posL / self.ENC_PER_RAD - self.PI,                              # Left Upper Leg - Action[2]와 일치
            -U_posR / self.ENC_PER_RAD + self.PI,                              # Right Upper Leg (부호 반전) - Action[3]와 일치
            L_posL / self.ENC_PER_RAD- self.PI,                               # Left Lower Leg (변경 없음) - Action[4]와 일치
            L_posR / self.ENC_PER_RAD- self.PI                                # Right Lower Leg (변경 없음) - Action[5]와 일치
        ]
        # Wheel velocity: 실제로는 Action Scale이 곱해진 값이 들어옴
        # Observation을 policy-space로 만들기 위해 ACTION_SCALE_POLICY / actual_scale을 곱해야 함
        wheel_vel_raw = [self.WHEEL_SIGN * wL, self.WHEEL_SIGN * wR]
        
        # 현재 적용 중인 Action Scale 확인
        if self.mode == "AUTO::RUN":
            actual_action_scale = self.WHEEL_SCALE_DRIVE
        else:
            actual_action_scale = 1.0  # Manual 모드는 스케일 없음
        
        # Sim2Real 보정: 실제로는 actual_action_scale이 곱해진 값이 들어오므로
        # policy-space로 변환하려면 ACTION_SCALE_POLICY / actual_action_scale을 곱해야 함
        obs_scale_factor = self.ACTION_SCALE_POLICY / actual_action_scale
        wheel_vel = [v * obs_scale_factor for v in wheel_vel_raw]
        
        # BNO085에서 IMU 데이터 가져오기 (개별 토픽에서)
        with self.bno085_data_lock:
            if self.bno085_base_ang_vel is not None and self.bno085_quat is not None:
                # gyro: [gyro_x, gyro_y, gyro_z]
                base_ang = list(self.bno085_base_ang_vel)
                # quat: [quat_i, quat_j, quat_k, quat_real] -> (w, x, y, z) 순서로 변환
                quat = [self.bno085_quat[3], self.bno085_quat[0], self.bno085_quat[1], self.bno085_quat[2]]
            else:
                # BNO085 데이터가 없으면 기본값 사용
                base_ang = [0.0, 0.0, 0.0]
                quat = [1.0, 0.0, 0.0, 0.0]
        
        # Velocity command 사용 (GUI에서 받은 값)
        # 4차원 [vx, vy, vz, heading]
        vcmd = [self.velocity_command[i].item() for i in range(4)]
        
        # 실제로 전송된 Action을 policy-space로 역변환하여 사용
        # self.current_act는 firmware-space이므로 policy-space로 변환 필요
        # convert_act의 역변환: wheel은 부호 반전 후 스케일 제거, leg는 PI 제거
        is_auto = self.mode == "AUTO::RUN"
        if is_auto:
            actual_action_scale = self.WHEEL_SCALE_DRIVE
        else:
            actual_action_scale = 1.0
        
        policy_act = torch.zeros(6, device=self.device)
        policy_act[0] = self.WHEEL_SIGN * self.current_act[0] / actual_action_scale  # wheel_L 역변환
        policy_act[1] = self.WHEEL_SIGN * self.current_act[1] / actual_action_scale  # wheel_R 역변환
        policy_act[2] = self.current_act[2] - self.PI      # upper_L 역변환
        policy_act[3] = -(self.current_act[3] - self.PI)   # upper_R 역변환 (부호 반전)
        policy_act[4] = self.current_act[4] - self.PI      # lower_L 역변환
        policy_act[5] = self.current_act[5] - self.PI      # lower_R 역변환
        prev_act_list = [policy_act[i].item() for i in range(6)]
        
        # Quaternion을 RPY로 변환
        quat_tensor = torch.tensor(quat, dtype=torch.float32, device=self.device)
        
        # Quaternion 정규화
        n = torch.linalg.vector_norm(quat_tensor)
        if n.item() > 1e-8:
            quat_tensor = quat_tensor / n
        else:
            quat_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        
        # Quaternion double cover 문제 해결: 가장 가까운 표현 선택 (짧은 경로)
        # 이전 quaternion과의 내적을 계산하여 부호 선택
        if self.prev_quat is not None:
            # 현재 quaternion과 이전 quaternion의 내적
            dot_product = torch.dot(quat_tensor, self.prev_quat)
            
            # 내적이 음수면 (각도 > 90도) 부호 반전하여 가장 가까운 표현 선택
            # 이렇게 하면 360도 회전 시에도 연속적으로 유지됨
            if dot_product.item() < 0.0:
                quat_tensor = -quat_tensor
                dot_product = -dot_product
            
            # 추가 검증: 각 성분의 변화량이 너무 크면 부호 반전 고려
            # (이전 값과의 차이가 큰 경우, 반대 부호가 더 가까울 수 있음)
            diff = torch.abs(quat_tensor - self.prev_quat)
            diff_neg = torch.abs(-quat_tensor - self.prev_quat)
            if torch.sum(diff_neg).item() < torch.sum(diff).item():
                quat_tensor = -quat_tensor
        else:
            # 첫 번째 quaternion: w 성분이 양수가 되도록 정규화
            if quat_tensor[0].item() < 0.0:
                quat_tensor = -quat_tensor
        
        # 이전 quaternion 업데이트 (policy_loop와 동일한 변수 사용)
        self.prev_quat = quat_tensor.clone()
        
        rpy_tensor = self.quaternion_to_rpy(quat_tensor)
        rpy_list = [rpy_tensor[i].item() for i in range(3)]
        
        # 각 observation을 개별 토픽으로 publish
        leg_pos_tensor = torch.tensor(leg_pos, dtype=torch.float32, device=self.device)
        wheel_vel_tensor = torch.tensor(wheel_vel, dtype=torch.float32, device=self.device)
        base_ang_tensor = torch.tensor(base_ang, dtype=torch.float32, device=self.device)
        vcmd_tensor = torch.tensor(vcmd, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(prev_act_list, dtype=torch.float32, device=self.device)
        self.publish_observations(leg_pos_tensor, wheel_vel_tensor, base_ang_tensor, 
                                 vcmd_tensor, quat_tensor, rpy_tensor, actions_tensor)

    def to_neutral(self):
        # send_motor_cmd와 동일한 형식으로 포맷팅 (.2f)
        neutral_pkt = f"act,0.00,0.00,{self.PI:.2f},{self.PI:.2f},{self.PI:.2f},{self.PI:.2f}\n"
        try:
            self.serial.write(neutral_pkt.encode())
        except Exception as e:
            self.get_logger().warn(f"[ACT] Neutral 전송 실패: {e}")

    def destroy_node(self):
        self.running = False
        
        # GUI에 종료 신호 전송
        try:
            shutdown_msg = String()
            shutdown_msg.data = "shutdown"
            self.shutdown_pub.publish(shutdown_msg)
            time.sleep(0.1)  # 메시지 전송 대기
        except Exception as e:
            self.get_logger().warn(f"Failed to send shutdown signal: {e}")
        
        try:
            self.serial.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = BerkeleyControlNode()
    try:
        # spin handled in thread; keep process alive
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
