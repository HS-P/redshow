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


class BerkeleyControlNode(Node):
    def __init__(self):
        super().__init__('berkeley_control_node')

        # ────────────── ROS2 I/O ──────────────
        self.cmd_sub = self.create_subscription(String, 'redshow/cmd', self.cmd_callback, 10)
        self.manual_sub = self.create_subscription(Float64MultiArray, 'redshow/joint_cmd', self.manual_callback, 10)
        self.model_path_sub = self.create_subscription(String, 'redshow/model_path', self.model_path_callback, 10)
        self.feedback_pub = self.create_publisher(Float64MultiArray, 'redshow/feedback', 10)
        self.policy_hz_pub = self.create_publisher(Float64MultiArray, 'redshow/policy_hz', 10)
        self.shutdown_pub = self.create_publisher(String, 'redshow/shutdown', 10)

        # ────────────── Device ──────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ────────────── Policy load ──────────────
        self.declare_parameter('policy_name', 'model_4800.pt')
        policy_name = self.get_parameter('policy_name').get_parameter_value().string_value

        if os.path.isabs(policy_name):
            pt_path = policy_name
        elif os.path.exists(policy_name):
            pt_path = policy_name
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

        # 초기 모델 로드
        self.load_policy(pt_path)

        # 현재 모델 경로 저장
        self.current_policy_path = pt_path

        # ────────────── Control params ──────────────
        self.ENC_PER_RAD = 651.89865
        self.PI = 3.142

        self.ACTOR_DT = 1.0 / 50.0   # 50 Hz inference
        self.CTRL_DT  = 1.0 / 200.0  # 200 Hz motor send / feedback

        self.obs_buffer = deque(maxlen=4)

        self.prev_act = torch.zeros(6, device=self.device)
        self.smoothed_act = torch.zeros(6, device=self.device)

        # Manual input (raw 6D from GUI)
        self.manual_act = torch.zeros(6, device=self.device)
        
        # Auto mode debugging (1초마다 출력)
        self.auto_debug_count = 0
        self.auto_debug_last_time = time.time()
        self.auto_debug_hz_counter = 0
        self.auto_debug_hz_start_time = time.time()
        
        # Policy Hz tracking
        self.policy_hz_counter = 0
        self.policy_hz_start_time = time.time()
        self.current_policy_hz = 0.0

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
            self.serial = serial.Serial('/dev/ttyACM0', 1_000_000, timeout=0.1)
            self.get_logger().info(f"✓ Serial port opened: /dev/ttyACM0")
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
        self.mode = new_mode
        self.get_logger().info(f"[CMD] Mode changed: {self.mode}")

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

    def model_path_callback(self, msg: String):
        """모델 파일 경로 콜백: 실시간 모델 로딩"""
        model_path = msg.data.strip()
        
        if not model_path:
            self.get_logger().warn("[MODEL] Empty model path received. Ignoring.")
            return
        
        # 같은 경로면 무시
        if model_path == self.current_policy_path:
            self.get_logger().info(f"[MODEL] Same model path: {model_path}. Skipping reload.")
            return
        
        # 파일 존재 확인
        if not os.path.exists(model_path):
            self.get_logger().error(f"[MODEL] Model file not found: {model_path}")
            return
        
        # 파일 확장자 확인 (.pt만 지원)
        if not model_path.endswith('.pt'):
            self.get_logger().warn(f"[MODEL] Only .pt files are supported. Received: {model_path}")
            return
        
        self.get_logger().info(f"[MODEL] Loading new model: {model_path}")
        
        # AUTO 모드 실행 중이면 일시 중지
        was_auto_running = (self.mode == "AUTO::RUN")
        if was_auto_running:
            self.get_logger().info("[MODEL] Temporarily stopping AUTO mode for model reload...")
            self.mode = "AUTO::STOP"
            self.to_neutral()
        
        # 모델 로드 시도
        try:
            self.load_policy(model_path)
            self.current_policy_path = model_path
            self.get_logger().info(f"✓ Model loaded successfully: {model_path}")
            
            # AUTO 모드가 실행 중이었으면 다시 시작 (사용자가 RUN 버튼을 다시 눌러야 함)
            if was_auto_running:
                self.get_logger().info("[MODEL] Model reloaded. Please press RUN button to resume AUTO mode.")
        except Exception as e:
            self.get_logger().error(f"[MODEL] Failed to load model: {e}", exc_info=True)
            if was_auto_running:
                self.get_logger().warn("[MODEL] Previous model remains active. AUTO mode stopped.")
    
    def load_policy(self, pt_path: str):
        """모델 로드 함수 (재사용 가능)"""
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Policy file not found: {pt_path}")
        
        # 모델 로드
        ckpt = torch.load(pt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor")}
        
        # 기존 policy에 로드
        self.policy.load_state_dict(actor_state, strict=False)
        self.policy.eval()
        self.policy.to(self.device)
        
        # 액션 초기화 (새 모델에 맞게)
        self.prev_act = torch.zeros(6, device=self.device)
        self.smoothed_act = torch.zeros(6, device=self.device)
        
        # PT 파일 구조 확인 (Observation Group 정보가 있는지 확인)
        self.inspect_checkpoint_structure(ckpt, pt_path)
        
        self.get_logger().info(f"✓ Policy loaded: {pt_path}")
    
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
    # Sensor loop (200 Hz)
    # ─────────────────────────────
    def obs_loop(self):
        self.get_logger().info("[OBS_LOOP] Thread started")
        read_count = 0
        raw_line_count = 0
        last_log_time = time.time()
        while self.running:
            try:
                # timeout=0.1로 설정했으므로 읽을 데이터가 없으면 빈 바이트 반환
                line = self.serial.read_until(b"\n")
                
                # 빈 라인이면 계속
                if not line:
                    time.sleep(0.001)  # CPU 사용량 줄이기
                    continue
                
                raw_line_count += 1
                line_str = line.decode(errors="ignore").strip()
                
                # 디버깅: 처음 몇 개의 raw 라인 출력
                if raw_line_count <= 5:
                    self.get_logger().info(f"[OBS_LOOP] Raw line #{raw_line_count}: {line_str[:100]}")
                
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
                    
                fb = tuple(float(x) for x in parts[1:])
                self.obs_buffer.append(fb)
                read_count += 1
                
                # 디버깅: 센서 데이터 수신 확인 (2초마다)
                current_time = time.time()
                if current_time - last_log_time > 2.0:
                    self.get_logger().info(f"[OBS_LOOP] Reading sensor data, buffer size: {len(self.obs_buffer)}, read_count: {read_count}, raw_lines: {raw_line_count}")
                    last_log_time = current_time
            except serial.SerialTimeoutException:
                # timeout은 정상 (읽을 데이터가 없을 때)
                time.sleep(0.001)
            except Exception as e:
                self.get_logger().error(f"[OBS_LOOP] Error: {e}", exc_info=True)
                time.sleep(0.1)

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

            # 디버깅: obs_buffer 상태 확인
            current_time = time.time()
            if self.mode == "AUTO::RUN" and current_time - last_warn_time > 2.0:
                self.get_logger().info(f"[POLICY_LOOP] Running, obs_buffer size: {len(self.obs_buffer)}, loop_count: {loop_count}, mode: {self.mode}")
                last_warn_time = current_time

            if len(self.obs_buffer) < 4:
                # AUTO::RUN 모드에서 obs_buffer가 부족하면 경고
                if self.mode == "AUTO::RUN" and current_time - last_warn_time > 2.0:
                    self.get_logger().warn(f"[POLICY_LOOP] Waiting for sensor data, obs_buffer size: {len(self.obs_buffer)}/4")
                continue

            avg = [sum(s[i] for s in self.obs_buffer) / len(self.obs_buffer) for i in range(16)]
            (U_posL, U_posR, L_posL, L_posR, wL, wR,
             gx, gy, gz, ax, ay, az, qw, qx, qy, qz) = avg

            leg_pos = torch.tensor([
                U_posL / self.ENC_PER_RAD - self.PI,
                U_posR / self.ENC_PER_RAD - self.PI,
                -(L_posL / self.ENC_PER_RAD - self.PI),
                -(L_posR / self.ENC_PER_RAD - self.PI)
            ], dtype=torch.float32, device=self.device)

            wheel_vel = torch.tensor([-wL, -wR], dtype=torch.float32, device=self.device)
            base_ang = torch.tensor([gx, gy, gz], dtype=torch.float32, device=self.device)

            quat = torch.tensor([qw, qx, qy, qz], dtype=torch.float32, device=self.device)
            n = torch.linalg.vector_norm(quat)
            if n.item() > 1e-8:
                quat = quat / n
            else:
                quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

            vcmd = torch.zeros(4, device=self.device)

            obs = torch.cat([leg_pos, wheel_vel, base_ang, vcmd, quat, self.prev_act]).unsqueeze(0)

            if self.mode == "AUTO::RUN":
                with torch.no_grad():
                    act = self.policy.act_inference(obs).squeeze()

                # smoothing (policy-space)
                self.smoothed_act = 0.2 * self.smoothed_act + 0.8 * act
                self.prev_act = act.clone()

                # convert to firmware-space (Auto 모드이므로 wheel에 30배 적용)
                self.current_act = self.convert_act(self.smoothed_act, is_auto=True)
                
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
                    
                    # Auto mode 디버깅: 1초마다 Hz와 명령 출력
                    self.get_logger().info(
                        f"[AUTO_DEBUG] Policy Hz: {self.current_policy_hz:.1f}, "
                        f"Raw act: {[f'{x:.3f}' for x in act.tolist()]}, "
                        f"Converted: {[f'{x:.3f}' for x in self.current_act.tolist()]}, "
                        f"obs_buffer: {len(self.obs_buffer)}"
                    )

            # MANUAL::RUN: output updated in manual_callback
            # STOP: handled by cmd_callback -> to_neutral()

    # ─────────────────────────────
    # Motor control loop (200 Hz)
    # ─────────────────────────────
    def ctrl_loop(self):
        next_tick = time.perf_counter()
        log_count = 0

        while self.running:
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += self.CTRL_DT

            # Send motor command only when RUN
            if self.mode in ("AUTO::RUN", "MANUAL::RUN"):
                self.send_motor_cmd(self.current_act, log=(log_count < 5))
                if log_count < 5:
                    log_count += 1

            # Always publish feedback
            self.publish_feedback()
            
            # Policy Hz는 policy_loop에서만 publish (중복 방지)

    # ─────────────────────────────
    # Helpers
    # ─────────────────────────────
    def convert_act(self, act: torch.Tensor, is_auto: bool = False) -> torch.Tensor:
        """
        Convert policy/manual-space action -> firmware-space command.
        0 input should map to NEUTRAL pose (upper/lower = PI).
        Auto 모드일 때 wheel은 30배 스케일 적용.
        """
        converted = torch.zeros(6, device=self.device)
        wheel_scale = 30.0 if is_auto else 1.0
        # wheel: 부호 반전 후 Auto일 때 30배, Manual일 때 그대로, -30~30 범위로 클램프
        converted[0] = torch.clamp(-act[0] * wheel_scale, -30.0, 30.0)  # wheel_L
        converted[1] = torch.clamp(-act[1] * wheel_scale, -30.0, 30.0)  # wheel_R
        converted[2] = act[2] + self.PI      # upper_L
        converted[3] = act[3] + self.PI      # upper_R
        converted[4] = -act[4] + self.PI     # lower_L
        converted[5] = -act[5] + self.PI     # lower_R
        return converted

    def send_motor_cmd(self, act: torch.Tensor, log: bool = False):
        pkt = (
            f"act,{act[0].item():.2f},{act[1].item():.2f},"
            f"{act[2].item():.2f},{act[3].item():.2f},"
            f"{act[4].item():.2f},{act[5].item():.2f}\n"
        )
        self.serial.write(pkt.encode())

        if log:
            self.get_logger().info(
                f"[MOTOR_CMD] Mode: {self.mode}, "
                f"Converted: [{act[0].item():.4f}, {act[1].item():.4f}, {act[2].item():.4f}, "
                f"{act[3].item():.4f}, {act[4].item():.4f}, {act[5].item():.4f}], "
                f"Packet: {pkt.strip()}"
            )

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

        (U_posL, U_posR, L_posL, L_posR, wL, wR,
         gx, gy, gz, ax, ay, az, qw, qx, qy, qz) = avg

        leg_pos = [
            U_posL / self.ENC_PER_RAD - self.PI,
            U_posR / self.ENC_PER_RAD - self.PI,
            -(L_posL / self.ENC_PER_RAD - self.PI),
            -(L_posR / self.ENC_PER_RAD - self.PI),
        ]
        wheel_vel = [-wL, -wR]
        base_ang = [gx, gy, gz]
        vcmd = [0.0, 0.0, 0.0, 0.0]
        quat = [qw, qx, qy, qz]
        prev_act_list = [self.prev_act[i].item() for i in range(6)]

        feedback_data = leg_pos + wheel_vel + base_ang + vcmd + quat + prev_act_list

        msg = Float64MultiArray()
        msg.data = feedback_data
        self.feedback_pub.publish(msg)

    def to_neutral(self):
        neutral_pkt = "act,0.0,0.0,3.14,3.14,3.14,3.14\n"
        try:
            self.serial.write(neutral_pkt.encode())
        except Exception as e:
            self.get_logger().warn(f"[Serial] failed to write neutral: {e}")
        self.get_logger().info(f"→ Returned to neutral pose: {neutral_pkt.strip()}")

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
