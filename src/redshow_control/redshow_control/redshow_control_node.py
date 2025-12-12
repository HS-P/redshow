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

from rsl_rl.modules import ActorCritic


class BerkeleyControlNode(Node):
    def __init__(self):
        super().__init__('berkeley_control_node')

        # ────────────── ROS2 I/O ──────────────
        self.cmd_sub = self.create_subscription(String, 'redshow/cmd', self.cmd_callback, 10)
        self.manual_sub = self.create_subscription(Float64MultiArray, 'redshow/joint_cmd', self.manual_callback, 10)
        self.feedback_pub = self.create_publisher(Float64MultiArray, 'redshow/feedback', 10)

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

        self.policy = ActorCritic(
            num_actor_obs=23, num_critic_obs=23, num_actions=6,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128],
            activation="elu",
            init_noise_std=0.0
        )

        if not os.path.exists(pt_path):
            self.get_logger().error(f"Policy file not found: {pt_path}")
            raise FileNotFoundError(f"Policy file not found: {pt_path}")

        ckpt = torch.load(pt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor")}
        self.policy.load_state_dict(actor_state, strict=False)
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info(f"✓ Policy loaded: {pt_path} (from parameter: {policy_name})")

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

        # current_act MUST ALWAYS be "converted act" that matches firmware expectations:
        # [wheel_L, wheel_R, upper_L, upper_R, lower_L, lower_R] with PI offsets
        self.current_act = self.convert_act(torch.zeros(6, device=self.device))

        # Mode state
        self.mode = "STOP"  # "MANUAL::RUN", "AUTO::RUN", "MANUAL::STOP", "AUTO::STOP", etc.

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
                
                # Auto mode 디버깅: 1초마다 Hz와 명령 출력
                self.auto_debug_hz_counter += 1
                current_time_debug = time.time()
                elapsed = current_time_debug - self.auto_debug_hz_start_time
                if elapsed >= 1.0:
                    hz = self.auto_debug_hz_counter / elapsed
                    self.get_logger().info(
                        f"[AUTO_DEBUG] Policy Hz: {hz:.1f}, "
                        f"Raw act: {[f'{x:.3f}' for x in act.tolist()]}, "
                        f"Converted: {[f'{x:.3f}' for x in self.current_act.tolist()]}, "
                        f"obs_buffer: {len(self.obs_buffer)}"
                    )
                    self.auto_debug_hz_counter = 0
                    self.auto_debug_hz_start_time = current_time_debug

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
        if len(self.obs_buffer) >= 1:
            avg = [sum(s[i] for s in self.obs_buffer) / len(self.obs_buffer) for i in range(16)]
        else:
            avg = [0.0] * 16

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
