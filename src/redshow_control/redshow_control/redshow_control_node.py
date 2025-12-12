#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
import torch, time, threading
from collections import deque
from rsl_rl.modules import ActorCritic
import serial

class BerkeleyControlNode(Node):
    def __init__(self):
        super().__init__('berkeley_control_node')

        # ────────────── ROS2 I/O 설정 ──────────────
        self.cmd_sub = self.create_subscription(String, 'redshow/cmd', self.cmd_callback, 10)
        self.manual_sub = self.create_subscription(Float64MultiArray, 'redshow/joint_cmd', self.manual_callback, 10)

        # ────────────── 정책 로드 ──────────────
        import argparse, os
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy_name", type=str, required=True,
                            help="")
        args, _ = parser.parse_known_args()

        pt_path = os.path.join(os.path.dirname(__file__), args.policy_name)
        self.policy = ActorCritic(
            num_actor_obs=23, num_critic_obs=23, num_actions=6,
            actor_hidden_dims=[128,128,128], critic_hidden_dims=[128,128,128],
            activation="elu", init_noise_std=0.0
        )
        ckpt = torch.load(pt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        actor_state = {k:v for k,v in state_dict.items() if k.startswith("actor")}
        self.policy.load_state_dict(actor_state, strict=False)
        self.policy.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.get_logger().info(f"✓ Policy loaded: {args.policy_name}")

        # ────────────── 제어 관련 변수 ──────────────
        self.ENC_PER_RAD = 651.89865
        self.PI = 3.142
        self.ACTOR_DT = 1.0 / 50.0
        self.obs_buffer = deque(maxlen=4)
        self.prev_act = torch.zeros(6, device=self.device)
        self.smoothed_act = torch.zeros(6, device=self.device)

        # 시리얼 통신
        self.serial = serial.Serial('/dev/ttyACM0', 1000000, timeout=None)
        self.mode = "STOP"  # MANUAL::RUN, AUTO::RUN, 등

        # 스레드
        self.running = True
        self.obs_thread = threading.Thread(target=self.obs_loop, daemon=True)
        self.obs_thread.start()
        self.ctrl_thread = threading.Thread(target=self.ctrl_loop, daemon=True)
        self.ctrl_thread.start()

    # ──────────────────────────────────────────────
    # ROS2 콜백
    # ──────────────────────────────────────────────
    def cmd_callback(self, msg):
        self.mode = msg.data.strip()
        self.get_logger().info(f"[CMD] Mode changed: {self.mode}")
        if "STOP" in self.mode:
            self.to_neutral()

    def manual_callback(self, msg):
        self.manual_act = torch.tensor(msg.data, dtype=torch.float32, device=self.device)

    # ──────────────────────────────────────────────
    # 센서 수집 루프 (200Hz)
    # ──────────────────────────────────────────────
    def obs_loop(self):
        while self.running:
            try:
                line = self.serial.read_until(b"\n")
                if not line or not line.startswith(b"fb,"):
                    continue
                parts = line.decode(errors="ignore").strip().split(",")
                if len(parts) != 17:
                    continue
                fb = tuple(float(x) for x in parts[1:])
                self.obs_buffer.append(fb)
            except Exception as e:
                self.get_logger().warn(f"[Serial] {e}")
            time.sleep(1/200.0)

    # ──────────────────────────────────────────────
    # 제어 루프 (50Hz)
    # ──────────────────────────────────────────────
    def ctrl_loop(self):
        next_tick = time.perf_counter()
        while self.running:
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += self.ACTOR_DT
            if len(self.obs_buffer) < 4:
                continue

            avg = [sum(s[i] for s in self.obs_buffer)/4 for i in range(16)]
            (U_posL,U_posR,L_posL,L_posR,wL,wR,gx,gy,gz,ax,ay,az,qw,qx,qy,qz) = avg

            # 관측 구성
            leg_pos = torch.tensor([
                U_posL/self.ENC_PER_RAD - self.PI,
                U_posR/self.ENC_PER_RAD - self.PI,
                -(L_posL/self.ENC_PER_RAD - self.PI),
                -(L_posR/self.ENC_PER_RAD - self.PI)
            ], dtype=torch.float32, device=self.device)
            wheel_vel = torch.tensor([-wL, -wR], dtype=torch.float32, device=self.device)
            base_ang = torch.tensor([gx, gy, gz], dtype=torch.float32, device=self.device)
            quat = torch.tensor([qw,qx,qy,qz], dtype=torch.float32, device=self.device)
            quat = quat/torch.linalg.vector_norm(quat)
            vcmd = torch.zeros(4, device=self.device)
            obs = torch.cat([leg_pos, wheel_vel, base_ang, vcmd, quat, self.prev_act]).unsqueeze(0)

            # 모드별 처리
            if self.mode == "AUTO::RUN":
                with torch.no_grad():
                    act = self.policy.act_inference(obs).squeeze()
                self.smoothed_act = 0.2*self.smoothed_act + 0.8*act
                self.prev_act = act.clone()
                self.send_motor_cmd(self.smoothed_act)
            elif self.mode == "MANUAL::RUN":
                if hasattr(self, "manual_act"):
                    self.send_motor_cmd(self.manual_act)
            elif "STOP" in self.mode:
                self.to_neutral()

    # ──────────────────────────────────────────────
    # 보조 함수
    # ──────────────────────────────────────────────
    def send_motor_cmd(self, act):
        omega_L = -30.0 * act[0].item()
        omega_R = -30.0 * act[1].item()
        U_L = act[2].item() + self.PI
        U_R = act[3].item() + self.PI
        L_L = -act[4].item() + self.PI
        L_R = -act[5].item() + self.PI
        pkt = f"act,{omega_L:.2f},{omega_R:.2f},{U_L:.2f},{U_R:.2f},{L_L:.2f},{L_R:.2f}\n"
        self.serial.write(pkt.encode())

    def to_neutral(self):
        self.serial.write("act,0.0,0.0,3.14,3.14,3.14,3.14\n".encode())
        self.get_logger().info("→ Returned to neutral pose")

    def destroy_node(self):
        self.running = False
        self.serial.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = BerkeleyControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
