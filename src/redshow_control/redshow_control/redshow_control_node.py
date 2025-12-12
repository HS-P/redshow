#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
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
        # 피드백 데이터 publish
        self.feedback_pub = self.create_publisher(Float64MultiArray, 'redshow/feedback', 10)

        # ────────────── 정책 로드 ──────────────
        import os
        # ROS2 파라미터로 정책 파일명 받기 (기본값: model_4800.pt)
        self.declare_parameter('policy_name', 'model_4800.pt')
        policy_name = self.get_parameter('policy_name').get_parameter_value().string_value

        # 경로 처리: 절대 경로면 그대로 사용, 아니면 패키지 디렉토리 기준으로
        if os.path.isabs(policy_name):
            pt_path = policy_name
        elif os.path.exists(policy_name):
            # 상대 경로로 파일이 존재하면 그대로 사용
            pt_path = policy_name
        else:
            # 파일명만 입력된 경우 패키지 디렉토리에서 찾기
            pt_path = os.path.join(os.path.dirname(__file__), policy_name)
        self.policy = ActorCritic(
            num_actor_obs=23, num_critic_obs=23, num_actions=6,
            actor_hidden_dims=[128,128,128], critic_hidden_dims=[128,128,128],
            activation="elu", init_noise_std=0.0
        )
        if not os.path.exists(pt_path):
            self.get_logger().error(f"Policy file not found: {pt_path}")
            raise FileNotFoundError(f"Policy file not found: {pt_path}")
        
        ckpt = torch.load(pt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        actor_state = {k:v for k,v in state_dict.items() if k.startswith("actor")}
        self.policy.load_state_dict(actor_state, strict=False)
        self.policy.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.get_logger().info(f"✓ Policy loaded: {pt_path} (from parameter: {policy_name})")

        # ────────────── 제어 관련 변수 ──────────────
        self.ENC_PER_RAD = 651.89865
        self.PI = 3.142
        self.ACTOR_DT = 1.0 / 50.0  # Policy 추론 주기 (50Hz)
        self.CTRL_DT = 1.0 / 200.0  # 모터 제어 주기 (200Hz)
        self.obs_buffer = deque(maxlen=4)
        self.prev_act = torch.zeros(6, device=self.device)
        self.smoothed_act = torch.zeros(6, device=self.device)
        self.manual_act = torch.zeros(6, device=self.device)  # Manual 액션 초기화
        self.current_act = torch.zeros(6, device=self.device)  # 현재 전송할 액션 (200Hz로 전송)

        # 시리얼 통신
        self.serial = serial.Serial('/dev/ttyACM0', 1000000, timeout=None)
        self.mode = "STOP"  # MANUAL::RUN, AUTO::RUN, 등

        # 스레드
        self.running = True
        self.obs_thread = threading.Thread(target=self.obs_loop, daemon=True)
        self.obs_thread.start()
        self.policy_thread = threading.Thread(target=self.policy_loop, daemon=True)  # Policy 추론 (50Hz)
        self.policy_thread.start()
        self.ctrl_thread = threading.Thread(target=self.ctrl_loop, daemon=True)  # 모터 제어 (200Hz)
        self.ctrl_thread.start()
        # ROS2 스핀 스레드 (콜백 처리용)
        self.spin_thread = threading.Thread(target=self.ros_spin, daemon=True)
        self.spin_thread.start()
    
    def ros_spin(self):
        """ROS2 콜백을 처리하기 위한 스핀 스레드"""
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        while self.running and rclpy.ok():
            executor.spin_once(timeout_sec=0.01)

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
        self.get_logger().info(f"[MANUAL] Received joint_cmd: {msg.data}")

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
    # Policy 추론 루프 (50Hz)
    # ──────────────────────────────────────────────
    def policy_loop(self):
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
                # Policy 출력(정규화된 값)을 실제 모터 명령으로 변환
                self.current_act = self.convert_auto_act(self.smoothed_act)
            elif self.mode == "MANUAL::RUN":
                # Manual 액션: GUI에서 받은 값을 그대로 사용 (변환 없이)
                self.current_act = self.manual_act.clone()
            elif "STOP" in self.mode:
                self.current_act = torch.zeros(6, device=self.device)
                self.to_neutral()

    # ──────────────────────────────────────────────
    # 모터 제어 루프 (200Hz)
    # ──────────────────────────────────────────────
    def ctrl_loop(self):
        next_tick = time.perf_counter()
        while self.running:
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += self.CTRL_DT
            
            # 현재 액션을 모터로 전송
            if self.mode == "AUTO::RUN" or self.mode == "MANUAL::RUN":
                self.send_motor_cmd(self.current_act)
            
            # 피드백 데이터 publish (200Hz)
            if len(self.obs_buffer) >= 4:
                self.publish_feedback()

    # ──────────────────────────────────────────────
    # 보조 함수
    # ──────────────────────────────────────────────
    def convert_manual_act(self, act):
        """Manual 액션을 Auto와 같은 형식으로 변환"""
        # GUI에서 받은 값 [-10, 10] 범위를 [-1, 1]로 정규화 후 Auto와 같은 변환 적용
        normalized = act / 10.0  # [-10, 10] -> [-1, 1]
        converted = torch.zeros(6, device=self.device)
        # Auto 모드와 동일한 변환: send_motor_cmd_auto와 동일
        converted[0] = -30.0 * normalized[0]  # wheel_L
        converted[1] = -30.0 * normalized[1]  # wheel_R
        converted[2] = normalized[2] + self.PI  # upper_L
        converted[3] = normalized[3] + self.PI  # upper_R
        converted[4] = -normalized[4] + self.PI  # lower_L
        converted[5] = -normalized[5] + self.PI  # lower_R
        return converted
    
    def send_motor_cmd(self, act):
        """모터 명령 전송 (변환된 값 사용)"""
        # act는 이미 변환된 값 (omega, angle 형식)
        pkt = f"act,{act[0].item():.2f},{act[1].item():.2f},{act[2].item():.2f},{act[3].item():.2f},{act[4].item():.2f},{act[5].item():.2f}\n"
        self.serial.write(pkt.encode())
    
    def publish_feedback(self):
        """피드백 데이터 publish (관측 데이터)"""
        if len(self.obs_buffer) < 4:
            return
        
        avg = [sum(s[i] for s in self.obs_buffer)/4 for i in range(16)]
        (U_posL,U_posR,L_posL,L_posR,wL,wR,gx,gy,gz,ax,ay,az,qw,qx,qy,qz) = avg
        
        # 피드백 데이터 구성 (23차원: leg_pos(4) + wheel_vel(2) + base_ang(3) + vcmd(4) + quat(4) + prev_act(6))
        # 관측과 동일한 형식으로 전송
        leg_pos = [
            U_posL/self.ENC_PER_RAD - self.PI,
            U_posR/self.ENC_PER_RAD - self.PI,
            -(L_posL/self.ENC_PER_RAD - self.PI),
            -(L_posR/self.ENC_PER_RAD - self.PI)
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
        # 스핀은 스레드에서 처리하므로 여기서는 대기만
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
