#!/usr/bin/env python3
"""
A-RMA Adaptation Module Node
- 70개 observation history 수집 (23차원 * 70 = 1610차원)
- Adaptation Module을 거쳐서 EX_OBS (8차원) 생성
- /Redshow/Observation/ex_obs 토픽으로 publish
"""

import os
import time
import threading
from collections import deque
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class AdaptationModuleNode(Node):
    def __init__(self):
        super().__init__('adaptation_module_node')
        
        # ────────────── ROS2 I/O ──────────────
        # Observation 토픽 구독 (23차원)
        self.leg_position_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/leg_position', 
            self.leg_position_callback, 10
        )
        self.wheel_velocity_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/wheel_velocity', 
            self.wheel_velocity_callback, 10
        )
        self.base_ang_vel_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/base_ang_vel', 
            self.base_ang_vel_callback, 10
        )
        self.velocity_commands_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/velocity_commands', 
            self.velocity_commands_callback, 10
        )
        self.base_quat_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/base_quat', 
            self.base_quat_callback, 10
        )
        self.actions_sub = self.create_subscription(
            Float64MultiArray, '/Redshow/Observation/actions', 
            self.actions_callback, 10
        )
        
        # EX_OBS Publisher (8차원)
        self.ex_obs_pub = self.create_publisher(
            Float64MultiArray, '/Redshow/Observation/ex_obs', 10
        )
        
        # ────────────── Device ──────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"[INIT] Using device: {self.device}")
        
        # ────────────── Observation History ──────────────
        self.obs_dim = 23  # leg_position(4) + wheel_velocity(2) + base_ang_vel(3) + velocity_commands(4) + base_quat(4) + actions(6)
        self.history_length = 70  # 70 timesteps
        self.obs_history = deque(maxlen=self.history_length)
        
        # 현재 observation 버퍼 (각 토픽에서 받은 데이터 저장)
        self.current_obs = {
            'leg_position': None,
            'wheel_velocity': None,
            'base_ang_vel': None,
            'velocity_commands': None,
            'base_quat': None,
            'actions': None,
        }
        self.obs_lock = threading.Lock()
        
        # ────────────── Adaptation Module ──────────────
        self.adaptation_module = None
        self.load_adaptation_module()
        
        # ────────────── Processing Thread ──────────────
        self.running = True
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.get_logger().info("✓ Adaptation Module Node started")
        self.get_logger().info(f"  History length: {self.history_length} timesteps")
        self.get_logger().info(f"  Observation dimension: {self.obs_dim}")
        self.get_logger().info(f"  EX_OBS dimension: 8")
    
    def load_adaptation_module(self):
        """Adaptation Module 모델 로드"""
        # asset_arma 경로 찾기
        def find_asset_arma_path():
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'asset_arma'),
                '/home/pi/ros2_ws/src/asset_arma',
                '/home/sol/redshow/src/asset_arma',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            return None
        
        asset_arma_path = find_asset_arma_path()
        if asset_arma_path is None:
            self.get_logger().error("[ADAPTATION] asset_arma directory not found!")
            return
        
        model_path = os.path.join(asset_arma_path, 'adaptive_module_400.pt')
        if not os.path.exists(model_path):
            self.get_logger().error(f"[ADAPTATION] Model file not found: {model_path}")
            return
        
        try:
            # 모델 가중치 로드
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Adaptation Module 구조 정의 (checkpoint 구조에 맞게)
            # Input: 23차원 observation -> embed -> conv2d layers -> fc layers -> 8차원 output
            class AdaptationModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # Embedding: 23 -> 64
                    self.embed = torch.nn.Linear(23, 64)
                    # Conv2d layers (temporal convolution using 2D conv)
                    # weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                    self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=(1, 8), padding=(0, 3))
                    self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(1, 5), padding=(0, 2))
                    self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(1, 5), padding=(0, 2))
                    # FC layers
                    self.fc1 = torch.nn.Linear(64, 128)
                    self.fc2 = torch.nn.Linear(128, 8)  # Output: 8차원
            
                def forward(self, obs_history):
                    # obs_history: [batch, 70, 23]
                    batch_size = obs_history.shape[0]
                    # Embedding: [batch, 70, 23] -> [batch, 70, 64]
                    x = self.embed(obs_history)
                    # Reshape for Conv2d: [batch, 70, 64] -> [batch, 64, 1, 70] (channel first)
                    x = x.transpose(1, 2).unsqueeze(2)  # [batch, 64, 1, 70]
                    # Conv2d layers
                    x = torch.relu(self.conv1(x))
                    x = torch.relu(self.conv2(x))
                    x = torch.relu(self.conv3(x))
                    # Global average pooling: [batch, 64, 1, ?] -> [batch, 64]
                    x = x.mean(dim=(2, 3))  # [batch, 64]
                    # FC layers
                    x = torch.relu(self.fc1(x))
                    x = self.fc2(x)  # [batch, 8]
                    return x
            
            self.adaptation_module = AdaptationModule()
            self.adaptation_module.load_state_dict(state_dict)
            self.adaptation_module.to(self.device)
            self.adaptation_module.eval()
            
            self.get_logger().info(f"✓ Adaptation Module loaded from: {model_path}")
        except Exception as e:
            self.get_logger().error(f"[ADAPTATION] Failed to load model: {e}")
            import traceback
            self.get_logger().error(f"[ADAPTATION] Traceback: {traceback.format_exc()}")
    
    def leg_position_callback(self, msg):
        with self.obs_lock:
            self.current_obs['leg_position'] = list(msg.data)
    
    def wheel_velocity_callback(self, msg):
        with self.obs_lock:
            self.current_obs['wheel_velocity'] = list(msg.data)
    
    def base_ang_vel_callback(self, msg):
        with self.obs_lock:
            self.current_obs['base_ang_vel'] = list(msg.data)
    
    def velocity_commands_callback(self, msg):
        with self.obs_lock:
            self.current_obs['velocity_commands'] = list(msg.data)
    
    def base_quat_callback(self, msg):
        with self.obs_lock:
            self.current_obs['base_quat'] = list(msg.data)
    
    def actions_callback(self, msg):
        with self.obs_lock:
            self.current_obs['actions'] = list(msg.data)
    
    def assemble_observation(self):
        """현재 observation 버퍼에서 23차원 observation 조립"""
        with self.obs_lock:
            if any(v is None for v in self.current_obs.values()):
                return None
            
            # Observation 순서: leg_position(4) + wheel_velocity(2) + base_ang_vel(3) + 
            #                   velocity_commands(4) + base_quat(4) + actions(6) = 23
            obs = (
                self.current_obs['leg_position'] +
                self.current_obs['wheel_velocity'] +
                self.current_obs['base_ang_vel'] +
                self.current_obs['velocity_commands'] +
                self.current_obs['base_quat'] +
                self.current_obs['actions']
            )
            
            if len(obs) != self.obs_dim:
                self.get_logger().warn(f"[ADAPTATION] Observation dimension mismatch: {len(obs)} != {self.obs_dim}")
                return None
            
            return obs
    
    def processing_loop(self):
        """Adaptation Module 처리 루프 (50Hz)"""
        rate = 50.0  # 50Hz
        dt = 1.0 / rate
        
        while self.running:
            start_time = time.perf_counter()
            
            # Observation 조립
            obs = self.assemble_observation()
            if obs is not None:
                # History에 추가
                self.obs_history.append(obs)
                
                # History가 충분히 쌓였으면 EX_OBS 생성
                if len(self.obs_history) >= self.history_length and self.adaptation_module is not None:
                    try:
                        # History를 tensor로 변환: [70, 23]
                        obs_history_tensor = torch.tensor(
                            list(self.obs_history), 
                            dtype=torch.float32, 
                            device=self.device
                        ).unsqueeze(0)  # [1, 70, 23] (batch dimension 추가)
                        
                        # Adaptation Module inference
                        with torch.no_grad():
                            ex_obs = self.adaptation_module(obs_history_tensor)
                            ex_obs = ex_obs.squeeze(0)  # [8]
                        
                        # EX_OBS publish
                        msg = Float64MultiArray()
                        msg.data = [ex_obs[i].item() for i in range(8)]
                        self.ex_obs_pub.publish(msg)
                        
                    except Exception as e:
                        self.get_logger().error(f"[ADAPTATION] Processing error: {e}")
                        import traceback
                        self.get_logger().error(f"[ADAPTATION] Traceback: {traceback.format_exc()}")
            
            # Rate control
            elapsed = time.perf_counter() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def destroy_node(self):
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AdaptationModuleNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


