#!/usr/bin/env python3
"""
BNO085 IMU 센서 테스트 노드
라즈베리파이에서 I2C를 통해 BNO085 센서 데이터를 읽고 ROS2 토픽으로 발행합니다.
"""

import time
import sys
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

try:
    import board
    import busio
    from adafruit_bno08x import (
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_MAGNETOMETER,
        BNO_REPORT_ROTATION_VECTOR,
    )
    from adafruit_bno08x.i2c import BNO08X_I2C
except ImportError as e:
    print(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
    print("다음 명령어로 설치하세요:")
    print("  /usr/bin/python3 -m pip install adafruit-circuitpython-bno08x adafruit-blinka")
    sys.exit(1)


class BNO085Node(Node):
    def __init__(self):
        super().__init__('bno085_node')
        
        # YAW unwrapping을 위한 이전 yaw 값 저장
        self.prev_yaw = None
        
        # ROS2 Publisher: 개별 센서 데이터 발행
        self.accelerometer_pub = self.create_publisher(
            Float64MultiArray,
            '/Redshow/Sensor/accelerometer',
            10
        )
        self.gyroscope_pub = self.create_publisher(
            Float64MultiArray,
            '/Redshow/Sensor/gyroscope',
            10
        )
        self.magnetometer_pub = self.create_publisher(
            Float64MultiArray,
            '/Redshow/Sensor/magnetometer',
            10
        )
        self.quaternion_pub = self.create_publisher(
            Float64MultiArray,
            '/Redshow/Sensor/quaternion',
            10
        )
        self.rpy_pub = self.create_publisher(
            Float64MultiArray,
            '/Redshow/Sensor/rpy',
            10
        )
        
        # ROS2 Publisher: 상태 메시지 발행
        self.status_pub = self.create_publisher(
            String,
            '/Redshow/Sensor/Status',
            10
        )
        
        # I2C 초기화 및 센서 설정
        self.get_logger().info("BNO085 센서 초기화 중...")
        
        try:
            # I2C 버스 초기화 (400kHz 권장)
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
            self.bno = BNO08X_I2C(self.i2c)
            
            # 센서 기능 활성화
            self.bno.enable_feature(BNO_REPORT_ACCELEROMETER)
            self.bno.enable_feature(BNO_REPORT_GYROSCOPE)
            self.bno.enable_feature(BNO_REPORT_MAGNETOMETER)
            self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            
            self.get_logger().info("✓ BNO085 센서 초기화 완료")
            
            # 상태 메시지 발행
            status_msg = String()
            status_msg.data = "READY"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"✗ BNO085 센서 초기화 실패: {e}")
            status_msg = String()
            status_msg.data = f"ERROR: {str(e)}"
            self.status_pub.publish(status_msg)
            raise
        
        # 타이머 설정 (200Hz로 센서 데이터 읽기)
        self.timer = self.create_timer(0.005, self.read_sensor_data)  # 200Hz = 5ms
        
        # 시각화를 위한 카운터
        self.print_counter = 0
        self.print_interval = 200  # 200번마다 출력 (약 1초마다, 200Hz 기준)
        
        self.get_logger().info("BNO085 노드 시작됨 (200Hz)")
        self.get_logger().info("개별 센서 데이터는 /Redshow/Sensor/* 토픽으로 발행됩니다")
    
    def quaternion_to_rpy(self, quat_i, quat_j, quat_k, quat_real):
        """
        Quaternion (i, j, k, real)을 Roll, Pitch, Yaw (RPY)로 변환
        YAW 각도 wrapping 문제 해결 (unwrapping 적용)
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (quat_real * quat_i + quat_j * quat_k)
        cosr_cosp = 1 - 2 * (quat_i * quat_i + quat_j * quat_j)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (quat_real * quat_j - quat_k * quat_i)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (quat_real * quat_k + quat_i * quat_j)
        cosy_cosp = 1 - 2 * (quat_j * quat_j + quat_k * quat_k)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # YAW unwrapping: 이전 값과의 차이가 π를 넘어가면 2π 보정
        if self.prev_yaw is not None:
            yaw_diff = yaw - self.prev_yaw
            # 차이가 π보다 크면 -2π, -π보다 작으면 +2π
            if yaw_diff > math.pi:
                yaw = yaw - 2 * math.pi
            elif yaw_diff < -math.pi:
                yaw = yaw + 2 * math.pi
        
        # 이전 yaw 값 업데이트
        self.prev_yaw = yaw
        
        return [roll, pitch, yaw]
    
    def read_sensor_data(self):
        """센서 데이터를 읽고 ROS2 토픽으로 발행"""
        try:
            # 센서 데이터 읽기
            accel_x, accel_y, accel_z = self.bno.acceleration
            gyro_x, gyro_y, gyro_z = self.bno.gyro
            mag_x, mag_y, mag_z = self.bno.magnetic
            quat_i, quat_j, quat_k, quat_real = self.bno.quaternion
            
            # Quaternion을 RPY로 변환
            rpy = self.quaternion_to_rpy(quat_i, quat_j, quat_k, quat_real)
            
            # 개별 센서 데이터 토픽 발행
            # Accelerometer
            msg_accel = Float64MultiArray()
            msg_accel.data = [accel_x, accel_y, accel_z]
            self.accelerometer_pub.publish(msg_accel)
            
            # Gyroscope
            msg_gyro = Float64MultiArray()
            msg_gyro.data = [gyro_x, gyro_y, gyro_z]
            self.gyroscope_pub.publish(msg_gyro)
            
            # Magnetometer
            msg_mag = Float64MultiArray()
            msg_mag.data = [mag_x, mag_y, mag_z]
            self.magnetometer_pub.publish(msg_mag)
            
            # Quaternion
            msg_quat = Float64MultiArray()
            msg_quat.data = [quat_i, quat_j, quat_k, quat_real]
            self.quaternion_pub.publish(msg_quat)
            
            # RPY - Degree로 변환하여 RPY 순서대로 발행
            msg_rpy = Float64MultiArray()
            # RPY 순서: [roll, pitch, yaw] = [rpy[0], rpy[1], rpy[2]]
            # Radian을 Degree로 변환 (180 / π)
            DEG_PER_RAD = 180.0 / math.pi
            msg_rpy.data = [
                rpy[0] * DEG_PER_RAD,  # Roll (degree)
                rpy[1] * DEG_PER_RAD,  # Pitch (degree)
                rpy[2] * DEG_PER_RAD   # Yaw (degree)
            ]
            self.rpy_pub.publish(msg_rpy)
            
            # 시각화 출력 제거 (GUI에서 확인 가능)
            # self.print_counter += 1
            # if self.print_counter >= self.print_interval:
            #     self.print_counter = 0
            #     self.print_sensor_data(
            #         accel_x, accel_y, accel_z,
            #         gyro_x, gyro_y, gyro_z,
            #         mag_x, mag_y, mag_z,
            #         quat_i, quat_j, quat_k, quat_real
            #     )
                
        except Exception as e:
            self.get_logger().error(f"센서 데이터 읽기 오류: {e}", exc_info=True)
            time.sleep(0.1)
    
    def print_sensor_data(self, accel_x, accel_y, accel_z,
                          gyro_x, gyro_y, gyro_z,
                          mag_x, mag_y, mag_z,
                          quat_i, quat_j, quat_k, quat_real):
        """센서 데이터를 시각화하여 출력"""
        print("\n" + "=" * 80)
        print("BNO085 센서 데이터")
        print("=" * 80)
        
        print(f"\n📊 가속도 (Accelerometer) - m/s²:")
        print(f"   X: {accel_x:8.4f}  Y: {accel_y:8.4f}  Z: {accel_z:8.4f}")
        
        print(f"\n🌀 자이로스코프 (Gyroscope) - rad/s:")
        print(f"   X: {gyro_x:8.4f}  Y: {gyro_y:8.4f}  Z: {gyro_z:8.4f}")
        
        print(f"\n🧲 자기계 (Magnetometer) - uT:")
        print(f"   X: {mag_x:8.4f}  Y: {mag_y:8.4f}  Z: {mag_z:8.4f}")
        
        print(f"\n🔄 회전 벡터 (Rotation Vector) - Quaternion:")
        print(f"   I: {quat_i:8.4f}  J: {quat_j:8.4f}  K: {quat_k:8.4f}  Real: {quat_real:8.4f}")
        
        # 쿼터니언으로부터 오일러 각 계산 (간단한 변환)
        # Roll, Pitch, Yaw 계산
        roll = math.atan2(
            2 * (quat_real * quat_i + quat_j * quat_k),
            1 - 2 * (quat_i * quat_i + quat_j * quat_j)
        )
        pitch = math.asin(2 * (quat_real * quat_j - quat_k * quat_i))
        yaw = math.atan2(
            2 * (quat_real * quat_k + quat_i * quat_j),
            1 - 2 * (quat_j * quat_j + quat_k * quat_k)
        )
        
        print(f"\n📐 오일러 각 (Euler Angles) - rad:")
        print(f"   Roll:  {roll:8.4f}  Pitch: {pitch:8.4f}  Yaw:   {yaw:8.4f}")
        print(f"   Roll:  {math.degrees(roll):7.2f}°  Pitch: {math.degrees(pitch):7.2f}°  Yaw:   {math.degrees(yaw):7.2f}°")
        
        print("=" * 80 + "\n")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = BNO085Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"노드 실행 오류: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

