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
        
        # ROS2 Publisher: 센서 데이터 발행
        self.sensor_pub = self.create_publisher(
            Float64MultiArray, 
            '/Redshow/Sensor', 
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
        self.get_logger().info("센서 데이터는 /Redshow/Sensor 토픽으로 발행됩니다")
    
    def read_sensor_data(self):
        """센서 데이터를 읽고 ROS2 토픽으로 발행"""
        try:
            # 센서 데이터 읽기
            accel_x, accel_y, accel_z = self.bno.acceleration
            gyro_x, gyro_y, gyro_z = self.bno.gyro
            mag_x, mag_y, mag_z = self.bno.magnetic
            quat_i, quat_j, quat_k, quat_real = self.bno.quaternion
            
            # ROS2 메시지 생성
            # 데이터 순서: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, 
            #              mag_x, mag_y, mag_z, quat_i, quat_j, quat_k, quat_real]
            msg = Float64MultiArray()
            msg.data = [
                accel_x, accel_y, accel_z,      # 가속도 (m/s^2)
                gyro_x, gyro_y, gyro_z,         # 자이로 (rad/s)
                mag_x, mag_y, mag_z,            # 자기계 (uT)
                quat_i, quat_j, quat_k, quat_real  # 쿼터니언
            ]
            
            # 토픽 발행
            self.sensor_pub.publish(msg)
            
            # 시각화 출력 (1초마다)
            self.print_counter += 1
            if self.print_counter >= self.print_interval:
                self.print_counter = 0
                self.print_sensor_data(
                    accel_x, accel_y, accel_z,
                    gyro_x, gyro_y, gyro_z,
                    mag_x, mag_y, mag_z,
                    quat_i, quat_j, quat_k, quat_real
                )
                
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

