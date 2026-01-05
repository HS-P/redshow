# BNO085 센서 설정 가이드

## 1. 질문 답변

**Q: GUI와 Control 노드의 구조는?**
- **GUI**: 노트북에서 실행됩니다. 모델 파일 경로를 선택하고 RUN 버튼을 누르면 모델 경로가 ROS2 토픽으로 전송됩니다.
- **Control 노드**: 라즈베리파이에서 실행됩니다. `/redshow/model_path` 토픽을 구독하여 모델 파일 경로를 받고, 해당 경로에서 모델을 로드하여 실행합니다.
- 두 시스템은 동일한 ROS2 네트워크에 연결되어 있어야 하며, 모델 파일은 라즈베리파이의 해당 경로에 존재해야 합니다.

## 2. 라즈베리파이 I2C 설정

### I2C 활성화
```bash
sudo raspi-config
```
- `Interface Options` → `I2C` → `Yes` 선택

또는 직접 설정:
```bash
sudo apt-get update
sudo apt-get install -y i2c-tools python3-smbus
```

### I2C 클럭 속도 설정 (BNO085 권장: 400kHz)
`/boot/config.txt` 파일에 다음 줄 추가:
```bash
sudo nano /boot/config.txt
```

다음 줄 추가:
```
dtparam=i2c_arm_baudrate=400000
```

재부팅:
```bash
sudo reboot
```

### I2C 장치 확인
재부팅 후 다음 명령어로 센서가 인식되는지 확인:
```bash
sudo i2cdetect -y 1
```
BNO085는 보통 주소 `0x4A` 또는 `0x4B`에 나타납니다.

## 3. 필수 라이브러리 설치

### Python 패키지 설치
라즈베리파이에서 다음 명령어를 실행하세요:

```bash
# adafruit-blinka (Raspberry Pi에서 CircuitPython 라이브러리 사용을 위한 호환성 레이어)
/usr/bin/python3 -m pip install adafruit-blinka

# adafruit-circuitpython-bno08x (BNO085 센서 라이브러리)
/usr/bin/python3 -m pip install adafruit-circuitpython-bno08x
```

### 설치 확인
```bash
/usr/bin/python3 -c "import board; import busio; from adafruit_bno08x.i2c import BNO08X_I2C; print('✓ 라이브러리 설치 완료')"
```

## 4. 하드웨어 연결

### BNO085 센서 연결 (I2C)
라즈베리파이와 BNO085 센서를 다음과 같이 연결하세요:

- **라즈베리파이 3.3V** → **BNO085 VIN** (빨간선)
- **라즈베리파이 GND** → **BNO085 GND** (검은선)
- **라즈베리파이 SCL (GPIO 3)** → **BNO085 SCL** (노란선)
- **라즈베리파이 SDA (GPIO 2)** → **BNO085 SDA** (파란선)

## 5. 테스트 노드 실행

### 빌드
```bash
cd /home/hansol/redshow
colcon build --packages-select redshow_control
source install/setup.bash
```

### 테스트 노드 실행
```bash
ros2 run redshow_control bno085_test_node
```

### 토픽 확인
다른 터미널에서:
```bash
# 센서 데이터 확인
ros2 topic echo /Redshow/Sensor

# 상태 확인
ros2 topic echo /Redshow/Sensor/Status

# 토픽 목록 확인
ros2 topic list
```

## 6. ROS2 토픽 구조

### 센서 데이터 토픽
- **토픽 이름**: `/Redshow/Sensor`
- **메시지 타입**: `std_msgs/Float64MultiArray`
- **데이터 순서**: 
  ```
  [accel_x, accel_y, accel_z,      # 가속도 (m/s²)
   gyro_x, gyro_y, gyro_z,         # 자이로 (rad/s)
   mag_x, mag_y, mag_z,            # 자기계 (uT)
   quat_i, quat_j, quat_k, quat_real]  # 쿼터니언
  ```
- **발행 주기**: 50Hz (0.02초)

### 상태 토픽
- **토픽 이름**: `/Redshow/Sensor/Status`
- **메시지 타입**: `std_msgs/String`
- **값**: `"READY"` (정상) 또는 `"ERROR: ..."` (오류)

### 제어 토픽 (향후 사용)
- **토픽 이름**: `/Redshow/Control`
- **용도**: 제어 명령 전송

### 액션 토픽 (향후 사용)
- **토픽 이름**: `/Redshow/Action`
- **용도**: 액션 명령 전송

## 7. 문제 해결

### 센서가 인식되지 않는 경우
1. I2C 연결 확인 (SCL, SDA, 전원, GND)
2. I2C 활성화 확인: `sudo i2cdetect -y 1`
3. 권한 확인: `sudo usermod -a -G i2c $USER` (로그아웃 후 재로그인 필요)

### 라이브러리 import 오류
```bash
# pip 업그레이드
/usr/bin/python3 -m pip install --upgrade pip

# 라이브러리 재설치
/usr/bin/python3 -m pip uninstall adafruit-blinka adafruit-circuitpython-bno08x
/usr/bin/python3 -m pip install adafruit-blinka adafruit-circuitpython-bno08x
```

### I2C 속도 문제
`/boot/config.txt`에서 `dtparam=i2c_arm_baudrate=400000` 확인 후 재부팅

