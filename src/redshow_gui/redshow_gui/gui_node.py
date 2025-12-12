"""
실시간 모니터링 GUI - PyQt5 + ROS2 (Vanilla, 23D)
redshow/cmd (String), redshow/joint_cmd (Float64MultiArray)
"""
import sys
import torch
import numpy as np
import threading
from collections import deque

from PyQt5.QtWidgets import (
    QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
    QPushButton,QLabel,QTextEdit,QTableWidget,QTableWidgetItem,
    QGroupBox,QGridLayout,QDoubleSpinBox,QScrollArea
)
from PyQt5.QtCore import QTimer,Qt
from PyQt5.QtGui import QFont,QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import String,Float64MultiArray


# =========================
# ROS2 NODE
# =========================
class ROS2Node(Node):
    def __init__(self):
        super().__init__('gui_node')
        self.cmd_pub = self.create_publisher(String,'redshow/cmd',10)
        self.joint_pub = self.create_publisher(Float64MultiArray,'redshow/joint_cmd',10)

    def publish_cmd(self,cmd):
        msg=String()
        msg.data=cmd
        self.cmd_pub.publish(msg)

    def publish_joint_cmd(self,joints):
        msg=Float64MultiArray()
        msg.data=joints
        self.joint_pub.publish(msg)


# =========================
# MAIN GUI
# =========================
class MonitorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOL Monitor")
        self.setGeometry(100,100,1600,1000)

        # ---- state ----
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_mode=None
        self.is_running=False
        self.step=0

        # ---- buffers ----
        self.prev_act=torch.zeros(6,device=self.device)
        self.smoothed_act=torch.zeros(6,device=self.device)
        
        # 그래프용 데이터 버퍼 (최근 500개 샘플)
        self.max_history=500
        self.time_buffer=deque(maxlen=self.max_history)
        self.obs_buffer={}  # 각 observation별 버퍼
        self.act_buffer=deque(maxlen=self.max_history)  # 6차원 액션
        
        # Observation 그룹 정의
        self.obs_groups={
            'leg_pos': {'names':['leg_pos[0]','leg_pos[1]','leg_pos[2]','leg_pos[3]'], 'indices':[0,1,2,3]},
            'wheel_vel': {'names':['wheel_vel[0]','wheel_vel[1]'], 'indices':[4,5]},
            'base_ang': {'names':['base_ang[0]','base_ang[1]','base_ang[2]'], 'indices':[6,7,8]},
            'vcmd': {'names':['vcmd[0]','vcmd[1]','vcmd[2]','vcmd[3]'], 'indices':[9,10,11,12]},
            'quat': {'names':['quat[0]','quat[1]','quat[2]','quat[3]'], 'indices':[13,14,15,16]},
            'prev_act': {'names':['prev_act[0]','prev_act[1]','prev_act[2]','prev_act[3]','prev_act[4]','prev_act[5]'], 'indices':[17,18,19,20,21,22]}
        }
        
        # 각 observation별 버퍼 초기화
        for group_name,group_info in self.obs_groups.items():
            for idx in group_info['indices']:
                self.obs_buffer[idx]=deque(maxlen=self.max_history)

        # ---- ROS2 ----
        rclpy.init()
        self.ros2_node=ROS2Node()
        threading.Thread(target=self.ros_spin,daemon=True).start()

        # ---- timers ----
        self.manual_timer=QTimer()
        self.manual_timer.timeout.connect(self.send_manual_joint_cmd)

        self.ui_timer=QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)  # 10Hz 업데이트

        self.init_ui()

    # =========================
    # ROS
    # =========================
    def ros_spin(self):
        while rclpy.ok():
            rclpy.spin_once(self.ros2_node,timeout_sec=0.1)

    # =========================
    # UI SETUP
    # =========================
    def init_ui(self):
        cw=QWidget()
        self.setCentralWidget(cw)
        layout=QHBoxLayout(cw)
        layout.addWidget(self.make_control_panel(),1)
        layout.addWidget(self.make_obs_panel(),2)

    def make_control_panel(self):
        """좌측 패널: 버튼 + Manual 입력 + Action 그래프"""
        w=QWidget()
        l=QVBoxLayout(w)

        # 버튼들
        btn_group=QGroupBox("Control")
        btn_layout=QVBoxLayout()
        self.manual_btn=QPushButton("Manual")
        self.run_btn=QPushButton("Run")
        self.auto_btn=QPushButton("Auto")

        self.manual_btn.clicked.connect(self.on_manual)
        self.run_btn.clicked.connect(self.on_run)
        self.auto_btn.clicked.connect(self.on_auto)

        for b in (self.manual_btn,self.run_btn,self.auto_btn):
            b.setMinimumHeight(45)
            btn_layout.addWidget(b)
        btn_group.setLayout(btn_layout)
        l.addWidget(btn_group)

        # Manual 입력
        manual_group=QGroupBox("Manual Input")
        manual_layout=QVBoxLayout()
        self.manual_inputs=[]
        grid=QGridLayout()
        names=["Wheel L","Wheel R","Upper L","Upper R","Lower L","Lower R"]
        for i,n in enumerate(names):
            sb=QDoubleSpinBox()
            sb.setRange(-10,10)
            sb.setDecimals(4)
            grid.addWidget(QLabel(n),i,0)
            grid.addWidget(sb,i,1)
            self.manual_inputs.append(sb)
        manual_layout.addLayout(grid)
        manual_group.setLayout(manual_layout)
        l.addWidget(manual_group)

        # Action 그래프
        act_group=QGroupBox("Actions (6 dims)")
        act_layout=QVBoxLayout()
        self.act_fig=Figure(figsize=(6,4.4))  # 원래 4의 1.1배
        self.act_canvas=FigureCanvas(self.act_fig)
        self.act_ax=self.act_fig.add_subplot(111)
        self.act_ax.set_xlabel('Time (steps)')
        self.act_ax.set_ylabel('Action Value')
        self.act_ax.set_title('Action History')
        self.act_ax.grid(True)
        self.act_lines=[]
        act_names=["wheel_L","wheel_R","upper_L","upper_R","lower_L","lower_R"]
        colors=['r','g','b','c','m','y']
        for i,(name,color) in enumerate(zip(act_names,colors)):
            line,=self.act_ax.plot([],[],label=name,color=color,linewidth=1.5)
            self.act_lines.append(line)
        self.act_ax.legend(loc='upper right',fontsize=8)
        self.act_fig.tight_layout()
        act_layout.addWidget(self.act_canvas)
        act_group.setLayout(act_layout)
        l.addWidget(act_group)

        # Statistics
        stats_group=QGroupBox("Statistics")
        stats_layout=QGridLayout()
        stats_layout.addWidget(QLabel("Step:"),0,0)
        self.step_label=QLabel("0")
        stats_layout.addWidget(self.step_label,0,1)
        stats_group.setLayout(stats_layout)
        l.addWidget(stats_group)

        l.addStretch()
        return w

    def make_obs_panel(self):
        """우측 패널: Observations 그래프들 (그룹별) - 스크롤 가능"""
        # 스크롤 영역 생성
        scroll=QScrollArea()
        scroll.setWidgetResizable(False)  # False로 설정하여 스크롤 활성화
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 컨텐츠 위젯
        content_widget=QWidget()
        l=QVBoxLayout(content_widget)
        l.setSpacing(10)  # 그룹 간 간격
        l.setContentsMargins(5,5,5,5)

        # 각 그룹별로 그래프 생성
        self.obs_figs={}
        self.obs_canvases={}
        self.obs_axes={}
        self.obs_lines={}

        for group_name,group_info in self.obs_groups.items():
            group_box=QGroupBox(f"{group_name.upper()} ({len(group_info['names'])} dims)")
            group_layout=QVBoxLayout()
            group_layout.setContentsMargins(5,5,5,5)
            
            # 그래프 생성 (높이를 더 키움: 5.0)
            fig=Figure(figsize=(8,5.0))
            canvas=FigureCanvas(fig)
            ax=fig.add_subplot(111)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Value')
            ax.set_title(f'{group_name}')
            ax.grid(True,alpha=0.3)
            
            # 각 차원별 라인 생성
            lines=[]
            colors=plt.cm.tab10(np.linspace(0,1,len(group_info['names'])))
            for i,(name,color) in enumerate(zip(group_info['names'],colors)):
                line,=ax.plot([],[],label=name,color=color,linewidth=1.5)
                lines.append(line)
            ax.legend(loc='upper right',fontsize=7,ncol=2)
            fig.tight_layout()
            
            group_layout.addWidget(canvas)
            group_box.setLayout(group_layout)
            l.addWidget(group_box)
            
            # 저장
            self.obs_figs[group_name]=fig
            self.obs_canvases[group_name]=canvas
            self.obs_axes[group_name]=ax
            self.obs_lines[group_name]=lines

        # 스트레치 제거 (스크롤 작동을 위해)
        # l.addStretch()
        
        # 컨텐츠 위젯 크기 설정 (스크롤 작동을 위해)
        content_widget.setMinimumWidth(800)
        # 그룹 개수 * (그래프 높이 + 간격) 계산하여 최소 높이 설정
        num_groups=len(self.obs_groups)
        estimated_height=num_groups*(500+50)  # 각 그래프 약 500px + 간격 50px
        content_widget.setMinimumHeight(estimated_height)
        
        scroll.setWidget(content_widget)
        
        return scroll

    # =========================
    # BUTTON CALLBACKS
    # =========================
    def on_manual(self):
        self.current_mode="MANUAL"
        self.ros2_node.publish_cmd("MANUAL::STOP")

    def on_auto(self):
        self.current_mode="AUTO"
        self.ros2_node.publish_cmd("AUTO::STOP")

    def on_run(self):
        if self.current_mode is None:
            return
        self.is_running=not self.is_running
        cmd=f"{self.current_mode}::{'RUN' if self.is_running else 'STOP'}"
        self.ros2_node.publish_cmd(cmd)
        if self.current_mode=="MANUAL":
            self.manual_timer.start(20) if self.is_running else self.manual_timer.stop()

    # =========================
    # DATA FLOW
    # =========================
    def send_manual_joint_cmd(self):
        if self.current_mode=="MANUAL" and self.is_running:
            vals=[s.value() for s in self.manual_inputs]
            self.ros2_node.publish_joint_cmd(vals)
            # 액션 버퍼에 추가
            self.act_buffer.append(vals.copy())
            self.time_buffer.append(self.step)
            self.step+=1

    def update_ui(self):
        """UI 업데이트 (그래프 갱신)"""
        self.step_label.setText(str(self.step))
        
        # Manual 모드일 때 현재 입력값을 액션으로 표시
        if self.current_mode=="MANUAL":
            manual_act=np.array([s.value() for s in self.manual_inputs])
            
            # Action 그래프 업데이트
            if len(self.time_buffer)>0:
                times=np.array(self.time_buffer)
                if len(self.act_buffer)>0:
                    acts=np.array(list(self.act_buffer))
                    for i,line in enumerate(self.act_lines):
                        if acts.shape[0]>0:
                            line.set_data(times,acts[:,i])
                    self.act_ax.relim()
                    self.act_ax.autoscale_view()
                    self.act_canvas.draw()
        
        # Observations 그래프 업데이트 (실제 데이터가 들어오면)
        # 현재는 더미 데이터로 표시 (나중에 ROS2 subscriber로 받아서 업데이트)
        # TODO: ROS2 subscriber에서 obs 데이터를 받아서 self.obs_buffer에 추가
        
        # 각 그룹별 그래프 업데이트
        for group_name,group_info in self.obs_groups.items():
            if len(self.time_buffer)>0:
                times=np.array(self.time_buffer)
                ax=self.obs_axes[group_name]
                lines=self.obs_lines[group_name]
                
                for i,idx in enumerate(group_info['indices']):
                    if len(self.obs_buffer[idx])>0:
                        values=np.array(list(self.obs_buffer[idx]))
                        if len(values)==len(times):
                            lines[i].set_data(times,values)
                
                ax.relim()
                ax.autoscale_view()
                self.obs_canvases[group_name].draw()

    def closeEvent(self,event):
        """종료 시 정리"""
        if self.current_mode=="MANUAL" and self.is_running:
            self.manual_timer.stop()
            self.ros2_node.publish_cmd("MANUAL::STOP")
        elif self.current_mode=="AUTO" and self.is_running:
            self.ros2_node.publish_cmd("AUTO::STOP")
        try:
            self.ros2_node.destroy_node()
            rclpy.shutdown()
        except:
            pass
        event.accept()


# =========================
# ENTRY
# =========================
def main():
    app=QApplication(sys.argv)
    w=MonitorGUI()
    w.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
