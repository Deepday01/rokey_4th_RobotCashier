[강화학습 기반 3D Bin Packing 자동 포장 시스템]

조 이름: [F-1 - ROKEY]
팀원: [문홍일_박지훈_이승민_이창석_최대혁]

1. 📦 시스템 설계 및 플로우 차트

프로젝트의 전체적인 구조와 소프트웨어 흐름도입니다.

1-1. 시스템 설계도 (System Architecture)
<img width="689" height="490" alt="image" src="https://github.com/user-attachments/assets/packing_system_architecture.png" />

ROS2 기반 자동 포장 시스템의 노드 구조와 데이터 흐름을 나타냅니다.

여러 ROS2 노드가 협력하여 물체 인식 → packing 계획 생성 → 포장 실행의 전체 흐름을 수행합니다.

1-2. 플로우 차트 (Flow Chart)

프로젝트의 전체 동작 흐름과 주요 기능별 처리 과정을 나타냅니다.

🔹 Main Flow
<img width="321" height="591" alt="image" src="https://github.com/user-attachments/assets/packing_main_flow.png" />

물체 입력부터 packing plan 생성 및 실행까지의 전체 시스템 흐름

🔹 통합 Flow
<img width="884" height="661" alt="image" src="https://github.com/user-attachments/assets/packing_flow1.png" /> <img width="871" height="658" alt="image" src="https://github.com/user-attachments/assets/packing_flow2.png" />

각 ROS2 노드가 연동되어 최종 포장 동작으로 이어지는 전체 프로세스

🔹 Packing Plan 생성 로직
<p align="center"> <img src="https://github.com/user-attachments/assets/packing_plan_flow1.png" width="400"> </p> <p align="center"> <img src="https://github.com/user-attachments/assets/packing_plan_flow2.png" width="400"> </p>

입력된 물체 정보를 기반으로 배치 순서와 위치를 계산하는 과정

2. 🖥️ 운영체제 환경 (OS Environment)

이 프로젝트는 다음 환경에서 개발하였습니다.

OS: Ubuntu 22.04 LTS

ROS Version: ROS2 Humble

Language: Python

IDE: VS Code

Deep Learning Framework: PyTorch

Visualization: Matplotlib

Communication: ROS2 Service 기반 노드 통신

3. 🛠️ 사용 장비 목록 (Hardware List)

프로젝트에 사용된 주요 하드웨어 장비입니다.

장비명 (Model)	수량	비고
Control PC	1	RL 모델 연산 및 ROS2 노드 실행
Depth Camera	1	물체 인식 및 크기 정보 획득
Robot Manipulator	1	실제 포장 동작 수행
Robot Gripper	1	물체 파지
Basket / Container	1	물체 적재 공간
4. 📦 의존성 (Dependencies)

프로젝트 실행에 필요한 라이브러리입니다.

Python >= 3.10

ROS2 Humble

rclpy

Deep Learning

torch

numpy

Visualization

matplotlib

mpl_toolkits.mplot3d

5. ▶️ 실행 순서 (Usage Guide)

프로젝트를 실행하기 위한 순서입니다. 터미널 명령어를 순서대로 입력해 주세요.

Step 1. ROS DOMAIN 설정

ROS2 통신을 위한 네트워크 ID를 설정합니다.
※ 모든 터미널에서 동일하게 설정해야 합니다.

echo $ROS_DOMAIN_ID
export ROS_DOMAIN_ID=16
Step 2. ROS2 Workspace 빌드
cd ~/cashier_ws
colcon build
source install/setup.bash
Step 3. Workflow Node 실행

packing 시스템의 전체 동작을 관리하는 workflow 노드를 실행합니다.

ros2 run cashier_workflow workflow_node
Step 4. Launch 파일 실행

테스트 및 통합 실행을 위해 launch 파일을 사용할 수 있습니다.

ros2 launch cashier_workflow demo_split.launch.py

디버그 모드 실행

ros2 launch cashier_workflow demo_split_dev.launch.py debug_mode:=true

✔ 실행 순서 요약

ROS DOMAIN 설정
ROS2 Workspace 빌드
Workflow Node 실행 또는 launch 파일 실행
