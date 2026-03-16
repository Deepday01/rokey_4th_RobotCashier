# cobot_ws

ROS2 workspace 로키_4번째_로봇캐셔솔루션

## Structure
- src/: source packages
- bringup: launch and integration
- config: parameter files

## 1-1. 노드 설계도 (Node Architecture)
<p align="center">
<img width="1429" height="786" alt="Image" src="https://github.com/user-attachments/assets/ef689952-f678-47bb-b265-1b147d71e186" />
</p>

main : 배초/최종 제출용

dev : 팀 통합 브랜치

feature/기능명 : 개인 작업 브런치

- feature/voice
- feature/vision
- feature/planning
- feature/execute

fix/노드명/버그명

- fix/vision/debug

## 관리

1. 개인 작업 브랜치의에서 작업후 문제 없을 시 **dev로 merge**
2. dev로 테스팅 완료시 **main 으로 merge**


# 작업 내용

### 🔹 Main Flow
<p align="center">
<img width="413" height="374" alt="Image" src="https://github.com/user-attachments/assets/54775901-015c-46dd-bb69-c824a020adae" />
</p>
*사용자 인터랙션부터 로봇 적재까지의 전체 시스템 흐름*

---

### 🔹 통합 Flow

<img width="884" height="661" alt="image" src="https://github.com/user-attachments/assets/packing_flow1.png" />
<img width="871" height="658" alt="image" src="https://github.com/user-attachments/assets/packing_flow2.png" />

*각 ROS2 노드가 연동되어 최종 포장 동작으로 이어지는 전체 프로세스*

---

### 🔹 Workflow Node Flow

<p align="center">
<img width="400" src="WORKFLOW_FLOW_IMAGE_URL">
</p>

workflow node는 전체 시스템의 **중앙 제어 역할**을 수행하며 각 노드를 순차적으로 실행합니다.

주요 동작 흐름

1. 시스템 시작
2. voice_node 실행 (wake-up 대기)
3. vision_node 실행 (물체 인식)
4. voice_node 재실행 (물품 제외 요청 처리)
5. plan_packing_node 실행 (적재 계획 계산)
6. execute_packing_node 실행 (로봇 적재 수행)

---

### 🔹 Voice Node Flow
<p align="center"> <img width="400" src="VOICE_FLOW_IMAGE_URL"> </p>

voice node는 사용자 음성 명령을 인식하고 시스템 제어 명령으로 변환하는 노드입니다.

주요 기능

- Wake-up word 감지
- 음성 입력 STT 변환
- LLM 기반 명령 해석
- TTS 음성 안내
- 스캔 자세 이동 및 상품 취소 동작 실행

---


### 🔹 Vision Node Flow

<p align="center"> <img width="400" src="VISION_FLOW_IMAGE_URL"> </p>

vision node는 계산대 위 물체를 인식하고 위치 정보를 생성하는 노드입니다.

주요 기능

- YOLO 기반 객체 검출
- Depth 기반 거리 계산
- AprilTag 기반 좌표 보정
- 물체 방향(Yaw) 계산
- QR 코드 인식
- 물체 정보 ROS2 메시지 생성

---

### 🔹 Packing Plan Node Flow

<p align="center">
<img width="801" height="519" alt="Image" src="https://github.com/user-attachments/assets/79cfdd7e-3806-46ff-927e-8c1d3d7db43a" />
</p>

plan_packing node는 **물체 적재 계획을 계산**합니다.

주요 기능

- 물체 크기 정보 입력
- 배치 후보 생성
- 최적 적재 위치 계산
- packing plan 반환

---

### 🔹 Execute Packing Node Flow

<p align="center"> <img width="400" src="EXECUTE_FLOW_IMAGE_URL"> </p>

execute_packing node는 packing plan을 기반으로 로봇을 제어하여 실제 적재 동작을 수행하는 노드입니다.

주요 기능

- 물체 pick 동작 수행
- 회전 스테이션에서 자세 정렬
- 목표 위치로 물체 이동 및 배치
- ROS2 Action 기반 패킹 실행

---

### 🔹 RL 학습 로직
<p align="center">
<img width="401" height="640" alt="Image" src="https://github.com/user-attachments/assets/e7b2b620-5b86-4370-984c-71b6c04f9dd1" />
</p>
<p align="center">
<img width="401" height="479" alt="Image" src="https://github.com/user-attachments/assets/1629e427-98ad-494f-86ad-c88203b3e2e2" />
</p>
*환경 상태 입력 → 물체 선택 및 배치 위치 결정 → 보상을 통한 packing 정책 학습 과정*

---

# 2. 🖥️ 운영체제 환경 (OS Environment)

이 프로젝트는 다음 환경에서 개발하였습니다.

- **OS:** Ubuntu 22.04 LTS  
- **ROS Version:** ROS2 Humble  
- **Language:** Python  
- **IDE:** VS Code  
- **Deep Learning Framework:** PyTorch  
- **Visualization:** Matplotlib  
- **Communication:** ROS2 Service 기반 노드 통신  

---

# 3. 🛠️ 사용 장비 목록 (Hardware List)

프로젝트에 사용된 주요 하드웨어 장비입니다.

| 장비명 (Model) | 수량 | 비고 |
|---|---|---|
| Control PC | 1 | ROS2 노드 실행 |
| Intel RealSense | 1 | 물체 인식 및 위치 정보 획득 |
| Robot Manipulator M0609 | 1 | 실제 포장 동작 수행 |
| Robot Gripper | 1 | 물체 파지 |
| Basket / Container | 1 | 물체 적재 공간 |

---

# 4. 📦 의존성 (Dependencies)

프로젝트 실행에 필요한 라이브러리입니다.

- **Python >= 3.10**  
- **ROS2 Humble**  
- **rclpy**

### AI / LLM
- openai  
- langchain  
- langchain-openai  

### Voice Processing
- openwakeword  
- edge-tts  
- sounddevice  
- pyaudio  

### Computer Vision
- ultralytics  
- opencv-python  
- pupil-apriltags  
- pyzbar  

### Deep Learning / Numerical
- torch  
- numpy  
- scipy  

### Environment / Utilities
- python-dotenv  
- pymodbus==2.5.3  

### ROS Vision Interface
- ros-humble-cv-bridge  
- ros-humble-image-transport  

---

# 5. ▶️ 실행 순서 (Usage Guide)

프로젝트를 실행하기 위한 순서입니다. 터미널 명령어를 순서대로 입력해 주세요.

---

### Step 1. ROS DOMAIN 설정

ROS2 통신을 위한 네트워크 ID를 설정합니다.  
※ 모든 터미널에서 동일하게 설정해야 합니다.

```ruby
echo $ROS_DOMAIN_ID
export ROS_DOMAIN_ID=16
```
### Step 2. ROS2 Workspace 빌드

```ruby
cd ~/cashier_ws
colcon build
source install/setup.bash
```
### Step 3. Workflow Node 실행

packing 시스템의 전체 동작을 관리하는 workflow 노드를 실행합니다.

```ruby
ros2 run cashier_workflow workflow_node
```
### Step 4. Voice Node 실행

Voice Node를 실행합니다.
```ruby
ros2 run cashier_voice get_keyword
```
### Step 5. Vision Node 실행

Vision Node를 실행합니다.
```ruby
ros2 run cashier_vision vision_scan_items_action_server
```
### Step 6. PlanPacking Node 실행

PlanPacking Node를 실행합니다.
```ruby
ros2 run cashier_plan_packing packing_node
```
### Step 6. Execute Packing Node 실행

Execute Packing Node를 실행합니다.
```ruby
ros2 run cashier_execute_packing execute_packing_server
```

## 2. 인터페이스 폴더 구조

```jsx
src/cashier_interfaces/
├── msg/
│   ├── Item.msg
│   └── Placement.msg
├── srv/
│   └── ComputePackingPlan.srv
├── action/
│   ├── VoiceSession.action
│   ├── ScanItems.action
│   └── ExecutePacking.action
├── CMakeLists.txt
└── package.xml
```

## 3. src아래 패키지 안 내용물들

```jsx
workflow_node.py
각자의 코드들
```

## 4. casher_bringup

```
src/cashier_bringup/
├── launch/
│   ├── system.launch.py          # 최종 데모용 (전체 플로우)
│   ├── workflow_only.launch.py   # 개별 기능 테스트용
│   ├── voice_only.launch.py      
│   ├── vision_only.launch.py
│   └── debug.launch.py           # 개발 중 디버깅용(자유롭게 노드 체크)     
├── config/
│   ├── example.yaml              # 설정파일 필요하면 쓰고 아님 말고
├── package.xml
└── setup.py or CMakeLists.txt
```

# 참고사항

<aside>
💡

## Q. config 안의 .yaml 파일은 뭔가?

**정의**
각 노드의 설정값(parameter)을 따로 빼놓는 파일

**의의**
코드 안에 하드코딩하지 말고 바깥에서 바꿀 수 있게 만든 설정 파일

**예시**
launch 파일에서 노드를 실행할 때)

```python
Node(
package='cashier_vision',
executable='vision_node',
name='vision_node',
parameters=['config/vision.yaml']  
)

# vision.yaml의 parameter를 읽어온다.
```

**주석**
필요하다면 적극활용, 아직 어렵다면 PASS

</aside>



# 실행법
## 1. workflow node
### 단독 실행법
```
cd ~/cashier_ws/  
source install/setup.bash   
ros2 run cashier_workflow workflow_node  
```

### launch 파일로 실행
테스트용 더미 런치  
더미 노드에서 필요한 부분 주석처리해서 사용.  
```
ros2 launch cashier_workflow demo_split.launch.py 
ros2 launch cashier_workflow demo_split_dev.launch.py debug_mode:=true 
```