# cobot_ws

ROS2 workspace 로키_4번째_로봇캐셔솔루션

## Structure
- src/: source packages
- bringup: launch and integration
- config: parameter files


# Git 브랜치 규칙

## 네이밍

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

## 1. 전체 폴더 구조

```jsx
cashier_ws/
├── .gitignore
├── README.md
├── docs/
│   ├── architecture.md
│   ├── interfaces.md
│   └── workflow.md
├── src/
│   ├── cashier_interfaces/     # 공통 인터페이스 전용
│   ├── cashier_workflow/       # workflow_node
│   ├── cashier_voice/          # voice_node 
│   ├── cashier_vision/         
│   ├── cashier_plan_packing/
│   ├── cashier_execute_packing/
│   ├── doosan-robot2/          # 로봇 브링업 파일
│   └── cashier_bringup/        # launch파일 config
├── build/
├── install/
└── log/
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
