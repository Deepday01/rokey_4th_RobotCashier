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