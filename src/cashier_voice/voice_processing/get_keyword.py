# ros2 service call /get_keyword std_srvs/srv/Trigger "{}"
import os
from unittest import result
import rclpy
import pyaudio
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

from std_srvs.srv import Trigger
from voice_processing.MicController import MicController, MicConfig

from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT

# 추가
import warnings
from rclpy.action import ActionServer
from cashier_interfaces.action import VoiceSession
from voice_processing.tts_helper import speak
import re

############ Package Path & Environment Setting ############
current_dir = os.getcwd()
package_path = get_package_share_directory("cashier_voice")

is_laod = load_dotenv(dotenv_path=os.path.join(f"{package_path}/resource/.env"))
openai_api_key = os.getenv("OPENAI_API_KEY")

CMD_UNKNOWN = 0
CMD_SCAN = 1
CMD_PACK = 2

############ AI Processor ############
# class AIProcessor:
#     def __init__(self):


############ GetKeyword Node ############
class GetKeyword(Node):
    def __init__(self):


        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.5, openai_api_key=openai_api_key
        )

        prompt_content = """
            당신은 캐셔 로봇의 음성 명령 분석기입니다.

            <목표>
            사용자의 문장에서 다음 3가지를 추출하세요.
            1. 상품명
            2. 수량
            3. 명령 종류

            <상품 리스트>
            상품은 특정한 상품명으로 제한되며 매칭된 부분으로 출력된다.
            발음이 조금 다르거나 약간의 오타가 있어도 문맥상 유추 가능한 경우에는
            상품 리스트 내 항목으로 최대한 추론해 반환하세요.

            - 박하사탕 : halls
            - 호올스 : halls
            - 곤충 : insect
            - 곤충블록 : insect
            - 벌레장난감 : insect
            - 케러멜 : caramel
            - 사탕 : candy
            - 연고 : cream
            - 크림 : cream
            - 빨간이클립스 : eclipse_red
            - 빨간박하사탕 : eclipse_red
            - 초록이클립스 : eclipse_green
            - 초록박하사탕 : eclipse_green

            <명령 종류>
            - scan : 테이블 위 상품 전체를 스캔/인식 시작하는 명령
            - cancel : 이미 스캔된 상품 중 특정 상품을 취소/제거하는 명령
            - pack : 포장을 요청하는 명령
            - yes : 긍정
            - no : 부정
            - unknown : 명령이 불명확하거나 인식된 명령이 없는 경우

            <규칙>
            - 출력 형식은 반드시 다음 형식을 따르세요:
                상품1 상품2 ... / 수량1 수량2 ... / 명령
            - 각 항목은 공백으로 구분합니다.
            - 상품이 없으면 첫 번째 칸은 비웁니다.
            - 수량이 없으면 두 번째 칸은 비웁니다.
            - 명령은 반드시 scan, cancel, pack, yes, no, unknown 중 하나만 출력합니다.
            - scan 명령은 상품명이 없어도 되며,
                "스캔해줘", "계산해줘", "스캔 시작해줘" 같은 표현은 / / scan 으로 출력합니다.
            - cancel 명령은 상품명을 추출해야 하며,	수량이 명시되지 않았으면 기본값은 1로 판단합니다.
            - "한 개", "하나", "1개"는 모두 1로 처리합니다.
            - "두 개", "둘", "2개"는 모두 2로 처리합니다.
            - "세 개", "셋", "3개"는 모두 3으로 처리합니다.
            - 응 같은 긍정 표현은 yes로, 아니요 같은 부정 표현은 no로 처리합니다.
            - 긍정 명령은 상품과 수량 없이 / / yes 형식으로 출력합니다.
            - 부정 명령은 상품과 수량 없이 / / no 형식으로 출력
            - 포장 명령은 상품과 수량 없이 / / pack 형식으로 출력합니다.
            - 문장에 약간의 말실수나 표현 차이가 있어도 의미를 기준으로 판단합니다.

            <예시>
            입력: "스캔해줘"
            출력: / / scan

            입력: "스캔 시작해줘"
            출력: / / scan

            입력: "계산해줘"
            출력: / / scan

            입력 : "호올스"
            출력 : halls / / unknown

            입력: "호올스 하나 취소해줘"
            출력: halls / 1 / cancel

            입력: "곤충 두 개 취소해줘"
            출력: insect / 2 / cancel

            입력: "호올스 두 개 사탕 한 개 취소해줘"
            출력: halls candy / 2 1 / cancel

            입력: "포장해줘"
            출력: / / pack

            입력: "네"
            출력: / / yes

            입력: "아니요"
            출력: / / no

            입력: "응"
            출력: / / yes

            입력: "아니"
            출력: / / no

            입력: "필요해"
            출력: / / yes

            입력: "필요 없어"
            출력: / / no

            <사용자 입력>
            "{user_input}"
        """

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = self.prompt_template | self.llm
        # self.lang_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.stt = STT(openai_api_key=openai_api_key)


        super().__init__("get_keyword_node")
        # 오디오 설정
        mic_config = MicConfig(
            chunk=12000,
            rate=48000,
            channels=1,
            record_seconds=5,
            fmt=pyaudio.paInt16,
            device_index=10,
            buffer_size=24000,
        )
        self.mic_controller = MicController(config=mic_config)
        # self.ai_processor = AIProcessor()

        self.get_logger().info("MicRecorderNode initialized.")
        self.get_logger().info("wait for client's request...")
        # self.get_keyword_srv = self.create_service(
        #     Trigger, "get_keyword", self.get_keyword
        # )

        self._action_server = ActionServer(
            self,
            VoiceSession,
            "/voice/run_session",
            execute_callback=self.execute_callback,
        )

        self.wakeup_word = WakeupWord(mic_config.buffer_size)

    def extract_keyword(self, output_message):
        response = self.lang_chain.invoke({"user_input": output_message})
        result = [part.strip() for part in response.content.strip().split("/")]

        if len(result) != 3:
            self.get_logger().warn("LLM 출력 형식이 올바르지 않습니다.")
            self.get_logger().warn(f"raw response: {response.content}")
            return None

        items_text, count_text, command_text = result

        items = items_text.split() if items_text else []
        counts = count_text.split() if count_text else []
        command = command_text.strip()

        print(f"llm's response(items): {items}")
        print(f"llm's response(counts): {counts}")
        print(f"llm's response(command): {command}")

        return items, counts, command
    
    ### (3/9) 등록되지 않은 상품 체크 ###
    def find_unknown_items_in_text(self, text: str):
        known_aliases = {
            "호올스", "홀스", "박하사탕",
            "곤충", "곤충 블록", "곤충블록", "벌레 장난감", "벌레",
            "케러멜", "카라멜", "캐러멜",
            "사탕", "캔디",
            "연고", "크림",
            "빨간 이클립스", "빨간 박하사탕", "빨간이클립스", "빨간박하사탕",
            "초록 이클립스", "초록 박하사탕", "초록이클립스", "초록박하사탕",
        }

        stop_words = {
            "취소", "삭제", "제거", "빼", "빼줘",
            "해줘", "해주세요",
            "하나", "한개", "한", "두개", "두", "세개", "세",
            "개",
            "그리고", "이거", "저거", "그거", "좀"
        }

        # "콜라 하나", "사탕 두개" 같은 패턴에서 앞 단어 추출
        pattern = re.compile(r"([가-힣A-Za-z]+)\s*(하나|한\s*개|한개|두\s*개|두개|세\s*개|세개|\d+\s*개?|\d+)")
        candidates = []

        for match in pattern.finditer(text):
            raw_name = match.group(1).strip()

            if raw_name in stop_words:
                continue

            if raw_name not in known_aliases:
                candidates.append(raw_name)

        # 중복 제거
        unknown_items = []
        for item in candidates:
            if item not in unknown_items:
                unknown_items.append(item)

        return unknown_items
    ### 등록되지 않은 상품 체크 ###

    def count_item_by_name(self, items, target_name):
        return sum(1 for item in items if item.name == target_name)

    def remove_items_by_name(self, items, target_name, remove_count):
        new_items = []
        removed = 0

        for item in items:
            if item.name == target_name and removed < remove_count:
                removed += 1
                continue
            new_items.append(item)

        return new_items

    def parse_cancel_counts(self, counts, item_len):
        parsed_counts = []

        for i in range(item_len):
            # 수량이 부족하면 기본값 1
            if i < len(counts):
                try:
                    count = int(counts[i])
                except (ValueError, TypeError):
                    return None
            else:
                count = 1

            if count < 1:
                return None

            parsed_counts.append(count)

        return parsed_counts

    def finish_with_pack_check(self, goal_handle, result, items_in):
        max_retry = 3

        for attempt in range(max_retry):
            print("포장을 시작할까요? 포장을 원하시면 '포장해줘'라고 말씀해주세요.")
            speak("포장을 시작할까요? 포장을 원하시면 '포장해줘'라고 말씀해주세요.")

            output_message = self.stt.speech2text()
            self.get_logger().warn(f"recognized text(pack check): {output_message}")

            parsed = self.extract_keyword(output_message)

            if parsed is None:
                print("명령을 이해하지 못했습니다. 다시 말씀해주세요.")
                speak("명령을 이해하지 못했습니다. 다시 말씀해주세요.")
                continue

            _, _, command = parsed

            if command == "pack":
                print("포장을 시작합니다.")
                speak("포장을 시작합니다. 잠시만 기다려주세요.")
                result.success = True
                result.command = CMD_PACK
                result.items_out = items_in
                goal_handle.succeed()
                return result

            if command == "no":
                print("포장을 진행하지 않고 종료합니다.")
                speak("포장을 진행하지 않고 종료합니다.")
                result.success = True
                result.command = CMD_UNKNOWN
                result.items_out = items_in
                goal_handle.succeed()
                return result

            print("포장 여부를 다시 말씀해주세요.")
            speak("포장 여부를 다시 말씀해주세요.")

        print("포장 명령을 확인하지 못해 수정을 종료합니다.")
        speak("포장 명령을 확인하지 못해 수정을 종료합니다.")
        result.success = True
        result.command = CMD_UNKNOWN
        result.items_out = items_in
        goal_handle.succeed()
        return result


    def execute_callback(self, goal_handle):
        mode = goal_handle.request.mode # WAKEUP or EDIT를 받아서 모드를 결정

        feedback_msg = VoiceSession.Feedback()
        result = VoiceSession.Result()

        # -------------------------------
        # 1. WAKEUP 모드
        # -------------------------------
        if mode == "WAKEUP":
            try:
                print("open stream")
                self.mic_controller.open_stream()
                self.wakeup_word.set_stream(self.mic_controller.stream)
            except OSError:
                self.get_logger().error("Error: Failed to open audio stream")
                self.get_logger().error("please check your device index")
                result.success = False
                result.command = 0
                result.items_out = []
                goal_handle.abort()
                return result

            runtime = 0
            # WAKEUP 단어가 감지될 때까지 루프를 돌면서 피드백을 보냄
            while not self.wakeup_word.is_wakeup():
                feedback_msg.runtime = runtime
                goal_handle.publish_feedback(feedback_msg)
                runtime += 1
            print("안녕하세요. 고객님 무엇을 도와드릴까요?")
            speak("안녕하세요. 고객님 무엇을 도와드릴까요?")
            while True:
                # STT로 고객이 원하는 음성을 인식
                output_message = self.stt.speech2text()
                self.get_logger().warn(f"recognized text: {output_message}")
                # 명령 상품 수량 추출을 위해 LLM에 전달하고 결과를 받음
                parsed = self.extract_keyword(output_message)

                if parsed is None:
                    self.get_logger().warn("Failed to parse command from LLM output")
                    result.success = False
                    result.command = 0
                    result.items_out = []
                    goal_handle.abort()
                    return result

                items, counts, command = parsed
                self.get_logger().warn(f"parsed command: {command}")


                if command == "scan":
                    print("상품을 스캔하겠습니다. 잠시만 기다려주세요.")
                    speak("상품을 스캔하겠습니다. 잠시만 기다려주세요.")
                    result.command = CMD_SCAN
                    break  # scan 명령이 들어오면 WAKEUP 루프 탈출
                elif command == "pack" or command == "cancel" or command == "yes" or command == "no":
                    print("상품을 먼저 스캔해주세요.")
                    speak("상품을 먼저 스캔해주세요.")
                    continue  # pack 명령이 들어오면 안내 메시지 출력 후 계속 WAKEUP 루프 유지
                else:
                    print("알 수 없는 명령입니다. 다시 시도해주세요.")
                    speak("알 수 없는 명령입니다. 다시 시도해주세요.")
                    continue  # 명령이 불명확하면 안내 메시지 출력 후 계속 WAKEUP 루프 유지

            # 일단 1차 WAKEUP 모드에서는 스캔 명령만 필요, 실제 상품 리스트는 빈 리스트로 반환
            result.success = True
            result.items_out = []

            goal_handle.succeed()
            return result

        # -------------------------------
        # 2. EDIT 모드
        # -------------------------------
        elif mode == "EDIT":
            print("상품 스캔이 완료되었습니다. 수정할 상품이 있으신가요?")
            speak("상품 스캔이 완료되었습니다. 수정할 상품이 있으신가요?")
            items_in = list(goal_handle.request.items_in)

            state = "ASK_EDIT"

            self.get_logger().info(f"EDIT mode request received: {len(items_in)} items")

            for i, item in enumerate(items_in):
                self.get_logger().info(
                    f"[{i}] item_id={item.item_id}, name={item.name}"
                )

            while True:
                output_message = self.stt.speech2text()
                self.get_logger().warn(f"recognized text: {output_message}")

                parsed = self.extract_keyword(output_message)
                if parsed is None:
                    self.get_logger().warn("명령 해석에 실패했습니다. 다시 말씀해주세요.")
                    speak("명령 해석에 실패했습니다. 다시 말씀해주세요.")
                    continue

                items, counts, command = parsed

                ### (3/9) 등록되지 않은 상품 체크 ###
                unknown_items = self.find_unknown_items_in_text(output_message)

                if command == "cancel" and unknown_items:
                    print(
                        f"등록되지 않은 상품이 포함되어 있습니다: {', '.join(unknown_items)}. "
                        "다시 말씀해주세요."
                    )
                    speak(
                        f"등록되지 않은 상품이 포함되어 있습니다: {', '.join(unknown_items)}. "
                        "다시 말씀해주세요."
                    )
                    continue
                ### 등록되지 않은 상품 체크 ###

                self.get_logger().warn(f"parsed items: {items}")
                self.get_logger().warn(f"parsed counts: {counts}")
                self.get_logger().warn(f"parsed command: {command}")

                # 1) 처음 수정 여부를 묻는 상태
                if state == "ASK_EDIT":
                    if command == "yes":
                        print("취소할 상품명과 수량을 말씀해주세요.")
                        speak("취소할 상품명과 수량을 말씀해주세요.")
                        state = "ASK_CANCEL_DETAIL"
                        continue
                    
                    elif command == "no":
                        print("수정을 종료합니다.")
                        speak("수정을 종료합니다.")
                        # pack 관련 적용
                        return self.finish_with_pack_check(goal_handle, result, items_in)

                    elif command == "cancel":
                        # 바로 아래 취소 처리로 이어짐
                        state = "ASK_CANCEL_DETAIL"   

                    else:
                        print("수정할 상품이 있으시면 네, 없으시면 아니요, 또는 바로 취소할 상품을 말씀해주세요.")
                        speak("수정할 상품이 있으시면 네, 없으시면 아니요, 또는 바로 취소할 상품을 말씀해주세요.")
                        continue

                # 2) 한 번 수정 성공 후, 추가 수정 여부를 묻는 상태
                if state == "ASK_MORE_EDIT":
                    if command == "yes":
                        print("취소할 상품명과 수량을 말씀해주세요.")
                        speak("취소할 상품명과 수량을 말씀해주세요.")
                        state = "ASK_CANCEL_DETAIL"
                        continue
                    
                    elif command == "no":
                        # pack 관련 적용
                        print("수정을 종료합니다.")
                        speak("수정을 종료합니다.")
                        return self.finish_with_pack_check(goal_handle, result, items_in)

                    elif command == "cancel":
                        # 바로 아래 취소 처리로 이어짐
                        state = "ASK_CANCEL_DETAIL"

                    else:
                        print("추가 수정이 있으시면 네, 없으시면 아니요, 또는 바로 취소할 상품을 말씀해주세요.")
                        speak("추가 수정이 있으시면 네, 없으시면 아니요, 또는 바로 취소할 상품을 말씀해주세요.")
                        continue

                # 3) 취소할 상품명/수량을 받는 상태
                if state == "ASK_CANCEL_DETAIL":
                    if command == "no":
                        # pack 관련 적용
                        print("수정을 종료합니다.")
                        speak("수정을 종료합니다.")
                        return self.finish_with_pack_check(goal_handle, result, items_in)

                    elif command == "yes":
                        print("취소할 상품명과 수량을 다시 말씀해주세요.")
                        speak("취소할 상품명과 수량을 다시 말씀해주세요.")
                        continue

                    if command != "cancel":
                        print("취소할 상품명과 수량을 다시 말씀해주세요.")
                        speak("취소할 상품명과 수량을 다시 말씀해주세요.")
                        continue

                    if not items:
                        print("취소할 상품명이 없습니다. 다시 말씀해주세요.")
                        speak("취소할 상품명이 없습니다. 다시 말씀해주세요.")
                        continue

                    cancel_counts = self.parse_cancel_counts(counts, len(items))

                    if cancel_counts is None:
                        print("취소 수량을 이해하지 못했습니다. 다시 말씀해주세요.")
                        speak("취소 수량을 이해하지 못했습니다. 다시 말씀해주세요.")
                        continue

                    # 1) 먼저 전부 취소 가능한지 검사
                    for target_name, cancel_count in zip(items, cancel_counts):
                        available_count = self.count_item_by_name(items_in, target_name)

                        if available_count == 0:
                            print(f"현재 스캔된 상품 목록에 {target_name} 이(가) 없습니다. 다시 말씀해주세요.")
                            speak(f"현재 스캔된 상품 목록에 {target_name} 이(가) 없습니다. 다시 말씀해주세요.")
                            break

                        if cancel_count > available_count:
                            print(
                                f"현재 스캔된 {target_name}은(는) {available_count}개입니다. "
                                "취소할 상품 개수를 다시 말해주세요."
                            )
                            speak(
                                f"현재 스캔된 {target_name}은(는) {available_count}개입니다. "
                                "취소할 상품 개수를 다시 말해주세요."
                            )
                            break
                    # 2) 전부 가능하면 실제 삭제
                    else:
                        updated_items = list(items_in)

                        for target_name, cancel_count in zip(items, cancel_counts):
                            updated_items = self.remove_items_by_name(updated_items, target_name, cancel_count)

                        removed_log = ", ".join(
                            [f"{name} {count}개" for name, count in zip(items, cancel_counts)]
                        )
                        print(f"{removed_log} 취소했습니다.")
                        speak(f"{removed_log} 취소했습니다.")

                        items_in = updated_items

                        self.get_logger().info(f"updated item count: {len(items_in)}")
                        for i, item in enumerate(items_in):
                            self.get_logger().info(
                                f"[UPDATED {i}] item_id={item.item_id}, name={item.name}"
                            )

                        if len(items_in) == 0:
                            print("모든 상품이 취소되었습니다. 수정을 종료합니다.")
                            speak("모든 상품이 취소되었습니다. 수정을 종료합니다.")
                            # pack 관련 적용
                            result.success = True
                            result.command = CMD_UNKNOWN
                            result.items_out = items_in
                            goal_handle.succeed()
                            return result

                        print("추가로 수정할 상품이 있으신가요?")
                        speak("추가로 수정할 상품이 있으신가요?")
                        state = "ASK_MORE_EDIT"
                        continue

                    # break로 빠진 경우 다시 입력 받기
                    continue

        # -------------------------------
        # 3. 지원하지 않는 모드
        # -------------------------------
        else:
            self.get_logger().warn(f"Unsupported mode: {mode}")
            result.success = False
            result.command = CMD_UNKNOWN
            result.items_out = []
            goal_handle.abort()
            return result

def main():
    rclpy.init()
    node = GetKeyword()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
