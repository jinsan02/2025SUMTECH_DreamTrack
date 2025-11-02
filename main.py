import os
import json
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------------------------------
# 0. 환경 설정 및 클라이언트 초기화
# -----------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
client = OpenAI(api_key=api_key)
# 모델 이름을 gpt-5-nano-2025-08-07로 고정합니다.
MODEL_NAME = "gpt-5-nano-2025-08-07"
print("OpenAI 클라이언트가 성공적으로 초기화되었습니다.")


# -----------------------------------------------------------------
# 1) AI #1: 종합 진단 리포트 생성 (Markdown)
# -----------------------------------------------------------------
def generate_initial_report(survey_response_json: str) -> Optional[str]:
    """
    [AI #1]
    사용자가 응답한 'questionset.json' 기반의 JSON 입력을 분석하여,
    후속 AI가 심화 질문을 생성할 수 있도록
    8단계의 상세하고 구조화된 Markdown 리포트를 생성합니다.
    """
    print("Step 1: [AI #1] 8단계 종합 진단 리포트 생성 중...")

    # AI #1: 상세한 8단계 리포트 생성 프롬프트
    system_prompt = """
    고등학생 대상 설문 결과를 바탕으로, 후속 AI 모델이 해당 학생에게 심화질문(Deep-dive Q&A)을
    효과적으로 생성할 수 있도록 충분히 구체적이고 구조화된 진로 진단 분석 리포트를 작성하세요. 
    분석/결과는 반드시 풍부한 근거와 논리적 전개를 바탕으로 단계별 reasoning을
    명확히 밝힌 뒤 결론 및 추천(Conclusion)을 제시해야 하며,각 항목별 내용은 심화 인터뷰나 추가 질문이 가능한 수준의 상세 정보, 예시, 근거, 데이터 요약을 포함해야 합니다.
    
    - 분석 대상: 고등학생(사용자) 설문 결과(JSON)
    - 분석 목적: 학생의 흥미, 강점, 성향, 현재 상태와 목표 간 Gap, 성장/보완점, 객관적 준비 방안 등을 충분한 근거와 논거로 도출하여, 후속 AI가 추가 질문·심화 코칭/진단을 이어갈 수 있는 context-rich output 제공
    - 분석 기준: Holland RIASEC 프레임워크, 설문 해석 가이드, 진로지도 전문 분석 기준
    
    # 진단·분석 단계 및 단계별 지시사항
    
    ## 1. 입력 정보 요약
    - 응답된 설문 각 항목(수치, 선택, 서술형)을 원문 또는 의미가 정확히 드러나도록 정리해 표기
    - [심화질문 가능 포인트: ...]로 마킹 가능
    
    ## 2. RIASEC 진단 및 흥미/강점 도출
    - 리커트 점수(Q1-7), 주된 활동/흥미/과목(Q8-9)을 RIASEC 6유형(R, I, A, S, E, C)에 정확히 매핑
    - reasoning: 점수 및 경험 데이터를 해석하여 추론 근거, 유형 근거, 왜 이런 유형이 도출됐는지 서술
    - 결론: 상위 2~3개 유형을 점수/근거와 함께 분석적 텍스트로 도출. 관련 직업/전공/활동 예시 및 학생 특성과의 연결 근거 제시.
    
    ## 3. 현재 상태/역량/경험
    - Q12, Q13, Q17, Q20, Q21 등에서 구체적 활동사례, 보유한 역량·지식·특징을 분석
    - reasoning: 각 활동/경험이 어떠한 의미를 지니는지, 해당 진로/성향과의 연계성까지 논리적으로 도출
    
    ## 4. 희망 진로/분야/목표 및 동기
    - 학생의 명확한 희망 진로(Q10, Q18)가 있다면 선택 이유(Q11, Q19)와 심층 동기를 구체적으로 분석
    - 목표가 불분명할 경우, 현재 정보상 가능성이 높거나, 성향/강점 기반 추천 직무/전공을 도출하되 그 사유(근거) 명시
    
    ## 5. Gap(차이/과제) 분석
    - 희망 분야(4단계) 및 현재 준비상태(3단계), 역량, 경험, 흥미 간 차이와 그 이유, 극복/개선 필요점 등 분석
    - reasoning: Gap 규명에 데이터 논거/구체 경험, 스스로 파악하기 어려운 미세 과제도 세부적 추론
    
    ## 6. 보완점 및 성장방향
    - Q15, Q22 등에서 본인의 강점과 부족/보완이 요구되는 역량·경험(역량/지식/태도/정서 등) 도출, 구체적 근거 포함
    - reasoning: 왜 이런 보완점이 중요한가, 어떤 상황에서 드러났는가 등 심층 분석
    
    ## 7. 맞춤형 준비방안 및 단계별 로드맵 (초안)
    - Gap 해소를 위한 구체적 실천계획(1~4단계 등 Action Plan), 실제 실행 가능한 방안·경험·학습·멘토링 추천 (후속 AI가 이 초안을 정교화할 것임)
    - reasoning: 단계별 방안이 해당 학생에게 왜 적합한지, 예상 효과/위험요소 논리적으로 제시
    
    ## 8. 종합 코멘트 및 동기부여
    - 전반 분석 요약, 성장 잠재력, 심리적 동기 및 후속 성장 촉진 조언
    - reasoning: 앞선 분석 바탕으로, 동기·자신감 제고를 위한 구체적 근거 포함
    
    # Reminder
    - 반드시 분석(Reasoning) → 결론/추천(Conclusion) 순서로 각 단계 작성
    - 각 항목 reasoning 내에서 후속 AI가 쉽게 참조할 수 있는 분석 포인트, 세부 근거/맥락/예시 모두 충실히 포함
    """

    user_prompt = f"다음은 학생의 복합 진단 설문 응답(JSON)입니다:\n\n{survey_response_json}"

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        result_markdown = resp.choices[0].message.content
        print("Step 1: [AI #1] 8단계 종합 리포트 생성 완료.")
        return result_markdown
    except Exception as e:
        print(f"Step 1 오류: {e}")
        return None


# -----------------------------------------------------------------
# 2) AI #2: 갭(Gap) 요약 및 심화질문 생성 (***'개인화 초점'으로 대폭 수정됨***)
# -----------------------------------------------------------------
def generate_gap_and_questions(full_report_markdown: str) -> Optional[Dict[str, Any]]:
    """
    [AI #2]
    AI #1의 상세 리포트(Markdown)를 읽고,
    1. 'Gap 분석(5단계)' 요약 제시
    2. '개인화'된 심화 질문 5개 생성
    """
    print(f"Step 2: [AI #2] 갭(Gap) 요약 및 심화질문 5개 생성 중...")

    system_prompt = """
    당신은 'DreamTrack'의 2단계 '학생 중심의 심층 면접관(Student-Centric Interviewer)' AI입니다.
    당신의 임무는 [AI #1의 8단계 종합 리포트]를 정밀하게 읽고, 
    다음 2가지 핵심 결과물을 포함하는 JSON 객체를 생성하는 것입니다.

    # 1. 분석 및 지시사항

    ## 1.1. Gap 요약 (To User)
    - 리포트의 `## 5. Gap(차이/과제) 분석` 섹션의 'Conclusion(결론)' 부분을
      찾아, 학생이 이해하기 쉬운 2~3문장의 '갭 요약'으로 재구성하세요.

    ## 1.2. 심화 질문 5개 생성 (To AI #3)
    - **(핵심 임무)** 리포트 전체, 특히 `## 7. 맞춤형 준비방안 ... (초안)` 섹션과
      `[심화질문 가능 포인트]` 마크를 검토하세요.
    - AI #3이 '최종 로드맵'을 확정하는 데 **반드시 필요한** 5개의 질문을 생성하세요.
    
    - **[핵심 질문 방향성: '기술'이 아닌 '사람(학생)'에 집중]**
        - AI #1 리포트는 '기술적 갭'(예: '알고리즘 부족')을 제안했습니다. 
        - 당신의 임무는 이 '기술적 제안'에 대한 학생의 **'개인적 반응'**을 묻는 것입니다.
        - 질문은 학생의 **선호도(Preference)**, **현실적 제약(Constraint)**, 
          **학습 동기(Motivation)**, **개인적 장벽(Barrier)**을 파악하는 데 초점을 맞춰야 합니다.
        - **학생에게 과제를 제시하지 말고, 학생의 생각과 현실적인 계획을 물어보세요.**

    # 2. 출력 형식 (JSON)
    - 다른 설명 없이, 반드시 아래 JSON 형식으로만 응답하세요.
    - **(중요) `question` 필드에는 학생에게 할 실제 질문만 생성하세요. 절대 설명용 괄호를 넣지 마세요.**

    [JSON 형식]
    {
      "gapSummary": "AI #1의 분석 결과를 2-3문장으로 요약합니다.",
      "deepDiveQuestions": [
        {"id": 1, "question": "실제 생성된 개인화 질문 1"},
        {"id": 2, "question": "실제 생성된 개인화 질문 2"},
        {"id": 3, "question": "실제 생성된 개인화 질문 3"},
        {"id": 4, "question": "실제 생성된 개인화 질문 4"},
        {"id": 5, "question": "실제 생성된 개인화 질문 5"}
      ]
    }
    """

    user_prompt = f"""
    [AI #1 종합 리포트 (Markdown)]
    
    {full_report_markdown}

    ---
    위 리포트를 바탕으로 'Gap 요약'과 **'기술적이지 않고 개인화된'** 심화 질문 5개를 
    요청한 JSON 형식으로 생성해 주세요.
    """

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        result = json.loads(resp.choices[0].message.content)
        print("Step 2: [AI #2] 갭 요약 및 심화질문 생성 완료.")
        return result
    except Exception as e:
        print(f"Step 2 오류: {e}")
        return None


 user_prompt = f"""
    [목표 분야]
    {target_field}

    [최종 로드맵 (AI #3의 산출물)]
    {json.dumps(final_roadmap, ensure_ascii=False, indent=2)}

    [웹 검색 결과 (참조용)]
    {json.dumps(google_search_results, ensure_ascii=False, indent=2)}
    
    위 정보를 바탕으로, [목표 분야]와 [최종 로드맵]을 연결지어
    친화적인 최신 트렌드 퀴즈 **1개**를 요청한 JSON 형식으로 생성하세요.
    """

# -----------------------------------------------------------------
# 4) AI #4: 직무 관련 퀴즈 생성 (동기부여 초점)
# -----------------------------------------------------------------
def generate_quiz(
    target_field: str,
    final_roadmap: Dict[str, Any] 
) -> Optional[Dict[str, Any]]:
    """
    [AI #4]
    목표 분야와 최종 로드맵을 바탕으로,
    동기부여 및 최신 트렌드를 묻는 친화적인 퀴즈 1개를 생성합니다.
    """
    print(f"Step 4: [AI #4] '{target_field}' 관련 최신 트렌드 퀴즈 생성 중...")

    system_prompt = """
    당신은 'DreamTrack'의 4단계 '동기부여 튜터(Engagement Tutor)' AI입니다.
    학생의 [목표 분야]와 방금 생성된 [최종 로드맵]을 바탕으로, 
    학생의 흥미를 유발하고 지속적인 동기를 부여할 수 있는
    **재미있고 친화적인 톤**의 **단 하나의 퀴즈**를 생성하세요.

    # 1. 분석 및 지시사항

    ## 1.1. 퀴즈 유형
    - **(웹 검색 가정)** [목표 분야]와 관련된 **'최신 시사 상식'** 또는 **'기술 트렌드'**에
      대한 질문을 **1개** 만드세요.
    - AI의 훈련 데이터에 기반하더라도, 마치 오늘 뉴스를 본 것처럼
      "최근 ... 분야에서 주목받는 OOO 기술" 같은 형식을 사용하세요.

    ## 1.2. 톤 앤 매너
    - 퀴즈의 `question`과 `explanation`은 딱딱한 정보 전달이 아닌,
      학생의 **관심사를 격려**하고 **[최종 로드맵] 실행을 응원**하는
      **매우 친화적이고 긍정적인** 어조여야 합니다.

    # 2. 출력 형식 (JSON)
    - 다른 설명 없이, 반드시 아래 JSON 형식(단일 객체)으로만 응답하세요.

    [JSON 형식 (배열이 아닌 단일 객체)]
    {
      "quizTitle": "(예: 🚀 데이터 사이언티스트를 향한 스텝 업 퀴즈!)",
      "quiz": {
        "type": "trend",
        "question": "(예: [웹 검색 가정] 최근 AI 분야에서, '데이터 사이언티스트'의 작업을 획기적으로 도와주는 '자율 AI 에이전트' 기술이 큰 화제예요. 이 기술의 핵심 목표는 무엇일까요?)",
        "options": [
            "(예: 사람의 개입 없이 스스로 데이터를 분석하고 리포트를 작성하는 것)",
            "(예: 엑셀 속도를 100배 빠르게 하는 것)", 
            "(예: 웹사이트를 더 예쁘게 디자인하는 것)"
        ],
        "answerIndex": 0,
        "explanation": "딩동댕! 🤖 맞아요. AI가 스스로 분석가가 되어 일하는 거죠. (목표)님이 방금 확정된 '미니 프로젝트'부터 차근차근 해나가다 보면, 언젠가 이런 AI 에이전트와 함께 일하게 될 거예요. 정말 멋지지 않나요? 당신의 첫걸음을 응원합니다! 🔥"
      }
    }
    """

    user_prompt = f"""
    [목표 분야]
    {target_field}

    [최종 로드맵 (AI #3의 산출물)]
    {json.dumps(final_roadmap, ensure_ascii=False, indent=2)}
    
    위 정보를 바탕으로, [목표 분야]와 [최종 로드맵]을 연결지어
    친화적인 최신 트렌드 퀴즈 **1개**를 요청한 JSON 형식으로 생성하세요.
    """
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, 
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        final_output = json.loads(resp.choices[0].message.content)
        print("Step 4: [AI #4] 퀴즈 생성 완료.")
        return final_output
    except Exception as e:
        print(f"Step 4 오류: {e}")
        return None


# -----------------------------------------------------------------
# 5) 메인 실행 블록 (데모)
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("========================================")
    print("DreamTrack (New) 4-AI 파이프라인 시뮬레이션 시작")
    print("========================================\n")

    # (데모) 'questionset.json'에 대한 시뮬레이션 응답
    sample_survey_response = {
      "no_1": 4, "no_2": 5, "no_3": 5, "no_4": 2, "no_5": 3, "no_6": 2, "no_7": 3,
      "no_8": "복잡한 문제 탐구",
      "no_9": "수학",
      "no_10": ["IT·소프트웨어 개발"],
      "no_11": ["개인적 흥미/관심", "강점 혹은 적성"],
      "no_12": ["수리·과학적 사고", "기술적 스킬(코딩, 기계조작 등)"],
      "no_13": "학교/학과 동아리 활동",
      "no_14": ["관련 온라인 강의 수강", "교내·외 프로젝트 혹은 경진대회 참여"],
      "no_15": ["전문 지식/기술 부족"],
      "no_16": ["인터넷 검색(블로그, 기사 등)", "유튜브/온라인 강의"],
      "no_17": ["성실함", "호기심/탐구심"],
      "no_18": "데이터 사이언티스트 또는 AI 엔지니어",
      "no_19": "파이썬으로 데이터 분석을 해보니 재미있었고, 수학/과학 강점을 살릴 수 있다고 생각해서입니다.",
      "no_20": "코딩 동아리에서 공공데이터를 파이썬으로 분석하는 프로젝트를 주도했습니다.",
      "no_21": "교내 코딩 대회에 참가해 3등을 했습니다. 알고리즘 공부가 부족했지만, 포기하지 않고 끝까지 풀었습니다.",
      "no_22": "AI를 하려면 영어를 잘해야 하는데, 전문 용어나 논문 읽기가 두렵습니다. 어휘력이 부족합니다.",
      "no_23": "온라인 강의로 머신러닝을 공부하고, 캐글 같은 경진대회에도 참여해보고 싶습니다.",
      "no_24": "3Blue1Brown, 코딩하는 김XX"
    }

    # AI #1에 전달할 입력을 JSON 문자열로 변환
    survey_input_json = json.dumps(sample_survey_response, ensure_ascii=False)

    # 1) [AI #1] 8단계 종합 리포트 생성 (Markdown)
    full_report = generate_initial_report(survey_input_json)
    if not full_report:
        print("프로필 생성 실패로 종료합니다.")
        exit(1)

    print("\n[AI #1 종합 리포트 결과 (Markdown)]")
    print(full_report)
    print("----------------------------------------")

    # 2) [AI #2] 갭 요약 및 심화질문 생성 (JSON)
    gap_and_questions = generate_gap_and_questions(full_report)
    if not gap_and_questions:
        print("갭 분석 및 심화질문 생성 실패로 종료합니다.")
        exit(1)

    print("\n[AI #2 갭 요약 및 심화질문 결과 (JSON)]")
    print(json.dumps(gap_and_questions, ensure_ascii=False, indent=2))
    print("----------------------------------------")

    # 3) (사용자) 심화 답변 시뮬레이션 (논리적인 예시 답변)
    # (AI #2의 '개인화된' 질문 방향성에 맞춘 시뮬레이션 답변)
    simulated_deep_dive_answers = [
        {"id": 1, "question": "(정량화) ...일주일에 현실적으로 몇 시간을 투자할 수 있나요?", 
         "answer": "네, 리포트에서 제안한 것처럼 주당 6~8시간 정도는 충분히 투자할 수 있습니다."},
        {"id": 2, "question": "(선호도) ...'통계 지식'과 '프로젝트 경험' 중 어떤 활동에 마음이 더 끌리나요?", 
         "answer": "Q23에 적었듯이, '캐글 경진대회' 같은 실제 '프로젝트 경험'을 먼저 해보고 싶습니다."},
        {"id": 3, "question": "(장벽 확인) ...'영어 어휘력' 보완을 당장 시작할 때 가장 큰 어려움은 무엇인가요?", 
         "answer": "Q22에서 말했듯이 논문 읽기는 아직 부담스럽습니다. 대신 기술 블로그를 읽는 것부터 시작하고 싶습니다."},
        {"id": 4, "question": "(학습 스타일) ...'알고리즘 공부 부족'을 보완하기 위해, '강의 수강'과 '문제 풀이' 중 어떤 방식이 더 잘 맞을 것 같나요?", 
         "answer": "저는 Q21에서처럼 일단 부딪혀보는 걸 좋아해서, '문제 풀이 사이트(백준 등)'에서 쉬운 문제부터 풀면서 모르는 개념이 나올 때마다 '온라인 기초 강의'를 찾아보는 방식이 잘 맞을 것 같습니다."},
        {"id": 5, "question": "(동기 검증) ...'미니 프로젝트'와 '캐글' 중 선호도를 묻는 질문)", 
         "answer": "Q23에 '캐글'을 적긴 했지만, 사실 Q20의 '공공데이터'나 Q21의 '교내 대회'처럼 좀 더 익숙한 환경에서 시작하고 싶습니다. 지금 당장은 '교내 경진대회 참가'나 '미니 프로젝트'를 1순위로 하고 싶습니다."}
    ]
    
    print(f"\n[사용자 심화 답변 (시뮬레이션)]\n{json.dumps(simulated_deep_dive_answers, ensure_ascii=False, indent=2)}")
    print("----------------------------------------")

    # 4) [AI #3] 최종 단기 로드맵 확정 (JSON)
    final_roadmap = generate_final_roadmap(full_report, simulated_deep_dive_answers, horizon_days=14)
    if not final_roadmap:
        print("최종 로드맵 생성 실패로 종료합니다.")
        exit(1)

    print("\n[AI #3 최종 단기 로드맵 결과 (JSON)]:")
    print(json.dumps(final_roadmap, ensure_ascii=False, indent=2))
    print("----------------------------------------")
    
    # 5) [AI #4] 직무 퀴즈 생성 (JSON)
    target_field_from_report = "데이터 사이언티스트 (AI 엔지니어)" 
    quiz = generate_quiz(target_field_from_report, final_roadmap)
    if not quiz:
        print("퀴즈 생성 실패로 종료합니다.")
        exit(1)

    print("\n[AI #4 직무 퀴즈 결과 (JSON)]:")
    print(json.dumps(quiz, ensure_ascii=False, indent=2))
    print("----------------------------------------")

    print("파이프라인 시뮬레이션 완료.")