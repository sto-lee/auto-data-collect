from openai import OpenAI
from dotenv import load_dotenv
import time
import json

load_dotenv()

# 1. 시스템 프롬프트 불러오기
with open("make_SFT_prompt.txt", 'r', encoding='utf-8') as f:
    make_SFT_prompt = f.read()

with open("SFT_check_prompt.txt", 'r', encoding='utf-8') as f:
    SFT_check_prompt = f.read()

with open("make_DPO_prompt.txt", 'r', encoding='utf-8') as f:
    make_DPO_prompt = f.read()

with open("DPO_check_prompt.txt", 'r' ,encoding='utf-8') as f:
    DPO_check_prompt = f.read()

# 2. OpenAI 클라이언트 초기화
generater = OpenAI()
checker = OpenAI()

dpo_generater = OpenAI()
dpo_checker = OpenAI()

# 3. 결과 저장 리스트 및 기존 질문 집합
results = []
existing_questions = set()

collect_num = 420

# 4. 기존 질문 초기화 함수 (불러온 기존 데이터가 있을 경우)
def initialize_existing_questions(results):
    for item in results:
        for msg in item["messages"]:
            if msg["role"] == "user":
                existing_questions.add(msg["content"].strip())

# 5. 중복 질문 제거 함수
def filter_duplicates(existing_questions, items):
    valid_items = []
    for item in items:
        user_msg = next((m for m in item["messages"] if m["role"] == "user"), None)
        if user_msg and user_msg["content"].strip() not in existing_questions:
            valid_items.append(item)
    return valid_items

# 6. 생성 루프
while len(results) < collect_num:
    try:
        # Step 1: 5개 데이터 생성 요청
        gen_msg = [
            {"role": "system", "content": make_SFT_prompt},
            {"role": "user", "content": "조세호 말투 데이터를 포맷에 맞게 5개 만들어줘"}
        ]
        gen_response = generater.chat.completions.create(
            model='gpt-4o-mini',
            messages=gen_msg
        )
        generated_text = gen_response.choices[0].message.content

        # Step 2: JSON 파싱
        try:
            new_items = json.loads(generated_text)
        except Exception as e:
            print(f"[!] JSON 파싱 오류: {e}")
            continue

        # Step 3: 중복 제거
        valid_items = filter_duplicates(existing_questions, new_items)
        if not valid_items:
            print("[!] 5개 모두 중복 질문이라 건너뜀")
            continue

        # Step 4: 피드백 기반 검수 (최대 5회 반복)
        for _ in range(5):
            check_input = json.dumps(valid_items, ensure_ascii=False, indent=2)
            check_msg = [
                {"role": "system", "content": SFT_check_prompt},
                {"role": "user", "content": f"{check_input}\n\n위 데이터가 조세호 말투와 추임새, 습관을 잘 반영하는지 판단해주세요. 문제 없으면 '만족스러운 답변입니다.'라고 답변해주세요."}
            ]
            check_response = checker.chat.completions.create(
                model='gpt-4o-mini',
                messages=check_msg
            )
            feedback_text = check_response.choices[0].message.content

            if "만족스러운 답변" in feedback_text:
                results.extend(valid_items)
                for item in valid_items:
                    for msg in item["messages"]:
                        if msg["role"] == "user":
                            existing_questions.add(msg["content"].strip())
                print(f"[✓] {len(results)} / {collect_num} accepted")
                break
            else:
                # 피드백 반영 요청
                gen_msg = [
                    {"role": "system", "content": make_SFT_prompt},
                    {"role": "user", "content": feedback_text + "\n위 피드백을 반영해서 다시 5개만 생성해줘"}
                ]
                gen_response = generater.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=gen_msg
                )
                generated_text = gen_response.choices[0].message.content

                try:
                    new_items = json.loads(generated_text)
                    valid_items = filter_duplicates(existing_questions, new_items)
                    if not valid_items:
                        print("[!] 피드백 반영본도 중복 질문이라 건너뜀")
                        break
                except Exception as e:
                    print(f"[!] 피드백 반영본 JSON 파싱 실패: {e}")
                    break

        time.sleep(1.5)

    except Exception as e:
        print(f"[!] 오류 발생: {e}")
        time.sleep(5)
        continue

# 7. 결과 저장 (.jsonl 형식: 한 줄에 하나의 JSON)
with open("test_checked_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Final dataset saved to checked_dataset.jsonl")

assistant_texts = []
for it in results:
    try:
        assistant_texts.append(it["messages"][2]["content"].strip())
    except StopIteration:
        continue  # 구조가 예상과 다르면 skip

print(f"assistant_texts 길이: {len(assistant_texts)}")

dpo_items = []

for i in assistant_texts:

    gen_msgs = [
        {"role": "system", "content": make_DPO_prompt},
        {"role": "user", "content": "\n".join(i)},  # 5개 문장 전달
    ]
    resp = dpo_generater.chat.completions.create(
        model='gpt-4o-mini',
        messages=gen_msgs
    )
    dpo_line = resp.choices[0].message.content.strip()

    while True:
        check_msg = [
            {"role": "system", "content": DPO_check_prompt},
            {"role": "user", "content": dpo_line + "\n위 데이터에서 preferred_output만 보고 조세호 스타일의 답변인지만 판단해줘. 다른 값은 절대 보지 말고. 그리고 preferred_output이 괜찮다면 '통과'라고만 답해. 수정 필요시 구체적 피드백을 작성주고."},
        ]
        ok = dpo_checker.chat.completions.create(
            model='gpt-4o-mini',
            messages=check_msg
        )
        judge = ok.choices[0].message.content.strip()
        if "통과" in judge:
            item = json.dumps({"input": {"messages": [{"role": "system", "content": "들어오는 문장을 조세호 스타일 답변으로 생성"}, {"role": "user", "content": i}], "preferred_output": [{"role": "assistant", "content": dpo_line}], "non_preferred_output": [{"role": "assistant", "content": i}]}}, ensure_ascii=False)
            dpo_items.append(item)
            break
        else:
            # 피드백 기반 재생성
            feedback_msgs = [
                {"role": "system", "content": make_DPO_prompt},
                {"role": "user", "content": dpo_line + "\n" + judge + "\n위 피드백을 반영해서 다시 작성해줘"},
            ]
            resp = dpo_generater.chat.completions.create(
                model='gpt-4o-mini',
                messages=feedback_msgs
            )
            dpo_line = resp.choices[0].message.content.strip()
            time.sleep(1)

with open("create_DPO_data.jsonl", "w", encoding="utf-8") as f:
    for line in dpo_items:
        f.write(line + "\n")

print("✅ Final dataset saved to create_DPO_data.jsonl")