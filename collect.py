from openai import OpenAI
from dotenv import load_dotenv

import ast
import time
import json
import re

load_dotenv()

# Load system prompts
with open("make_SFT_prompt.txt", 'r', encoding='utf-8') as f:
    make_SFT_prompt = f.read()

with open("make_DPO_prompt.txt", 'r', encoding='utf-8') as f:
    make_DPO_prompt = f.read()

with open("SFT_check_prompt.txt", 'r', encoding='utf-8') as f:
    SFT_check_prompt = f.read()

with open("DPO_check_prompt.txt", 'r', encoding='utf-8') as f:
    DPO_check_prompt = f.read()

# Initialize clients
sft_generator = OpenAI()
sft_checker = OpenAI()
dpo_generator = OpenAI()
dpo_checker = OpenAI()
gen_quest = OpenAI()

# 저장용 리스트 및 질문 집합
questions = []
existing_questions = set()

# 수집할 데이터 개수
num_of_data = 5

# 질문 생성
while len(questions) < num_of_data:
    try:
        gen_quest_msg = [
            {"role": "system", "content": "당신은 특정 인물에게 할 질문을 생성하는 질문 생성기입니다. 질문은 파이썬의 리스트 형태로 저장할 수 있도록 생성해주세요 (ex. ['질문 1','질문 2'])"},
            {"role": "user", "content": "조세호에게 할 질문 5개를 생성해주세요"}
        ]
        gen_question = gen_quest.chat.completions.create(
            model='gpt-4o-mini',
            messages=gen_quest_msg
        )
        try:
            new_qs = ast.literal_eval(gen_question.choices[0].message.content)
        except Exception:
            raw = gen_question.choices[0].message.content.replace("[", "").replace("]", "")
            new_qs = [q.strip().strip("'\"") for q in raw.split(",") if q.strip()]

        for q in new_qs:
            q_clean = q.strip()
            if q_clean and q_clean not in existing_questions:
                questions.append(q_clean)
                existing_questions.add(q_clean)
        
    except Exception as e:
        print(f"[!] Error occurred: {e}")
        time.sleep(5)
        continue

print(f"질문 수: {len(questions)}")

# 질문 txt로 저장
with open("questions.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(questions))
print(f"✅ 질문 {len(questions)}개를 questions.txt 파일로 저장했습니다.")

# SFT 데이터 생성
sft_items = []

for i in range(0, len(questions), 5):
    batch = questions[i : i + 5]  # 최대 5개씩 처리
    print(f"input data: {batch}")

    try:
        gen_msg = [
            {"role": "system", "content": make_SFT_prompt},
            {"role": "user", "content": "\n".join(batch)},
        ]
        gen_response = sft_generator.chat.completions.create(
            model="gpt-4o-mini",
            messages=gen_msg
        )
        args = json.loads(gen_response.choices[0].message.content)
        generated_items = args
        print(f"gg {generated_items}")

        # 검수 루프
        while True:
            check_msg = [
                {"role": "system", "content": SFT_check_prompt},
                {
                    "role": "user",
                    "content": (
                        json.dumps(generated_items, ensure_ascii=False, indent=2)
                        + "\n위 데이터가 조세호 스타일로 생성되었는지 확인해줘. 문제 없으면 '만족스러운 답변입니다.'라고 말해줘."
                    ),
                },
            ]

            check_response = sft_checker.chat.completions.create(
                model="gpt-4o-mini",
                messages=check_msg
            )
            feedback = check_response.choices[0].message.content

            if "만족스러운 답변" in feedback:
                sft_items.extend(generated_items)
                print(
                    f"    ↳ Accepted: {len(sft_items):>3} / {num_of_data} (batch {i//5 + 1})"
                )
                break
            else:
                # 피드백 반영 재생성
                gen_msg = [
                    {"role": "system", "content": make_SFT_prompt},
                    {
                        "role": "user",
                        "content": feedback + "\n위 피드백을 반영해서 답변을 수정해줘",
                    },
                ]
                gen_response = sft_generator.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=gen_msg
                )
                print(gen_response.choices[0].message.content)
                args = json.loads(gen_response.choices[0].message.content)
                generated_items = args

        time.sleep(1.5)

    except Exception as e:
        print(f"[!] Error while creating SFT data: {e}")
        time.sleep(5)
        continue

with open("create_SFT_data.jsonl", "w", encoding="utf-8") as f:
    for item in sft_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# SFT 데이터 생성 끝
print("✅ Final dataset saved to create_SFT_data.jsonl")

# --------------------
# DPO 데이터 생성
dpo_items = []

# SFT assistant 답변 추출
assistant_texts = []
for it in sft_items:
    print(f"it: {it}")
    try:
        print(it["messages"][2]["content"])
        assistant_texts.append(it["messages"][2]["content"].strip())
    except StopIteration:
        continue  # 구조가 예상과 다르면 skip

print(f"assistant_texts 길이: {len(assistant_texts)}")

for i in range(0, len(assistant_texts), 5):
    batch = assistant_texts[i : i + 5]
    print(f"dpo input: {batch}")

    # ---------------------- 생성 ----------------------
    gen_msgs = [
        {"role": "system", "content": make_DPO_prompt},
        {"role": "user", "content": "\n".join(batch)},  # 5개 문장 전달
    ]
    resp = dpo_generator.chat.completions.create(
        model='gpt-4o-mini',
        messages=gen_msgs
    )
    dpo_line = resp.choices[0].message.content.strip()
    print(f"dpo_line: {dpo_line}")

    # ---------------------- 검수 & 피드백 루프 ----------------------
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
            dpo_items.append(dpo_line)
            print(f"    ↳ DPO batch {i//5 + 1} 저장")
            break
        else:
            # 피드백 기반 재생성
            feedback_msgs = [
                {"role": "system", "content": make_DPO_prompt},
                {"role": "user", "content": judge + "\n위 피드백을 반영해서 다시 작성해줘"},
            ]
            resp = dpo_generator.chat.completions.create(
                model='gpt-4o-mini',
                messages=feedback_msgs
            )
            dpo_line = resp.choices[0].message.content.strip()
            time.sleep(1)

with open("create_DPO_data.jsonl", "w", encoding="utf-8") as f:
    for line in dpo_items:
        f.write(line + "\n")

print("✅ Final dataset saved to create_DPO_data.jsonl")