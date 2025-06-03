from openai import OpenAI
from dotenv import load_dotenv
import time
import json

load_dotenv()

# Load system prompts
with open("speechstyle.txt", 'r', encoding='utf-8') as f:
    prompt = f.read()

with open("check_prompt.txt", 'r', encoding='utf-8') as f:
    check_prompt = f.read()

# Initialize OpenAI clients
generater = OpenAI()
checker = OpenAI()

results = []

while len(results) < 300:
    try:
        # Initial generation
        gen_msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "조세호 말투 데이터를 포맷에 맞게 5개 만들어줘"}
        ]
        gen_response = generater.chat.completions.create(
            model='gpt-4o-mini',
            messages=gen_msg
        )
        generated_text = gen_response.choices[0].message.content

        # Loop until checker confirms quality
        for _ in range(5):  # max 5 retries
            check_msg = [
                {"role": "system", "content": check_prompt},
                {"role": "user", "content": f"{generated_text}\n해당 데이터의 답변들이 조세호의 말투, 추임새 특정과 습관이 반영되었는지 판단하고 만약 부족하다면 피드백을 해주세요"}
            ]
            check_response = checker.chat.completions.create(
                model='gpt-4o-mini',
                messages=check_msg
            )
            feedback_text = check_response.choices[0].message.content

            if any(kw in feedback_text for kw in ["적절", "충분", "잘 반영"]):
                results.append({"generated": generated_text})
                print(f"[✓] {len(results)} / 300 accepted")
                break
            else:
                gen_msg.append({"role": "assistant", "content": generated_text})
                gen_msg.append({"role": "user", "content": feedback_text + "\n피드백을 반영해서 다시 생성해줘"})

                gen_response = generater.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=gen_msg
                )
                generated_text = gen_response.choices[0].message.content

        time.sleep(1.5)

    except Exception as e:
        print(f"[!] Error occurred: {e}")
        time.sleep(5)
        continue

# Save results only (without feedback)
with open("checked_dataset.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Final dataset saved to checked_dataset.json")