from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

with open("speechstyle.txt", 'r', encoding='utf-8') as f:
    prompt = f.read()

with open("check_prompt.txt", 'r', encoding='utf-8') as f:
    check_prompt = f.read()

generater = OpenAI()
checker = OpenAI()

response = generater.responses.create(
    model='gpt-4o-mini',
    input=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": "조세호 말투 데이터를 포맷에 맞게 5개 만들어줘"
        }
    ]
)
print(response.output[0].content[0].text)

print("-----------------------------------------------")

check = checker.responses.create(
    model='gpt-4o-mini',
    input=[
        {
            "role": "system",
            "content": check_prompt
        },
        {
            "role": "user",
            "content": response.output[0].content[0].text + "\n해당 데이터의 답변들이 조세호의 말투, 추임새 특정과 습관이 반영되었는지 판단하고 만약 부족하다면 피드백을 해주세요"
        }
    ]
)
print(check.output[0].content[0].text)