import base64
import json
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import os


def call_ai(base64_string: str) -> dict:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Accept either a raw base64 payload or a data URL from the frontend.
    if "," in base64_string and base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_string)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "이 노트의 내용을 분석해서 전체 문자와 중심 내용을 "
                            "다음 JSON 형식으로 답해줘: "
                            '{"text": "출력된 전체 텍스트", '
                            '"important": "중요한 텍스트와 왜 중요한지 정리한 것"}'
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded_image}",
                    },
                ],
            }
        ],
    )

    clean_text = response.output_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[len("```json"):]
    elif clean_text.startswith("```"):
        clean_text = clean_text[len("```"):]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]

    # Return the model's JSON object directly; FastAPI serializes it for the frontend.
    return json.loads(clean_text.strip())