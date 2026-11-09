import base64
import csv
import json
import os
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

MODEL = "gpt-4o-mini"
USAGE_LOG_PATH = "usage_log.csv"

# USD per 1M tokens: input, output, cached input
MODEL_PRICING = {
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-5.6-terra": (2.00, 12.00, 0.20),
}


def _log_usage(response) -> None:
    usage = response.usage
    if usage is None:
        return

    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    cached_tokens = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0

    pricing = MODEL_PRICING.get(MODEL)
    if pricing is None:
        input_rate = output_rate = cached_input_rate = 0.0
        estimated_cost_usd = ""
    else:
        input_rate, output_rate, cached_input_rate = pricing
        regular_input_tokens = max(input_tokens - cached_tokens, 0)
        cost = (
            regular_input_tokens * input_rate
            + cached_tokens * cached_input_rate
            + output_tokens * output_rate
        ) / 1_000_000
        estimated_cost_usd = f"{cost:.8f}"

    file_exists = os.path.exists(USAGE_LOG_PATH)
    with open(USAGE_LOG_PATH, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "model",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "total_tokens",
                "estimated_cost_usd",
            ])
        writer.writerow([
            datetime.now().astimezone().isoformat(timespec="seconds"),
            MODEL,
            input_tokens,
            cached_tokens,
            output_tokens,
            input_tokens + output_tokens,
            estimated_cost_usd,
        ])

    if estimated_cost_usd:
        print(
            f"Token usage: input={input_tokens}, output={output_tokens}; "
            f"estimated cost=${float(estimated_cost_usd):.6f}"
        )
    else:
        print(f"Token usage: input={input_tokens}, output={output_tokens}")


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
        model=MODEL,
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

    _log_usage(response)

    clean_text = response.output_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[len("```json"):]
    elif clean_text.startswith("```"):
        clean_text = clean_text[len("```"):]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]

    # Return the model's JSON object directly; FastAPI serializes it for the frontend.
    return json.loads(clean_text.strip())
