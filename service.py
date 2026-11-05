import ai_model
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class AnalyzeImageRequest(BaseModel):
    base64_string: str


@app.post("/result")
def analyze_image(request: AnalyzeImageRequest):
    return ai_model.call_ai(request.base64_string)
