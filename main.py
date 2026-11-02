from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json
from prompts import SYSTEM_PROMPT

# Подключаемся к локальному серверу LM Studio
client = OpenAI(
    api_key="lm-studio", 
    base_url="http://localhost:1234/v1"
)

app = FastAPI(
    title="Product Card Generator",
    description="Сервис для генерации карточек товаров (КИП) с помощью LLM"
)

class Product(BaseModel):
    name: str
    manufacturer: str
    specs: dict
    description: str

@app.post("/generate")
def generate_card(product: Product):
    try:
        response = client.chat.completions.create(
            model="qwen2.5-coder-14b-instruct", # Убедись, что имя модели совпадает с LM Studio
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Данные прибора:\n{product.model_dump_json(indent=2)}"}
            ],
            temperature=0.1
            # Убрали response_format, так как LM Studio его не поддерживает
        )
        
        # Достаем ответ
        content = response.choices[0].message.content.strip()
        
        # Защита: локальные модели иногда оборачивают JSON в ```json ... ```
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Парсим в JSON
        result_json = json.loads(content)
        return result_json
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Модель вернула невалидный JSON: {content}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Перейдите по адресу /docs для тестирования сервиса"}