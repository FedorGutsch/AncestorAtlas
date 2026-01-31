from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
from pathlib import Path
import uvicorn
from PIL import Image
import io

from face_detection import analyze_photo

app = FastAPI(title="Face Analysis Demo")

templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze/")
async def analyze_photo_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {"error": "Файл должен быть изображением"}
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Конвертируем RGBA → RGB чтобы избежать ошибки при сохранении как JPEG
    if img.mode in ('RGBA', 'LA', 'P'):
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'RGBA':
            rgb_img.paste(img, mask=img.split()[-1])
        else:
            rgb_img.paste(img)
        img = rgb_img
    
    # Сохраняем как JPEG (без альфа-канала)
    file_path = f"temp_{file.filename}.jpg"
    img.save(file_path, 'JPEG', quality=95)
    
    result = analyze_photo(file_path)
    return result


if __name__ == "__main__":
    uvicorn.run(app=app)