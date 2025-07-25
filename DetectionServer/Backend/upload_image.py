from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil

app = FastAPI()

upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "请上传图片文件"})

    save_path = upload_dir / file.filename
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "message": "上传成功", "path": str(save_path)}
