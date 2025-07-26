from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from asyncio import Queue
from infer import init_model, draw_detections, infer as model_infer
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()
model = init_model()
image_queue = Queue()


@app.post('/upload/')
async def upload(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return JSONResponse(status_code=400, content={'error': '请上传图片文件'})
    await image_queue.put(Image.open(file.file).copy())  # 将图片保存进内存
    return {'message': '成功上传图片'}


@app.get('/infer/')
async def infer():
    if image_queue.empty():
        return JSONResponse(status_code=400, content={'error': '请先上传图片'})

    image = await image_queue.get()

    results = model_infer(image, model)
    processed_image = draw_detections(image, results, save=False)

    # 将处理后的图片转化为Base64字符串
    buf = BytesIO()
    processed_image.save(buf, format='JPEG')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {
        'results': [
            {
                'type': r['type'],  # 需要把tensor转化为float，下同
                'conf': float(r['conf']),
                'xyxy': [float(v) for v in r['xyxy']]
            }
            for r in results
        ],
        'image_base64': img_base64
    }
