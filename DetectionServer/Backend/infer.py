"""
This script utilizes an Ultralytics pre-trained model to predict an image, and
returns the matches expected by the user.
"""

from functools import lru_cache
from pathlib import Path
from torch import nn
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

model_path = Path('Models/Yolov8n.pt')


@lru_cache()  # 确保模型只加载一次
def init_model() -> nn.Module:
    if model_path.is_file():
        return YOLO(model_path)
    model_path.parent.mkdir(exist_ok=True)
    return YOLO(model_path)


def infer(image: Path or Image, model: nn.Module, threshold: float = 0.5) -> list:
    preds, results = model(image)[0], []
    boxes, names = preds.boxes, model.names
    for i in range(len(boxes)):
        conf = boxes.conf[i]
        if conf < threshold:  # 去除置信度较低的结果
            continue
        results.append({
            'type': names[boxes.cls[i].item()],  # 预测对象类型
            'conf': conf,  # 置信度
            'xyxy': boxes.xyxy[i],  # 坐标(x1, y1, x2, y2)
        })
    return results


def draw_detections(image: Path or Image, detections: list, save: bool = False) -> Image:
    if not isinstance(image, Image.Image):
        image_path = image
        image = Image.open(image).convert("RGB")
    else:
        image_path = 'Uploads/sample.jpg'
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=32)
    except:
        font = ImageFont.load_default(size=32)

    for det in detections:  # 绘制边界框、类型和置信度
        label = f"{det['type']} {det['conf']:.2f}"
        xyxy = det['xyxy']
        x1, y1, x2, y2 = map(int, xyxy)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, y1 - 18), label, fill="yellow", font=font)

    if save:
        image.save(image_path.parent / Path(
            image_path.name.split('.')[0] + '.pred' + image_path.suffix
        ))
    return image


if __name__ == '__main__':
    model, image = init_model(), Path('Images/Workers.jpg')
    results = infer(image, model, threshold=0.5)
    draw_detections(image, results, True)
