from ultralytics import YOLOv10
from PIL import Image
import torch


class YoloDetector:
    def __init__(self, 
                 model_id, 
                 image_size, 
                 conf_threshold, 
                 device=None):
        self.model_id = model_id
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model_id={model_id}, image_size={image_size}, conf_threshold={conf_threshold} ...", end=" ")
        self.model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
        print("done.")

    def inference(self, image):
        results = self.model.predict(source=image, imgsz=self.image_size, conf=self.conf_threshold, device=self.device)
        classes = results[0].boxes.cls
        class_names = results[0].names
        bboxes = results[0].boxes.xywh
        conf = results[0].boxes.conf
        annotated_image = results[0].plot()
        return [{"class": class_names[int(c)], "bbox": b, "conf": co} \
            for c, b, co in zip(classes, bboxes, conf)], annotated_image
    
    def __test__(self):
        image = Image.open("test/test.jpg")
        return self.inference(image)
    
    
if __name__ == "__main__":
    detector = YoloDetector("yolov10m", 640, 0.25, device="cuda")
    res, _ = detector.__test__()
    print(res)
    print(float(res[0]['bbox'][0]))