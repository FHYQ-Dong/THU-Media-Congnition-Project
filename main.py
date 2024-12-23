# main.py

import cv2
import requests
import json
import time
from yolo import YoloDetector
from camera import Camera
from botarm import ArmController
import argparse


def iou(box1, box2):
    """
    calculate the IoU of two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter_area = inter_x * inter_y

    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area


def get_n_images_inference(detector: YoloDetector, cap: Camera, num_images=5, overlap_threshold=0.8):
    """
    get `num_images` inference results from YOLO detector, and merge the results with IoU > `overlap_threshold`, to avoid the problem that yolo may miss some objects in one frame.
    """
    result = [] # pattern: [{"class": <class>, "bbox": <bbox>, "conf": <conf>}]
    for _ in range(num_images):
        frame = cap.read()
        if frame is None:
            print("Camera read failed.")
            break
        res, _ = detector.inference(frame)
        for r in res:
            for idx, obj in enumerate(result):
                if r["class"] == obj["class"] and iou(r["bbox"], obj["bbox"]) > overlap_threshold:
                    result[idx] = {
                        "class": r["class"],
                        "bbox": (r["bbox"] * r["conf"] + obj["bbox"] * obj["conf"]) / (r["conf"] + obj["conf"]),
                        "conf": obj["conf"] + r["conf"]
                    }
                    break
            else:
                result.append(r)
                print(result)
                
    return result


def call_large_language_model(task_description, class_list, api_key):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    """
    call the large language model API to help select the most suitable target.
    """
    payload = {
        "model": "Pro/Qwen/Qwen2-VL-7B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "In the following context, a json-like string will be given which has the pattern of {\"question\": <question>, \"choices\": [<choice1>, <choice2>, ...]}. You are required to choose a choice that best answers the question. You MUST NOT return `None`. Your answer should be a json-like string with the pattern of {\"choice\": <choice>}."
            },
            {
                "role": "user",
                "content": json.dumps({
                    "question": task_description,
                    "choices": class_list
                })
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            rsp_json = response.json()
            model_output = rsp_json["choices"][0]["message"]["content"]
            return model_output
        else:
            print("LLM API failed with error code: ", response.status_code, response.text)
            return None
    except Exception as e:
        print("LLM calling failed with error: ", e)
        return None


def find_bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def find_bbox_by_class(class_name, yolo_result):
    for r in yolo_result:
        if r["class"] == class_name:
            return r["bbox"]
    return None


def main(args):
    cap = Camera(args.camera_id)
    arm = ArmController(args.arm_port, "115200", r"./mapps.json")
    detector = YoloDetector(args.yolo_id, (1920, 1088), 0.25)
    
    if not cap:
        print("Camera initialization failed.")
        return

    while True:
        user_input = input("Please input your instruction: \n").strip()
        if user_input.lower() == 'q':
            print("Quitting...")
            break
        
        yolo_result = get_n_images_inference(detector, cap, num_images=10, overlap_threshold=0.8)
        print("Yolo detection result: ", yolo_result)
        if len(yolo_result) == 0:
            print("Yolo failed to detect any object, skipping.")
            continue

        selected_class = call_large_language_model(user_input, [r["class"] for r in yolo_result], args.apikey)
        if not selected_class:
            print("LLM failed to select any class, skipping.")
            continue
        print("LLM output: ", selected_class)
        
        target_bbox = find_bbox_by_class(json.loads(selected_class)["choice"], yolo_result)
        if target_bbox is None:
            print("Failed to find target bbox, skipping.")
            continue
        target_center = find_bbox_center(target_bbox)
        print("Target center: ", target_center)
        
        arm.pick(target_center[0], target_center[1], speed=100)
        time.sleep(3)
        
        bin_kind = "grey"
        for color in ["red", "green", "blue", "grey"]:
            if color in user_input:
                bin_kind = color
                break
        arm.drop(bin_kind, speed=100)

    cap.release()


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", "-k", type=str, required=True, help="API key for the LLM API")
    parser.add_argument("--camera_id", "-c", type=int, default=0, help="Camera ID, default is 0")
    parser.add_argument("--arm_port", "-p", type=str, default="COM5", help="Port for the robot arm, default is COM5")
    parser.add_argument("--yolo_id", "-y", type=str, default="yolov10x", help="Yolo detector ID, default is yolov10x")
    main(parser.parse_args())
