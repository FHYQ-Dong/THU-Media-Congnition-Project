from camera import Camera
import cv2
import numpy as np
import time
import json
import imutils
from imutils import contours, perspective
from llmapi import LLMAPI
from botarm import ArmController
import argparse


def cut_image(image, x1, y1, x2, y2, image_show=False):
    """
    Cut the image to the interested region
    """
    image = image.copy()
    image = image[y1:y2, x1:x2]
    if image_show:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


def detect_rects(image, image_show=False): 
    """
    Detect the rectangles in the image
    """
    # gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    if image_show:
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts) # sort the contours

    # save the sub-images and spin them
    boxes = []
    splitted_images = []
    for c in cnts:
        # discard the small contours
        if cv2.contourArea(c) < 100:
           continue
        # calculate the minimum "bounding box" (sliding rectangle)
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        
        # calculate the width and height of the bounding box
        width = int(((box[1][0] - box[0][0]) ** 2 + (box[1][1] - box[0][1]) ** 2) ** 0.5)
        height = int(((box[2][0] - box[1][0]) ** 2 + (box[2][1] - box[1][1]) ** 2) ** 0.5)
        
        # spin the sub-image
        # spin matrix
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # warp the image
        warped = cv2.warpPerspective(image, M, (width, height))
        # discard the small images (the box-recgion may generate wrong target)
        if warped.shape[0] < 32 or warped.shape[1] < 32: 
            continue
        
        # add the sub-image to the list
        splitted_images.append(warped)
        
        if image_show:
            cv2.imwrite("images/cropped_rotated.jpg", warped)
            cv2.imshow("Cropped Rotated Image", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        boxes.append(box)
    return image, splitted_images, boxes


def get_llm_response(prompt, jpeg_bytes_list, llm_agent, print_response=False):
    """
    Get the response from the LLM API
    """
    resp = llm_agent.analyse_images(prompt, jpeg_bytes_list)
    if print_response:
        print(resp)
        print(resp.choices[0].message.content)
    return json.loads(resp.choices[0].message.content)["selected_images"]


def find_bbox_center(bbox):
    """
    Find the center of the bounding box
    """
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = bbox
    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4
    return (center_x, center_y)


def get_selected_target(image, selected_images_idx, boxes, show_image=False):
    """
    Get the selected target according to the selected images
    """
    for idx in selected_images_idx:
        cv2.drawContours(image, [boxes[idx].astype("int")], -1, (0, 255, 0), 1)
    if show_image:
        cv2.imwrite("images/selected_target.jpg", image)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image, boxes


def decide_bin(prompt):
    """
    Decide the bin kind according to the prompt
    """
    bin_kind = "grey"
    for color in ["red bin", "green bin", "blue bin", "grey bin"]:
        if color in prompt:
            bin_kind = color.split(" ")[0]
            break
    return bin_kind


def main(args):
    # initialize the camera, arm and llm agent
    cap = Camera(args.camera_id)
    arm = ArmController(args.arm_port, "115200", r"./mapps.json")
    llm_agent = LLMAPI(apikey=args.apikey)
    
    while True:
        # capture image, cut the image, detect the rectangles
        image = cap.read()
        image = cut_image(image, 184, 104, 481, 385, True)
        image, splitted_images, boxes = detect_rects(image, True)

        # splitted_images is MatLike type, convert it to jpg bytes
        image = image.copy()
        jpeg_bytes_list = [cv2.imencode('.jpg', img)[1].tobytes() for img in splitted_images]

        # get the prompt and selected images
        prompt = input("Please input the prompt: ")
        if prompt == "exit":
            break
        selected_images_idx = get_llm_response(prompt, jpeg_bytes_list, llm_agent)
        image = image.copy()
        image, boxes = get_selected_target(image, selected_images_idx, boxes, True)
        bbox_centers = [find_bbox_center(boxes[idx]) for idx in selected_images_idx]

        # move the arm
        for i in range(len(bbox_centers)):
            arm.pick(float(bbox_centers[i][0]) + 235, float(bbox_centers[i][1]) + 160, speed=100)
            time.sleep(3)
            bin_kind = decide_bin(prompt)
            arm.drop(bin_kind, speed=100)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", "-k", type=str, required=True, help="API key for the LLM API")
    parser.add_argument("--camera_id", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--arm_port", "-p", type=str, default="COM5", help="Port for the robot arm")
    main(parser.parse_args())
