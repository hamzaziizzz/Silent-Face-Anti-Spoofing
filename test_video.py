# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test_video.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


time_list_1 = []
time_list_2 = []


def check_image(image):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test_video(video_path, model_dir, device_id):
    global time_list_1, time_list_2
    model_test = AntiSpoofPredict(device_id, "resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")
    image_cropper = CropImage()

    if video_path == '0':
        video_path = int(video_path)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (600, 800))
        frame_count += 1

        result = check_image(frame)
        if result is False:
            continue

        start_time_bbox = time.time()
        image_bbox = model_test.get_bbox(frame)
        end_time_bbox = time.time()
        time_list_1.append(end_time_bbox - start_time_bbox)
        # print(f"Time taken to get_bbox() = {end_time_bbox - start_time_bbox}")
        prediction = np.zeros((1, 3))
        test_speed = 0
        # print(prediction)
        model_name = "4_0_0_80x80_MiniFASNetV1SE.pth"
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img)
        end = time.time()
        time_list_2.append(end - start)
        # print(prediction)
        # print(f"Time for liveness predict {end - start}")

        # print(prediction)
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        # print(label, value)
        if label == 1:
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            frame,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)

        cv2.imshow('Inference', frame)

        if frame_count == 32:
            frame_count = 0
            print(f"Average time to get_bbox() = {sum(time_list_1)}")
            print(f"Average time to predict liveness = {sum(time_list_2)}")
            time_list_1, time_list_2 = [], []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    desc = "test_video"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--video_path",
        type=str,
        default="rtsp://grilsquad:grilsquad@192.168.5.1:554/stream1",
        help="video used to test")
    args = parser.parse_args()
    test_video(args.video_path, args.model_dir, args.device_id)