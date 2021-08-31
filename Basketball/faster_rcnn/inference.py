import os
import cv2
from utils import *
import torch


def print_annotation(img, annotation):
    print(len(annotation), ' object found.')

    for object in annotation:
        object = object.cpu().detach().numpy()

        color = (0, 0, 255)
        cv2.rectangle(img, (object[0], object[1]), (object[2], object[3]), color, 10)

    return img


if __name__ == '__main__':
    video_data = "./../../data/input/videos/video1_short.mkv"
    output_path = "./../../data/output/test_results/FasterRCNN/video1_short"
    output_name = "result_{}.jpg"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model_load().to(device)
    vs = cv2.VideoCapture(video_data)

    frame_number = 1
    max_frame_number = 2
    while True:

        _, img = vs.read()
        im_tensor = image_processing(img).to(device)

        output = model(im_tensor)

        img = print_annotation(img, output[0]['boxes'])

        cv2.imwrite(os.path.join(output_path,
                                 output_name.format(frame_number)), img)
        frame_number += 1
        if frame_number > max_frame_number:
            break
