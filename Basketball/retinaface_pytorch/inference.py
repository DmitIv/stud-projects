import os
import cv2
from detector import Retinaface_Detector
import numpy as np


def print_annotation(img, annotation):
    results = annotation
    print(len(results), ' faces found.')

    for result in results:
        face = result[0]
        landmark = result[1]

        color = (0, 0, 255)
        cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), color, 2)

        for l in range(landmark.shape[0]):
            color = (0, 0, 255)
            if l == 0 or l == 3:
                color = (0, 255, 0)
            cv2.circle(img, (landmark[l][0], landmark[l][1]), 1, color, 2)

    return img

detector = Retinaface_Detector()
video_data = "./../../data/input/videos/video1_short.mkv"
output_path = "./../../data/output/test_results/video1_short/RetinaFace"
vs = cv2.VideoCapture(video_data)

output_name = "result_{}.jpg"
frame_number = 1
max_frame_number = np.Inf
while True:

    _, img = vs.read()
    img_parts = [img[:img.shape[0] // 2, :img.shape[1] // 3],
                 img[:img.shape[0] // 2, img.shape[1] // 3:2 * img.shape[1] // 3],
                 img[:img.shape[0] // 2, 2 * img.shape[1] // 3:],
                 img[img.shape[0] // 2:, :img.shape[1] // 3],
                 img[img.shape[0] // 2:, img.shape[1] // 3:2 * img.shape[1] // 3],
                 img[img.shape[0] // 2:, 2 * img.shape[1] // 3:]]


    img_part_with_annot = []
    for i in range(len(img_parts)):
        img = img_parts[i]
        results = detector.detect(img)

        img_with_annotation = print_annotation(img, results)
        img_part_with_annot.append(img_with_annotation)

    img_up_part = np.concatenate([img_part_with_annot[0], img_part_with_annot[1],
                                  img_part_with_annot[2]], axis=1)
    img_bottom_part = np.concatenate([img_part_with_annot[3], img_part_with_annot[4],
                                      img_part_with_annot[5]], axis=1)
    img = np.concatenate([img_up_part, img_bottom_part], axis=0)

    cv2.imwrite(os.path.join(output_path,
                             output_name.format(frame_number)), img)
    frame_number += 1
    if frame_number > max_frame_number:
        break
