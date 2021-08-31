import numpy as np
import cv2
import torch
import torchvision


def image_processing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    tensor = torch.from_numpy(image).float()
    return tensor


def model_load():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model
