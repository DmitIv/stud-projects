import os
import sys
import cv2

import numpy as np
import matplotlib.pyplot as plt
import glob

from sklearn.utils.linear_assignment_ import linear_assignment

ROOT_DIR = os.path.abspath("./PersonDetectionandTracking")
sys.path.append(ROOT_DIR)

import helpers as helpers
import detector as detector
import tracker as tracker

ROOT_DIR = os.path.abspath("./Mask_RCNN")
sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
# from mrcnn import visualize
# Import COCO config
import samples.coco.coco as coco

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = "./../models_dumps/MaskRCNN/mask_rcnn_coco.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = "./../data/input/images/frames_video1_short"
VIDEO_DIR = "./../data/input/videos"
VIDEO_FILE = "video1_short.mkv"

VIDEO_PATH = os.path.join(VIDEO_DIR, VIDEO_FILE)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0  # frame counter

# frames_per_second = 30
# t = 1 # time period in seconds for max_age
max_age = 3 # no.of consecutive unmatched detection before
# a track is deleted

min_hits = 1  # no. of consecutive matches needed to establish a track

tracker_list = []  # list for trackers
# list for track ID
last_track_id = 0

# Kalman filter for scene switching detetcction
F = 1
R = 1
Q = 0
H = 1
B = 0

def track_id_list():
    global last_track_id
    last_track_id += 1
    return str(last_track_id)


# track_id_list= deque(['1', '2', '3', '4', '5', '6', '7', '7', '8', '9', '10'])

debug = False


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''

    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        # trk = convert_to_cv2bbox(trk)
        for d, det in enumerate(detections):
            #   det = convert_to_cv2bbox(det)
            IOU_mat[t, d] = helpers.distance(trk, det)

            # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if (IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


scene_number = 0


def pipeline(model, img, new_scene):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global debug
    global scene_number

    frame_count += 1
    result = model.detect([img], verbose=1)[0]
    z_box = []  # measurement
    for ind, bbox in enumerate(result['rois']):
        if result['class_ids'][ind] == 1:
            z_box.append(bbox)
    x_box = []

    if new_scene:
        del tracker_list
        tracker_list = []
        scene_number += 1
        path = './../data/output/test_results/video1_short/PersonTracker_objects/scene{0}'.format(
            scene_number
        )
        if not os.path.exists(path):
            os.mkdir(path)

        cv2.imwrite(os.path.join(path, 'scene_start.jpg'), img)

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks \
        = assign_detections_to_trackers(x_box, z_box, iou_thrd=0.3)

    # Deal with matched detections
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            # tmp_trk.box = xx
            z_list = [z[0, 0], z[1, 0], z[2, 0], z[3, 0]]
            tmp_trk.box = z_list
            tmp_trk.hits += 1

    # Deal with unmatched detections
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            z_list = [z[0, 0], z[1, 0], z[2, 0], z[3, 0]]
            tmp_trk.box = z_list
            # tmp_trk.box = xx
            tmp_trk.id = track_id_list()
            print(tmp_trk.id)
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            z = np.array(tmp_trk.box)
            z = np.expand_dims(z, 0).T
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx

    # The list of tracks to be annotated
    good_tracker_list = []
    white_canvas = np.ones(img.shape) * 255
    image_without_annotation = img.copy()
    for trk in tracker_list:
        if trk.hits >= min_hits and trk.no_losses <= max_age:
            good_tracker_list.append(trk)
            x_cv2 = trk.box

            if (frame_count % 10) == 0:

                path = './../data/output/test_results/video1_short/PersonTracker_objects/scene{0}'.format(
                    scene_number
                )
                path = os.path.join(path, 'object{}'.format(trk.id))
                if not os.path.exists(path):
                    os.mkdir(path)

                object_on_image = image_without_annotation[x_cv2[0]:x_cv2[2], x_cv2[1]:x_cv2[3]]
                cv2.imwrite(os.path.join(path, 'frame{}.jpg'.format(frame_count)), object_on_image)

            img = helpers.draw_box_label(trk.id, img, x_cv2)  # Draw the bounding boxes on the
            center = (x_cv2[1] + x_cv2[3]) // 2, (x_cv2[0] + x_cv2[2]) // 2
            white_canvas = cv2.circle(white_canvas, center, 10, (0, 0, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.4
            font_color = (0, 0, 0)
            text_x = 'id=' + str(trk.id)
            cv2.putText(white_canvas, text_x, (center[0] - 25, center[1] + 25), font, font_size, font_color, 1,
                        cv2.LINE_AA)

    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]

    # cv2.imshow("frame", img)
    return img, white_canvas


def calc3colorsHist(image):
    chanels = cv2.split(image)
    hist = cv2.calcHist([chanels[0].astype(np.float32)], [0], None, [256], [0, 256])
    for chan in chanels[1:]:
        h = cv2.calcHist([chan.astype(np.float32)], [0], None, [256], [0, 256])
        hist = np.concatenate([hist, h], axis=-1)

    return hist


def correct(x, cov):
    global F
    global Q
    x_next = F * x
    cov_next = F * cov * F + Q

    return x_next, cov_next


def update(x, cov, data):
    # measurement update - correction

    global H
    global R

    K = H * cov / (H * cov * H + R)
    x = x + K * (data - H * x)
    cov = (1 - K * H) * cov

    return x, cov


if __name__ == "__main__":

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    cap = cv2.VideoCapture('./../data/input/videos/video1_short.mkv')

    frame_number = 0
    last_track_id = 0
    first_image = True
    scene_switching_border = 1
    flash_border = 4
    scene_switch = True
    while True:

        ret, img = cap.read()
        frame_number += 1
        print("Frame number {}".format(frame_number))
        if first_image:
            hist_prev = calc3colorsHist(img)
            cov = 0.1
            first_image = False
        else:
            hist_cur = calc3colorsHist(img)
            pred_hist, cov = correct(hist_prev, cov)
            pred_hist, cov = update(pred_hist, cov, hist_cur)

            equal_metric = np.sqrt(
                    np.divide(np.sum(np.power(np.subtract(hist_cur, pred_hist), 2)),
                              hist_cur.shape[0])) / np.mean(hist_cur)
            is_flash = equal_metric > flash_border
            scene_switch = not is_flash and equal_metric > scene_switching_border
            if scene_switch:
                last_track_id = 0
                cov = 0.1
                print('Scene switch.')

            if not is_flash:
                hist_prev = hist_cur

        np.asarray(img)
        new_img, centers = pipeline(model, img, scene_switch)
        cv2.imwrite('./../data/output/test_results/video1_short/PersonTracker/frame_{}.jpg'.format(frame_number),
                    new_img)
        cv2.imwrite('./../data/output/test_results/video1_short/PersonTracker_centers/frame_{}.jpg'.format(frame_number),
                    centers)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

