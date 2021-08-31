from fastai.vision import plt
import cv2
import numpy as np
import skimage.io as io
import os


def show_results(td, learner):
    size = len(td.test_ds)
    _, axs = plt.subplots(size, 3, figsize=(3, size))
    for num, image in enumerate(td.test_ds):
        pred, _, _ = learner.predict(image[0])
        _ax = axs
        if size > 1:
            _ax = _ax[num]

        image[0].show(ax=_ax[0], title='no mask', figsize=(100, 100))
        image[0].show(ax=_ax[1], y=pred, title='masked', figsize=(100, 100))
        pred.show(ax=_ax[2], title='mask only', alpha=1., figsize=(100, 100), cmap='binary')


AREA_THRESHOLD = 600
result_path = "/home/dmitri/dev/BachelorDiploma/thesis/mmcs_sfedu_thesis/img/mask_results"


def save_results(td, learner):
    size = len(td.test_ds)
    _, axs = plt.subplots(size, 3, figsize=(3, size))
    for num, image in enumerate(td.test_ds):
        _, pred, _ = learner.predict(image[0])

        lable_np_array = pred.numpy() * 254

        blur = cv2.GaussianBlur(lable_np_array.astype(np.uint8), (5, 5), 0).squeeze(0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Contour detection
        _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contour by area
        contour_list = []
        area_threshold = AREA_THRESHOLD

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                contour_list.append(contour)

        # Making binary mask by contour
        countMask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(countMask, contour_list, -1, (255, 255, 255), cv2.FILLED)

        # Restore accuracy of board between sky and other objects
        mask_rev = np.add(cv2.bitwise_not(thresh), cv2.bitwise_not(countMask))
        _, mask_rev = cv2.threshold(mask_rev, 0, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask_rev)

        img = image[0].data.permute(1, 2, 0).numpy()
        # print(img)
        img_mask = img.copy()
        img_mask[cv2.bitwise_not(mask) == 0] = [1, 0, 0]
        # print(img_mask)
        masked = cv2.addWeighted(img, 0.5, img_mask, 0.4, 0.5)

        io.imsave(os.path.join(result_path, '%d_image.png' % num), img)
        io.imsave(os.path.join(result_path, '%d_masked.png' % num), masked)
        io.imsave(os.path.join(result_path, '%d_mask.png' % num), mask)
