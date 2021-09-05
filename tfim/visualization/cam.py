from typing import List

import cv2
import numpy as np


__all__ = ["class_activation_map"]


def class_activation_map(
    img: np.ndarray,
    feature_map: np.ndarray,
    classifier_weights: np.ndarray,
    prediction: np.ndarray,
) -> List:
    img_size = img.shape[:-1]
    cam = np.zeros(dtype=np.float, shape=feature_map.shape[0:2])
    cam += np.sum(feature_map * np.squeeze(classifier_weights[:, prediction]), -1)
    cam /= np.max(cam)
    cam = cv2.resize(cam, img_size)
    heatmap = cv2.applyColorMap(np.int(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    cam = heatmap * 0.5 + img
    images = np.concatenate([img, cam], 1)
    return images
