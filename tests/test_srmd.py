import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from srmd_ncnn_py import SRMD

print("System version: ", sys.version)

filePATH = Path(__file__).resolve().absolute()

print("filePATH: ", filePATH)


def calculate_image_similarity(image1: np.ndarray, image2: np.ndarray) -> bool:
    # Resize the two images to the same size
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    # Convert the images to grayscale
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate the Structural Similarity Index (SSIM) between the two images
    (score, diff) = structural_similarity(grayscale_image1, grayscale_image2, full=True)
    print("SSIM: {}".format(score))
    return bool(score > 0.8)


_gpuid = -1

TEST_IMG = cv2.imread(str(filePATH.parent / "test.png"))


class Test_SRMD:
    def test_no_denoise(self) -> None:
        _scale = 2
        _noise = -1
        srmd = SRMD(gpuid=_gpuid, scale=_scale, noise=_noise)
        outimg = srmd.process_cv2(TEST_IMG)
        assert calculate_image_similarity(TEST_IMG, outimg)

    def test_denoise(self) -> None:
        _scale = 3
        _noise = 0
        srmd = SRMD(gpuid=_gpuid, scale=_scale, noise=_noise)
        outimg = srmd.process_cv2(TEST_IMG)
        assert calculate_image_similarity(TEST_IMG, outimg)
