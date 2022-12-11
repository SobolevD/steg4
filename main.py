import os

import cv2
import numpy

from utils.consts import M, SIGMA
from utils.in_out import read_image, write_image
from utils.nzb import svi_1_encode, svi_1_encode_changed
from utils.watermark import generate_watermark

if __name__ == '__main__':
    images = os.listdir('resources/images')

    for image_file_name in images:

        full_image_path = 'resources/images/' + image_file_name
        image = read_image(full_image_path)
        watermark = generate_watermark(image.size, M, SIGMA).reshape(image.shape[0], image.shape[1])

        result = svi_1_encode_changed(image, watermark, 7)
        write_image(result, 'resources/result/' + image_file_name)
        a = 3


