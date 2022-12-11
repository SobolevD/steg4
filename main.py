import os

import numpy as np

from utils.consts import M, SIGMA
from utils.in_out import read_image
from utils.nzb import svi_1_encode, get_plane
from utils.watermark import generate_watermark


def get_series_array(bit_plate, max_i):

    bit_plate_flatten = bit_plate.copy().flatten()

    result = np.zeros(max_i)

    current_bit = 0
    series_length = 0
    while current_bit < bit_plate_flatten.size:
        for series_bit in range (current_bit, bit_plate_flatten.size):
            if bit_plate_flatten[current_bit] == bit_plate_flatten[series_bit]:
                series_length += 1
            else:
                result[series_length - 1] += 1
                current_bit += series_length
                series_length = 0
                break
            if series_bit == bit_plate_flatten.size - 1:
                current_bit = bit_plate_flatten.size

    return result


if __name__ == '__main__':

    images_file_names = os.listdir('resources/images')
    bit_plate_num = 2

    K = np.array(images_file_names).size

    features_vector = []
    processed_images = []

    image_num = 0
    for image_file_name in images_file_names:
        full_image_path = 'resources/images/' + image_file_name
        image = read_image(full_image_path)

        if image_num % 10 == 0:
            print(f'Processing {image_num}%')

        image_to_process = ''
        if image_num < K / 2:
            watermark = generate_watermark(image.size, M, SIGMA).reshape(image.shape[0], image.shape[1])
            image_to_process = svi_1_encode(image, watermark, bit_plate_num)
            #write_image(image_with_watermark, 'resources/result/' + image_file_name)
        else:
            image_to_process = image

        bit_plate = get_plane(image_to_process, bit_plate_num)
        unwrapped_bit_plate = bit_plate.flatten()
        #relative_frequencies = get_relative_freq_3(unwrapped_bit_plate)
        series_array = get_series_array(bit_plate, 100)
        a = 3
        #features_vector.append(relative_frequencies)
        #processed_images.append(image_to_process)

        image_num += 1

    a = 3



