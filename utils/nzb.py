import numpy as np


def get_plane(image, plane_num):
    return ((image & (2 ** (plane_num - 1))) / (2 ** (plane_num - 1))).astype('uint8')


def svi_1_encode(original_image, watermark, bit_plate_num, q):
    num_for_clear_bit_plate = 255 - (2 ** (bit_plate_num - 1))
    prepared_watermark = np.random.randint(0, 2, size=watermark.size).astype(np.uint8)
    image_with_empty_bit_plate = original_image & num_for_clear_bit_plate

    result = np.zeros(original_image.size)

    right_border = int(q * original_image.size)
    result[0:right_border] = prepared_watermark.flatten()[0:right_border]

    new_plane = get_plane(original_image, bit_plate_num)
    result[right_border:] = new_plane.flatten()[right_border:]

    return image_with_empty_bit_plate | ((result.astype('uint8')).reshape(original_image.shape[0], original_image.shape[1]) * (2 ** (bit_plate_num - 1)))
