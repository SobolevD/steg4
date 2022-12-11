import numpy as np


def get_plane(channel_image, plane_num):
    return channel_image & (2 ** (plane_num - 1))


def svi_1_encode_changed(original_image, watermark, bit_plate_num):
    num_for_clear_bit_plate = 255 - (2 ** (bit_plate_num - 1))
    prepared_watermark_colored = ((watermark / 255) * (2 ** (bit_plate_num - 1))).astype(np.uint8)
    channel_with_empty_bit_plate = original_image & num_for_clear_bit_plate

    return channel_with_empty_bit_plate | prepared_watermark_colored
