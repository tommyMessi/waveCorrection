import cv2
import numpy as np
import math
import os


image_root = '/home/huluwa/trainset/images_o'
image_output_root = '/home/huluwa/trainset/image_warp'
image_offset_root = '/home/huluwa/trainset/image_offset'


image_name_list = os.listdir(image_root)

cnt_num = 0
for image_name in image_name_list:

    image_path = os.path.join(image_root, image_name)
    print(image_path)
    image_output_path = os.path.join(image_output_root, image_name)
    image_offset_paht = os.path.join(image_offset_root, image_name)

    img = cv2.imread(image_path,
                     cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (900,1200))


    rows, cols = img.shape
    img_output = np.zeros(img.shape, dtype=img.dtype)

    img_offset = np.zeros(img.shape, dtype=img.dtype)

    o_h = np.random.randint(8, 15)
    a_h = np.random.randint(180, 220)
    for i in range(rows):
        for j in range(cols):
            offset_x = 0

            offset_y = int(o_h * math.sin(2 * 3.14 * j / a_h))
            img_offset[i,j] = offset_y
            if i + offset_y < rows:
                img_output[i, j] = img[(i + offset_y) % rows, j]
            else:
                img_output[i, j] = 0
    cv2.imwrite(image_output_path, img_output)
    cv2.imwrite(image_offset_paht, img_offset)

