import keras
from dilatedUnet import UNet
from keras.models import load_model
from dilatedUnet import bce_dice_loss
from dilatedUnet import dice_coef
from dilatedUnet import r_square

import numpy as np

import cv2

model = load_model('weights/normal_unet_cv4.weights', {'r_square':r_square})

normalize = lambda x: (x - mean) / (std_dev + 1e-10)

# test = tiff.imread('images/train.tif')

test1 = cv2.imread('/000025.png')
test1 = cv2.resize(test1, (512,512))
test = test1.transpose(2,0,1)

#test.shape() (30,512,512)
mean, std_dev = np.mean(test), np.std(test)
normalize = lambda x: (x - mean) / (std_dev + 1e-10)

test = normalize(test)

#test_preds = model.predict(test)

out = model.predict(test,batch_size=10)
print(out[0])
print(test1)


test2 = cv2.imread('/000025.png', 0)
test2 = cv2.resize(test2, (512,512))
test_result = np.zeros((512,512))
out_img = out[0]*30
img_output = np.zeros(test2.shape, dtype=test2.dtype)
for i in range(511):
    for j in range(511):
        offset = int(out_img[i][j])

        # out_img[i][j] = out_img[i][j + test2[i][j]]
        test_result[i][j] = test2[i][j]

for i in range(512):
    for j in range(512):

        img_offset= out_img
        if i + out_img[i][j] < 512:
            img_output[i, j] = test2[(i - int(out_img[i][j])), j]
        else:
            img_output[i, j] = 0

cv2.imshow('4', img_output)
cv2.imshow('3', test2)
cv2.imshow('2', test1)
cv2.imshow('1', out[0])
cv2.waitKey(0)

