import numpy as np
np.random.seed(865)

from keras.models import Model
from keras.layers import Input, BatchNormalization, merge, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import  Dropout, concatenate, Conv2DTranspose, Lambda, Reshape, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
from os import path, makedirs
import argparse
import keras.backend as K
import logging
import pickle
import os
import sys
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy
from clr_callback import *
import cv2
import tifffile as tiff

def r_square(y_true, y_pred):
    # SSR = K.mean(K.square(y_pred-K.mean(y_true)),axis=-1)
    # SST = K.mean(K.square(y_true-K.mean(y_true)),axis=-1)
    # return SSR/SST
    return K.mean(K.square(y_pred - y_true), axis=-1)

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -1.0 * dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

class UNet():

    def __init__(self):
        self.net = None

    def load_data(self,image_root, mask_root):
        # imgs, msks = tiff.imread(image_path+'/train.tif'), tiff.imread(mask_path+'/label.tif') / 255
        #
        #
        #
        # montage_imgs = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.float32)
        # montage_msks = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.int8)
        # idxs = np.arange(imgs.shape[0])
        # np.random.shuffle(idxs)
        # idxs = iter(idxs)
        # for y0 in range(0, montage_imgs.shape[0], imgs.shape[1]):
        #     for x0 in range(0, montage_imgs.shape[1], imgs.shape[2]):
        #         y1, x1 = y0 + imgs.shape[1], x0 + imgs.shape[2]
        #         idx = next(idxs)
        #         montage_imgs[y0:y1, x0:x1] = imgs[idx]
        #         montage_msks[y0:y1, x0:x1] = msks[idx]
        image_names = os.listdir(image_root)
        montage_imgs = []
        montage_msks = []
        for image_name in image_names:
            image_path = os.path.join(image_root, image_name)
            msk_path = os.path.join(mask_root, image_name)
            image = cv2.imread(image_path, 0)
            mask = cv2.imread(msk_path, 0)
            image_512 = cv2.resize(image, (512,512))
            mask_512 = cv2.resize(mask, (512,512))

            montage_imgs.append(image_512)
            montage_msks.append(mask_512)

        return montage_imgs, montage_msks




    def compile(self,addition,classes,dilate,dilate_rate,loss=bce_dice_loss):
        K.set_image_dim_ordering('tf')
        x = inputs = Input(shape=(512,512), dtype='float32')
        x = Reshape((512,512) + (1,))(x)

        down1 = Conv2D(44, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        b1 = BatchNormalization()(down1)
        b1 = Dropout(rate=0.3)(b1)
        down1 = Conv2D(44, 3, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b1)
        b2 = BatchNormalization()(down1)
        b2 = Dropout(rate=0.3)(b2)
        down1pool = MaxPooling2D((2, 2), strides=(2, 2))(b2)
        down1pool = Dropout(rate=0.3)(down1pool)
        down2 = Conv2D(88, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down1pool)
        b3 = BatchNormalization()(down2)
        b3 = Dropout(rate=0.3)(b3)
        down2 = Conv2D(88, 3, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b3)
        b4 = BatchNormalization()(down2)
        b4 = Dropout(rate=0.3)(b4)
        down2pool = MaxPooling2D((2,2), strides=(2, 2))(b4)
        down2pool = Dropout(rate=0.3)(down2pool)
        down3 = Conv2D(176, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down2pool)
        b5 = BatchNormalization()(down3)
        b5 = Dropout(rate=0.3)(b5)
        down3 = Conv2D(176, 3, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b5)
        b6 = BatchNormalization()(down3)
        b6 = Dropout(rate=0.3)(b6)
        down3pool = MaxPooling2D((2, 2), strides=(2, 2))(b6)
        down3pool = Dropout(rate=0.3)(down3pool)

        if dilate == 1:
        # stacked dilated convolution at the bottleneck
            dilate1 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=1, kernel_initializer='he_normal')(down3pool)
            b7 = BatchNormalization()(dilate1)
            b7 = Dropout(rate=0.3)(b7)
            dilate2 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=2, kernel_initializer='he_normal')(b7)
            b8 = BatchNormalization()(dilate2)
            b8 = Dropout(rate=0.3)(b8)
            dilate3 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=4, kernel_initializer='he_normal')(b8)
            b9 = BatchNormalization()(dilate3)
            b9 = Dropout(rate=0.3)(b9)
            dilate4 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=8, kernel_initializer='he_normal')(b9)
            b10 = BatchNormalization()(dilate4)
            b10 = Dropout(rate=0.3)(b10)
            dilate5 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=16, kernel_initializer='he_normal')(b10)
            b11 = BatchNormalization()(dilate5)
            b11 = Dropout(rate=0.3)(b11)
            dilate6 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=32, kernel_initializer='he_normal')(b11)
            if addition == 1:
                dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
                up3 = UpSampling2D((2, 2))(dilate_all_added)
            else:
                up3 = UpSampling2D((2, 2))(dilate6)
        else:
            dilate1 = Conv2D(176,3, activation='relu', padding='same', kernel_initializer='he_normal')(down3pool)
            b7 = BatchNormalization()(dilate1)
            b7 = Dropout(rate=0.3)(b7)
            dilate2 = Conv2D(176,3, activation='relu', padding='same', kernel_initializer='he_normal')(b7)
            b8 = BatchNormalization()(dilate2)
            b8 = Dropout(rate=0.3)(b8)
            dilate3 = Conv2D(176,3, activation='relu', padding='same', kernel_initializer='he_normal')(b8)
            b9 = BatchNormalization()(dilate3)
            b9 = Dropout(rate=0.3)(b9)
            dilate4 = Conv2D(176,3, activation='relu', padding='same', kernel_initializer='he_normal')(b9)
            b10 = BatchNormalization()(dilate4)
            b10 = Dropout(rate=0.3)(b10)
            dilate5 = Conv2D(176,3, activation='relu', padding='same', kernel_initializer='he_normal')(b10)
            b11 = BatchNormalization()(dilate5)
            b11 = Dropout(rate=0.3)(b11)
            dilate6 = Conv2D(176,3, activation='relu', padding='same', kernel_initializer='he_normal')(b11)
            if addition ==1:
                dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
                up3 = UpSampling2D((2, 2))(dilate_all_added)
            else:
                up3 = UpSampling2D((2, 2))(dilate6)
        up3 = Conv2D(88,3, activation='relu', padding='same', kernel_initializer='he_normal')(up3)
        up3 = concatenate([down3, up3])
        b12 = BatchNormalization()(up3)
        b12 = Dropout(rate=0.3)(b12)
        up3 = Conv2D(88,3, activation='relu', padding='same', kernel_initializer='he_normal')(b12)
        b13 = BatchNormalization()(up3)
        up3 = Conv2D(88,3, activation='relu', padding='same', kernel_initializer='he_normal')(b13)

        up2 = UpSampling2D((2, 2))(up3)
        up2 = Conv2D(44,3, activation='relu', padding='same', kernel_initializer='he_normal')(up2)
        up2 = concatenate([down2, up2])
        b14 = BatchNormalization()(up2)
        b14 = Dropout(rate=0.3)(b14)
        up2 = Conv2D(44,3, activation='relu', padding='same', kernel_initializer='he_normal')(b14)
        b15 = BatchNormalization()(up2)
        up2 = Conv2D(44,3, activation='relu', padding='same', kernel_initializer='he_normal')(b15)

        up1 = UpSampling2D((2, 2))(up2)
        up1 = Conv2D(22,3, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
        up1 = concatenate([down1, up1])
        b16 = BatchNormalization()(up1)
        b16 = Dropout(rate=0.3)(b16)
        up1 = Conv2D(22,3, activation='relu', padding='same', kernel_initializer='he_normal')(b16)
        b17 = BatchNormalization()(up1)
        up1 = Conv2D(22,3, activation='relu', padding='same', kernel_initializer='he_normal')(b17)
        b18 = BatchNormalization()(up1)
        x = Conv2D(classes, 1, activation='softmax')(b18)
        x = Lambda(lambda x: x[:, :, :, 1], output_shape=(512,512))(x)
        self.net = Model(inputs=inputs, outputs=x)
        self.net.compile(optimizer=RMSprop(), loss='mse', metrics=[r_square])
        self.net.summary()
        return


    def train(self,lr,max_lr,weight_path):
        gen_trn = self.batch_generator(imgs=X_train, msks=Y_train, batch_size=3)
        gen_val = self.batch_generator(imgs=X_val, msks=Y_val, batch_size=3)
        clr_triangular = CyclicLR(mode='triangular')
        clr_triangular._reset(new_base_lr=lr, new_max_lr=max_lr)
        cb = [clr_triangular,EarlyStopping(monitor='val_loss', min_delta=1e-3,
        patience=3000, verbose=1, mode='min'),
            ModelCheckpoint(weight_path,
            monitor='val_loss', save_best_only=True, verbose=1),
            TensorBoard(log_dir='./logs', batch_size=3,
            write_graph=True, write_images=True)]

        self.net.fit_generator(generator=gen_trn, steps_per_epoch=10, epochs=epochs,
                               validation_data=gen_val, validation_steps=10, verbose=1, callbacks=cb)

        return

    def batch_generator(self, imgs, msks, batch_size,transform=True):

        img_np = np.array(imgs)

        mean, std_dev = np.mean(img_np), np.std(img_np)
        normalize = lambda x: (x - mean) / (std_dev + 1e-10)

        while True:

            # img_batch = np.zeros((batch_size,) + (512,512), dtype=imgs.dtype)
            # msk_batch = np.zeros((batch_size,) + (512,512), dtype=msks.dtype)
            img_batch = []
            msk_batch = []

            for batch_idx in range(batch_size):
                # Sample a random window.
                # y0, x0 = np.random.randint(0, H - wdw_H), np.random.randint(0, W - wdw_W)
                # y1, x1 = y0 + wdw_H, x0 + wdw_W
                #
                # img_batch[batch_idx] = imgs[y0:y1, x0:x1]
                # msk_batch[batch_idx] = msks[y0:y1, x0:x1]
                a = np.random.randint(0, len(msks))
                img_batch.append(imgs[a])
                gray_img = msks[a].astype(np.float32)
                for i in range(512):
                    for j in range(512):
                        gray_num = gray_img[i][j]
                        if gray_num > 200:
                            gray_img[i][j] = float((gray_num - 255 +20)/50)
                        else:
                            gray_img[i][j] = float((gray_num + 20)/50)



                msk_batch.append(gray_img)


            img_batch = normalize(img_batch)
            msk_batch = np.array(msk_batch)
            yield img_batch, msk_batch


if __name__ == "__main__":
    arg_list = argparse.ArgumentParser()
    arg_list.add_argument('--epochs', help='Number of epochs to run', default='30000', type=int)
    arg_list.add_argument('--classes', help='Number of output classes', default='2', type=int)
    arg_list.add_argument('--lr', help='Learning rate', default='0.00001', type=float)
    arg_list.add_argument('--max_lr', help='Learning rate', default='0.0005', type=float)
    arg_list.add_argument('--image_path', help='Path to stack of training images', default='/home/huluwa/trainset/image_warp', type=str)
    arg_list.add_argument('--mask_path', help='Path to stack of ground truth annotations', default='/home/huluwa/trainset/image_offset', type=str)
    arg_list.add_argument('--dilate', help='Add dilated convolutions', default='1', type=int)
    arg_list.add_argument('--weight_path', help='path to save weights', default='weights/normal_unet_cv4.weights', type=str)
    arg_list.add_argument('--dilate_rate', help='rate of dilation in downsampling convs', default='1', type=int)
    arg_list.add_argument('--addition', help='add the central layers together as a concat operation', default='1', type=int)
    arg_list.add_argument('--gpu', help='specify the GPU to use', default='0', type=str)

    args = vars(arg_list.parse_args())

    epochs = args['epochs']
    lr = args['lr']
    max_lr = args['max_lr']
    classes = args['classes']
    image_root = args['image_path']
    mask_root = args['mask_path']
    dilate = args['dilate']
    dilate_rate = args['dilate_rate']
    weight_path = args['weight_path']
    addition = args['addition']
    gpu = args['gpu']
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    model_init = UNet()
    (X_train, Y_train) = model_init.load_data(image_root=image_root,mask_root=mask_root)

    (X_val, Y_val) = model_init.load_data(image_root=image_root,mask_root=mask_root)
    model_init.compile(addition=addition,classes=2,dilate=dilate,dilate_rate=dilate_rate)
    model_init.train(lr=lr,max_lr=max_lr,weight_path=weight_path)
