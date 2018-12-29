# -*- coding: utf-8 -*-

import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
import math
import random
from tensorpack.dataflow.base import RNGDataFlow
from scipy import ndimage
from data_augment import ImageDataGenerator


class MyDataFlow(RNGDataFlow):
    def __init__(self, data_dir,image_file_name,label_file_name, shuffle=True):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.image_file_name = image_file_name
        self.label_file_name = label_file_name

    def get_data(self):
        image_paths =[]
        label_paths =[]
        for file_name in sorted(os.listdir(os.path.join(self.data_dir,'image'))):
            if self.image_file_name in file_name:
                image_paths.append(os.path.join(self.data_dir, 'image', file_name))

        for file_name in sorted(os.listdir(os.path.join(self.data_dir,'label'))):
            if self.label_file_name in file_name:
                label_paths.append(os.path.join(self.data_dir, 'label', file_name))

        if len(image_paths)!=len(label_paths):
            raise NotImplementedError('image dataset length is not match label dataset length')
        print('image length :',len(image_paths))
        print('label length :',len(label_paths))
        rand_list=list()
        data_len = len(image_paths)
        for index in range(data_len):
            rand_list.append(index)
        if self.shuffle==True:
            random.shuffle(rand_list)
            for i in rand_list:
                image = sitk.ReadImage(image_paths[i])
                label = sitk.ReadImage(label_paths[i])
                castImageFilter = sitk.CastImageFilter()
                castImageFilter.SetOutputPixelType(sitk.sitkInt16)
                image = castImageFilter.Execute(image)

                #label spacing setting
                label.SetSpacing(image.GetSpacing())
                castImageFilter.SetOutputPixelType(sitk.sitkInt8)
                label = castImageFilter.Execute(label)

                #HU setting
                statisticsFilter = sitk.StatisticsImageFilter()
                statisticsFilter.Execute(image)

                intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
                intensityWindowingFilter.SetOutputMaximum(255)
                intensityWindowingFilter.SetOutputMinimum(0)
                intensityWindowingFilter.SetWindowMaximum(
                statisticsFilter.GetMean() + statisticsFilter.GetSigma());
                intensityWindowingFilter.SetWindowMinimum(
                statisticsFilter.GetMean() - statisticsFilter.GetSigma());

                image = intensityWindowingFilter.Execute(image)

                #transpose axis
                image_np = sitk.GetArrayFromImage(image).astype('float32')
                resize_factor = image.GetSpacing()[::-1]
                image_np = ndimage.zoom(image_np, resize_factor, order=0, mode='constant', cval=0.0)

                label_np = sitk.GetArrayFromImage(label).astype('float32')
                resize_factor = label.GetSpacing()[::-1]
                label_np = ndimage.zoom(label_np, resize_factor, order=0, mode='constant', cval=0.0)

                yield [image_np, label_np]


def gen_data(df):
    while True:
        for data in df.get_data():
            yield data


class Preprocess(object):
    def __init__(self):
        self.num_channels = 1

        self.datagen = ImageDataGenerator(
            rotation_range=[5., 5., 5.],
            zoom_range=[0.9, 1.1],
            width_shift_range=0.1,
            height_shift_range=0.1,
            depth_shift_range=0.1,
            horizontal_flip=False,
            fill_mode='constant', cval=0.)

    def sample_z_norm(self, img):
        for self.num_channel in range(self.num_channels):
            img[..., self.num_channel] -= np.mean(img[..., self.num_channel])
            img[..., self.num_channel] /= np.std(img[..., self.num_channel])

        # print('sample z norm img shape :',img.shape)
        return img

    def simple_preprocess_img(self, img):
        img = np.expand_dims(img, -1)
        # print('preprocessed img shape :',img.shape)
        return img

    def simple_sample_z_norm(self, img):
        for self.num_channel in range(self.num_channels):
            img[..., self.num_channel] -= np.mean(img[..., self.num_channel])
            img[..., self.num_channel] /= np.std(img[..., self.num_channel])
        return img

    def simple_preprocess_mask(self, mask):
        #print('before preprocessd mask shape :',mask.shape)
        # value, count = np.unique(mask, return_counts=True)
        # print('image value : ' + str(value) + ' image Count : ' + str(count))
        mask_dummy = np.zeros(mask.shape + (3,))
        mask_dummy[:, :, :, 0][mask == 1] = 1.
        mask_dummy[:, :, :, 1][mask == 2] = 1.
        mask_dummy[:, :, :, 2][mask == 3] = 1.
        #print('processed mask shape :',mask_dummy.shape)
        # value, count = np.unique(mask_dummy[:, :, :, :], return_counts=True)
        # print('image value : ' + str(value) + ' image Count : ' + str(count))
        return mask_dummy

    def whole_merge_mask(self, mask):
        mask = np.sum(mask, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        # print('merged mask shape :',mask.shape)
        return mask

    def resize_whole(self, img_mask):
        image = img_mask[0].copy()
        label = img_mask[1].copy()

        max_size = 180
        check_size = (image.shape[0] / max_size) * (image.shape[1] / max_size) * (image.shape[2] / max_size)
        if check_size > 1.:
            reduction_factor = ((max_size * max_size * max_size) / (
                        image.shape[0] * image.shape[1] * image.shape[2])) ** (1 / 3)
            resize_factor = (reduction_factor, reduction_factor, reduction_factor)
            image = ndimage.zoom(image, resize_factor + (1.,), order=0, mode='constant', cval=0.0)
            label = ndimage.zoom(label, resize_factor + (1.,), order=0, mode='constant', cval=0.0)
        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)
        return [image, label]

    def crop_to_patch(self, img_mask):
        mask_dummy = img_mask[1].copy()
        mask_dummy = np.sum(mask_dummy, axis=-1)
        coordinates = np.argwhere(mask_dummy != 0)
        start_end = [[np.min(coordinates[:, 0]), np.max(coordinates[:, 0])],
                     [np.min(coordinates[:, 1]), np.max(coordinates[:, 1])],
                     [np.min(coordinates[:, 2]), np.max(coordinates[:, 2])]]

        x_start = start_end[0][0]
        x_end = start_end[0][1]
        y_start = start_end[1][0]
        y_end = start_end[1][1]
        z_start = start_end[2][0]
        z_end = start_end[2][1]

        # define margin for minimum size
        min_size = 24
        if x_end - x_start < min_size:
            x_margin = math.ceil((min_size - (x_end - x_start)) / 2)
            if x_start - x_margin < 0:
                x_start = 0
            else:
                x_start -= x_margin
            if x_end + x_margin > mask_dummy.shape[0]:
                x_end = mask_dummy.shape[0]
            else:
                x_end += x_margin
        if y_end - y_start < min_size:
            y_margin = math.ceil((min_size - (y_end - y_start)) / 2)
            if y_start - y_margin < 0:
                y_start = 0
            else:
                y_start -= y_margin
            if y_end + y_margin > mask_dummy.shape[1]:
                y_end = mask_dummy.shape[1]
            else:
                y_end += y_margin
        if z_end - z_start < min_size:
            z_margin = math.ceil((min_size - (z_end - z_start)) / 2)
            if z_start - z_margin < 0:
                z_start = 0
            else:
                z_start -= z_margin
            if z_end + z_margin > mask_dummy.shape[2]:
                z_end = mask_dummy.shape[2]
            else:
                z_end += z_margin

        # define margin 20% of object size
        x_margin = math.ceil((x_end - x_start) * 0.2)
        y_margin = math.ceil((y_end - y_start) * 0.2)
        z_margin = math.ceil((z_end - z_start) * 0.2)
        if x_start - x_margin < 0:
            x_start = 0
        else:
            x_start -= x_margin
        if x_end + x_margin > mask_dummy.shape[0]:
            x_end = mask_dummy.shape[0]
        else:
            x_end += x_margin

        if y_start - y_margin < 0:
            y_start = 0
        else:
            y_start -= y_margin
        if y_end + y_margin > mask_dummy.shape[1]:
            y_end = mask_dummy.shape[1]
        else:
            y_end += y_margin

        if z_start - z_margin < 0:
            z_start = 0
        else:
            z_start -= z_margin
        if z_end + z_margin > mask_dummy.shape[2]:
            z_end = mask_dummy.shape[2]
        else:
            z_end += z_margin
        # print("x~z start end point :",x_start,x_end,y_start,y_end,z_start,z_end)
        # print("crop image shape :",img_mask[0].shape)
        # print("crop mask shape :",img_mask[1].shape)
        img_mask[0] = img_mask[0][x_start:x_end, y_start:y_end, z_start:z_end, :]
        img_mask[1] = img_mask[1][x_start:x_end, y_start:y_end, z_start:z_end, :]
        return img_mask

    def resize_patch(self, img_mask):
        crop_image = img_mask[0].copy()
        crop_label = img_mask[1].copy()

        resize_factor = ()
        for num_axis in range(3):
            if crop_image.shape[num_axis] < 24:
                resize_factor += (24. / crop_image.shape[num_axis],)
            else:
                resize_factor += (1.,)
        if resize_factor != (1., 1., 1.):
            crop_image = ndimage.zoom(crop_image, resize_factor + (1.,), order=0, mode='constant', cval=0.0)
            crop_label = ndimage.zoom(crop_label, resize_factor + (1.,), order=0, mode='constant', cval=0.0)

        max_size = 180
        check_size = (crop_image.shape[0] / max_size) * (crop_image.shape[1] / max_size) * (
                    crop_image.shape[2] / max_size)
        if check_size > 1.:
            reduction_factor = ((max_size * max_size * max_size) / (
                        crop_image.shape[0] * crop_image.shape[1] * crop_image.shape[2])) ** (1 / 3)
            resize_factor = (reduction_factor, reduction_factor, reduction_factor)
            crop_image = ndimage.zoom(crop_image, resize_factor + (1.,), order=0, mode='constant', cval=0.0)
            crop_label = ndimage.zoom(crop_label, resize_factor + (1.,), order=0, mode='constant', cval=0.0)
        crop_image = np.expand_dims(crop_image, 0)
        crop_label = np.expand_dims(crop_label, 0)
        return [crop_image, crop_label]

    def data_aug(self, img_mask):
        seed = np.random.randint(0, 100, 1)[0]
        for img in self.datagen.flow(img_mask[0], batch_size=1, seed=seed):
            break
        for mask in self.datagen.flow(img_mask[1], batch_size=1, seed=seed):
            break
        print('-------------------------------------------------')
        print('input image shape :',img.shape)
        print('input label shape :',mask.shape)
        print('-------------------------------------------------')
        return [img, mask]