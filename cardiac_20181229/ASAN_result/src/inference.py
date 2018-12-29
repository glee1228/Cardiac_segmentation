import argparse
import glob
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import SimpleITK as sitk
from scipy import ndimage
import tensorflow as tf
from model import average_dice_coef, average_dice_coef_loss
import numpy as np
from matplotlib import pyplot as plt
from show import myshow2d, resample

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type = str , default = "/home/pirl/Downloads/cardiac/data/test/CHD")
parser.add_argument("--test_filename",type =str, default = "_dia.mha")
parser.add_argument("--output_path", type = str , default = "/home/pirl/Downloads/cardiac/data/output/CHD")
parser.add_argument("--model_name", type = str , default = "/home/pirl/Downloads/cardiac/data/model/CHD_dia/checkpoint_chd_dia.h5")


FLAGS, unparsed = parser.parse_known_args()

model_name = FLAGS.model_name
test_path = FLAGS.test_path
test_filename = FLAGS.test_filename

test_list = []
test_filename_list =[]
for filename in os.listdir(test_path):
    if test_filename in filename :
        test_filename_list.append(filename)
        test_list.append(filename)

for i in range(len(test_list)):

    #read test image
    test_open_path = os.path.join(test_path,test_list[i])
    test=sitk.ReadImage(test_open_path)
    #set PixelType Int16
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkInt16)
    test_pix = castImageFilter.Execute(test)

    #HU setting
    statisticsFilter = sitk.StatisticsImageFilter()
    statisticsFilter.Execute(test_pix)

    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    intensityWindowingFilter.SetWindowMaximum(
    statisticsFilter.GetMean() + 2*statisticsFilter.GetSigma());
    intensityWindowingFilter.SetWindowMinimum(
    statisticsFilter.GetMean() - statisticsFilter.GetSigma());

    test_hu = intensityWindowingFilter.Execute(test_pix)

    #transpose axis
    test_np = sitk.GetArrayFromImage(test_hu).astype('float32')
    resize_factor = test_hu.GetSpacing()[::-1]
    test_np = ndimage.zoom(test_np, resize_factor, order=0, mode='constant', cval=0.0)

    #dimension expand : equal to train dataset
    test_np = np.expand_dims(test_np, -1)
    test_np = np.expand_dims(test_np, 0)
    model = load_model(model_name, custom_objects={
        'tf':tf,
        'average_dice_coef_loss': average_dice_coef_loss,
        'average_dice_coef' : average_dice_coef
    })
    # prediction
    predictions = model.predict(test_np, verbose=1)

    #print('predict :'+str(predictions.shape))
    predictions = np.squeeze(predictions,axis=0)
    #print('predict image shape :',predictions.shape)
    mask_dummy = np.zeros((predictions.shape[0],predictions.shape[1],predictions.shape[2]))
    mask_dummy[:, :, :][predictions[:,:,:,0] >= 0.7] = 1.
    mask_dummy[:, :, :][predictions[:,:,:,1] >= 0.7] = 2.
    mask_dummy[:, :, :][predictions[:,:,:,2] >= 0.7] = 3.
    #print('predict image GetValue :',predictions[:,:,])


    mask=sitk.GetImageFromArray(mask_dummy)
    origin_size=mask.GetSize()
    origin_spacing = mask.GetSpacing()
    new_size = test.GetSize()
    resize_factor = (float(1./(new_size[0]/origin_size[0])), float(1./(new_size[1]/origin_size[1])), float(1./(new_size[2]/origin_size[2])))
    #print(resize_factor)
    result=resample(mask,resize_factor)
    result.SetSpacing(test.GetSpacing())
    result.SetOrigin(test.GetOrigin())
    print('test_input size : ',test.GetSize())
    print('test_output size :',result.GetSize())

    # print(resize_factor)
    # mask_dummy = ndimage.zoom(mask_dummy, resize_factor - (1.,), order=0, mode='nearest', cval=0.0)
    # print('mask dummy shape :',mask_dummy.shape)
    # predictions = sitk.GetImageFromArray(mask_dummy)
    sitk.WriteImage(result, os.path.join(FLAGS.output_path,test_filename_list[i][:-4]+'_output.mha'), True)
    # print(file step)
    print('step {}/{} complete'.format(i + 1, len(test_list)))

