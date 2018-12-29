import argparse
import glob
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import SimpleITK as sitk
from scipy import ndimage
import tensorflow as tf
from keras_contrib.layers import InstanceNormalization
from model import average_dice_coef, average_dice_coef_loss
import numpy as np
from matplotlib import pyplot as plt
from show import myshow2d

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str, default='/home/pirl/Downloads/cardiac/data/model')
parser.add_argument("--epoch_number", type = int, default =5)
parser.add_argument("--test_path", type = str , default = "/home/pirl/Downloads/cardiac/data/test/CHD")
parser.add_argument("--test_filename",type =str, default = "_dia.mha")
parser.add_argument("--output_path", type = str , default = "/home/pirl/Downloads/cardiac/data/output/CHD")
parser.add_argument("--model_name", type = str , default = "/home/pirl/Downloads/cardiac/data/model/CHD_dia/checkpoint_roi_chd.h5")
parser.add_argument("--n_classes", type=int, default=4)

FLAGS, unparsed = parser.parse_known_args()

n_classes = FLAGS.n_classes
model_name = FLAGS.model_name
test_path = FLAGS.test_path
epoch_number = FLAGS.epoch_number
test_filename = FLAGS.test_filename

test_list = []
for filename in os.listdir(test_path):
    if test_filename in filename :
        test_list.append(filename)
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_gen = test_datagen.flow_from_directory(
#     test_path,
#     target_size = (512,512,171),
#     batch_size =1,
#     class_mode ='categorical')
test_open_path = os.path.join(test_path,test_list[0])
print(test_open_path)
test=sitk.ReadImage(test_open_path)
castImageFilter = sitk.CastImageFilter()
castImageFilter.SetOutputPixelType(sitk.sitkInt16)
test = castImageFilter.Execute(test)

#HU setting
statisticsFilter = sitk.StatisticsImageFilter()
statisticsFilter.Execute(test)

intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
intensityWindowingFilter.SetOutputMaximum(255)
intensityWindowingFilter.SetOutputMinimum(0)
intensityWindowingFilter.SetWindowMaximum(
statisticsFilter.GetMean() + 2*statisticsFilter.GetSigma());
intensityWindowingFilter.SetWindowMinimum(
statisticsFilter.GetMean() - statisticsFilter.GetSigma());

test = intensityWindowingFilter.Execute(test)

#transpose axis
test_np = sitk.GetArrayFromImage(test).astype('float32')
resize_factor = test.GetSpacing()[::-1]
test_np = ndimage.zoom(test_np, resize_factor, order=0, mode='constant', cval=0.0)

ttt = sitk.GetImageFromArray(test_np)
sitk.WriteImage(ttt, '/home/pirl/Downloads/cardiac/data/test_output/1.mha', True)
#dimension expand : equal to train dataset
test_np = np.expand_dims(test_np, -1)
test_np = np.expand_dims(test_np, 0)
model = load_model(model_name, custom_objects={
    'tf':tf,
    'average_dice_coef_loss': average_dice_coef_loss,
    'average_dice_coef' : average_dice_coef
})

predictions = model.predict(test_np, verbose=1)
#yhat = model.predict_classes(test_np)
print('predict :'+str(predictions.shape))
predictions = np.squeeze(predictions,axis=0)
print('predict image shape :',predictions.shape)
#print('predict image GetValue :',predictions[:,:,])
predictions = sitk.GetImageFromArray(predictions)
sitk.WriteImage(predictions, '/home/pirl/Downloads/cardiac/data/test_output/1_M.mha', True)











#print('predict sitk image shape :',predictions.GetSize())
#myshow2d(predictions[:,:,34])


# generator = datagen.flow_from_directory(
#         '',
#         target_size=(150, 150),
#         batch_size=16,
#         class_mode=None,  # only data, no labels
#         shuffle=False)  # keep data in same order as labels
#
# probabilities = model.predict_generator(generator, 2000)
#
# y_true = np.array([0] * 1000 + [1] * 1000)
# y_pred = probabilities > 0.5
#
# confusion_matrix(y_true, y_pred)