from keras import backend as K
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from show import myshow2d


mask=sitk.ReadImage('/home/pirl/Downloads/cardiac/data/train/CHD/label/sample2_dia_M.mha')
np_mask=sitk.GetArrayFromImage(mask)
print(np_mask.shape)


def simple_preprocess_mask(mask):
    print('before preprocessd mask shape :', mask.shape)
    value, count = np.unique(mask[:, :, :], return_counts=True)
    print('image value : ' + str(value) + ' image Count : ' + str(count))
    # print('preprocess mask num labels :',self.num_labels)
    mask_dummy = np.zeros(mask.shape + (3,))
    mask_dummy[:,:,:,0][mask == 1] =1.
    mask_dummy[:,:,:,1][mask == 2] =2.
    mask_dummy[:,:,:,2][mask == 3] =3.
    print('processed mask shape :', mask_dummy.shape)
    return mask_dummy

# np_mask = np.sum(np_mask, axis=-1)
# print(np_mask.shape)
# np_mask = np.expand_dims(np_mask, axis=-1)

np_mask=simple_preprocess_mask(np_mask)
np_mask = np.sum(np_mask, axis=-1)
np_mask = np.expand_dims(np_mask, axis=-1)
print(np_mask.shape)
value, count = np.unique(np_mask[:,:,:,:], return_counts=True)
print('image value : '+str(value)+' image Count : '+ str(count))

result = sitk.GetImageFromArray(np_mask)
myshow2d(result[:,:,50])