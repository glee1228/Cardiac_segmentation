import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
import math
import random

class NiftiDataset(object):
  """
  load image-label pair for training, testing and inference.
  Currently only support linear interpolation method
  Args:
		data_dir (string): Path to data directory.
    image_filename (string): Filename of image data.
    label_filename (string): Filename of label data.
    transforms (list): List of SimpleITK image transformations.
    train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.
  """

  def __init__(self,
    data_dir = '',
    image_filename = '',
    label_filename = '',
    transforms=None,
    train=False):

    # Init membership variables
    self.data_dir = data_dir
    self.image_filename = image_filename
    self.label_filename = label_filename
    self.transforms = transforms
    self.train = train

  def get_dataset(self):
    image_paths = []
    label_paths = []
    if 'image' in os.listdir(self.data_dir):
      for i in sorted(os.listdir(self.data_dir+'/image')):
        if self.image_filename in i:
          image_paths.append(os.path.join(self.data_dir,'image',i))
    if 'label' in os.listdir(self.data_dir):
      for i in sorted(os.listdir(self.data_dir+'/label')):
        if self.label_filename in i:
          label_paths.append(os.path.join(self.data_dir,'label',i))

    print(image_paths)
    print('--------------------------------------------')
    print(label_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths,label_paths))
    dataset = dataset.map(lambda image_path, label_path: tuple(tf.py_func(
      self.input_parser, [image_path, label_path], [tf.float32,tf.int32])))

    self.dataset = dataset
    self.data_size = len(image_paths)
    return self.dataset

  def read_image(self,path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    return reader.Execute()

  def input_parser(self,image_path, label_path):
    # read image and label
    image = self.read_image(image_path.decode("utf-8"))
    image.SetSpacing((0.28,0.28,0.4))
     # cast image and label
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkInt16)
    image = castImageFilter.Execute(image)
    #print(image)
    if self.train:
      label = self.read_image(label_path.decode("utf-8"))
      label.SetSpacing(image.GetSpacing())
      castImageFilter.SetOutputPixelType(sitk.sitkInt8)
      label = castImageFilter.Execute(label)
    else:
      label = sitk.Image(image.GetSize(),sitk.sitkInt8)
      label.SetOrigin(image.GetOrigin())
      label.SetSpacing(image.GetSpacing())
    #print(label)
    sample = {'image':image, 'label':label}

    if self.transforms:
      for transform in self.transforms:
        sample = transform(sample)
    print('no transform')
    # convert sample to tf tensors
    image_np = sitk.GetArrayFromImage(sample['image'])
    label_np = sitk.GetArrayFromImage(sample['label'])

    image_np = np.asarray(image_np,np.float32)
    label_np = np.asarray(label_np,np.int32)

    # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
    image_np = np.transpose(image_np,(2,1,0))
    label_np = np.transpose(label_np,(2,1,0))
    print(image_np.shape,label_np.shape)
    return image_np, label_np

class Normalization(object):
  """
  Normalize an image to 0 - 255
  """

  def __init__(self):
    self.name = 'Normalization'

  def __call__(self, sample):
    # normalizeFilter = sitk.NormalizeImageFilter()
    # image, label = sample['image'], sample['label']
    # image = normalizeFilter.Execute(image)
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image, label = sample['image'], sample['label']
    image = resacleFilter.Execute(image)

    return {'image': image, 'label': label}

class StatisticalNormalization(object):
  """
  Normalize an image by mapping intensity with intensity distribution
  """

  def __init__(self, sigma, max_sd, min_sd):
    self.name = 'StatisticalNormalization'
    assert isinstance(sigma, float)
    self.sigma = sigma
    self.max_sd = max_sd
    self.min_sd = min_sd

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    statisticsFilter = sitk.StatisticsImageFilter()
    statisticsFilter.Execute(image)

    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    npImage = sitk.GetArrayFromImage(image)
    # value2, count2 = np.unique(npImage, return_counts=True)
    # print('image value : '+str(value2[720])+' image Count : '+ str(count2[720]))
    intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean()+self.max_sd*self.sigma*statisticsFilter.GetSigma());
    intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean()-self.min_sd*self.sigma*statisticsFilter.GetSigma());

    image = intensityWindowingFilter.Execute(image)

    return {'image': image, 'label': label}

class Resample(object):

  def __init__(self, voxel_size):
    self.name = 'Resample'

    assert isinstance(voxel_size, (float, tuple))
    if isinstance(voxel_size, float):
      self.voxel_size = (voxel_size, voxel_size, voxel_size)
    else:
      assert len(voxel_size) == 3
      self.voxel_size = voxel_size

  def __call__(self, sample):
     image, label = sample['image'], sample['label']
     #statFilter = sitk.StatisticsImageFilter()
     #statFilter.Execute(label)
     #print('Before Resample Label GetMaximum :' + str(statFilter.GetMaximum()))
     npLabel = sitk.GetArrayFromImage(label)
     value, count = np.unique(npLabel, return_counts=True)
     #print('Origin Label value : ' + str(value) + '  Origin Label Count : ' + str(count))

     old_spacing = image.GetSpacing()
     old_size = image.GetSize()
     #print(str(old_spacing))
     new_spacing = (0.5,0.5,0.5)
     new_size = []

     for i in range(3):
        new_size.append(int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])))
     new_size = tuple(new_size)

     resampler = sitk.ResampleImageFilter()
     resampler.SetInterpolator(sitk.sitkBSpline)
     resampler.SetOutputSpacing(new_spacing)
     resampler.SetSize(new_size)

    # resample on image
     resampler.SetOutputOrigin(image.GetOrigin())
     resampler.SetOutputDirection(image.GetDirection())

     print("Resampling image...")
     image = resampler.Execute(image)

     # resample on segmentation
     resampler.SetInterpolator(sitk.sitkNearestNeighbor)
     resampler.SetOutputOrigin(label.GetOrigin())
     resampler.SetOutputDirection(label.GetDirection())
     #print(resampler)
     print("Resampling segmentation...")
     label = resampler.Execute(label)

     #print('ReSampled Image Size :'+str(image.GetSize())+' ReSampled Label Size :'+str(label.GetSize()))
     # statFilter = sitk.StatisticsImageFilter()
     # statFilter.Execute(label)
     npLabel = sitk.GetArrayFromImage(label)
     npImage = sitk.GetArrayFromImage(image)
     value,count = np.unique(npLabel,return_counts=True)
     value2,count2 = np.unique(npImage,return_counts=True)
     #print('Resampled image value : '+str(value2)+'  Resampled image Count : '+ str(count2))
     #print('Resampled Label value : '+str(value)+'  Resampled Label Count : '+ str(count))
     #print('After Resample Label GetMaximum :' + str(statFilter.GetMaximum()))
     return {'image': image, 'label': label}

class Padding(object):

  def __init__(self, output_size):
    self.name = 'Padding'

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

    assert all(i > 0 for i in list(self.output_size))

  def __call__(self,sample):
    image, label = sample['image'], sample['label']
    size_old = image.GetSize()



    if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (size_old[2] >= self.output_size[2]):
      return sample
    else:
      self.output_size = list(self.output_size)
      if size_old[0] > self.output_size[0]:
        self.output_size[0] = size_old[0]
      if size_old[1] > self.output_size[1]:
        self.output_size[1] = size_old[1]
      if size_old[2] > self.output_size[2]:
        self.output_size[2] = size_old[2]
 
      self.output_size = tuple(self.output_size)

      resampler = sitk.ResampleImageFilter()
      resampler.SetOutputSpacing(image.GetSpacing())
      resampler.SetSize(self.output_size)

      # resample on image
      resampler.SetInterpolator(sitk.sitkBSpline)
      resampler.SetOutputOrigin(image.GetOrigin())
      resampler.SetOutputDirection(image.GetDirection())
      image = resampler.Execute(image)

      # resample on label
      resampler.SetInterpolator(sitk.sitkNearestNeighbor)
      resampler.SetOutputOrigin(label.GetOrigin())
      resampler.SetOutputDirection(label.GetDirection())

      label = resampler.Execute(label)

      return {'image': image, 'label': label}

class RandomCrop(object):

  def __init__(self, output_size, drop_ratio=0.5, min_pixel=1, min_ratio = 0.7):
    self.name = 'Random Crop'

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

    assert isinstance(drop_ratio, float)
    if drop_ratio >=0 and drop_ratio<=1:
      self.drop_ratio = drop_ratio
    else:
      raise RuntimeError('Drop ratio should be between 0 and 1')

    assert isinstance(min_pixel, int)
    if min_pixel >=0 :
      self.min_pixel = min_pixel
    else:
      raise RuntimeError('Min label pixel count should be integer larger than 0')
    assert isinstance(min_ratio,float)
    if min_ratio >=0 and min_ratio<=1:
        self.min_ratio = min_ratio
    else :
        raise RuntimeError('Min Ratio shold be between 0 and 1')

  def __call__(self,sample):
    image, label = sample['image'], sample['label']
    size_old = image.GetSize()

    print('size_ old in random Crop : '+str(image.GetSize()))
    size_new = self.output_size
    pixel_count = size_new[0]*size_new[1]*size_new[2]
    print('size_ new in random Crop : '+str(size_new))
    contain_label = False

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

    statFilter = sitk.StatisticsImageFilter()
    statFilter.Execute(label)
    #print('Before Random Crop GetSum :'+str(statFilter.GetSum())+' Before Random Crop GetSum :'+str(statFilter.GetMaximum()))

    while not contain_label: 
      # get the start crop coordinate in ijk
      if size_old[0] <= size_new[0]:
        start_i = 0
      else:
        start_i = np.random.randint(0, size_old[0]-size_new[0])

      if size_old[1] <= size_new[1]:
        start_j = 0
      else:
        start_j = np.random.randint(0, size_old[1]-size_new[1])

      if size_old[2] <= size_new[2]:
        start_k = 0
      else:
        start_k = np.random.randint(0, size_old[2]-size_new[2])

      roiFilter.SetIndex([start_i,start_j,start_k])
      label_crop = roiFilter.Execute(label)
      print('label_crop_Size :'+str(label_crop.GetSize()))
      statFilter = sitk.StatisticsImageFilter()
      statFilter.Execute(label_crop)
      #print('After Random Crop GetSum :'+str(statFilter.GetSum())+' After Random Crop, Maximum :'+str(statFilter.GetMaximum()))
      # will iterate until a sub volume containing label is extracted
      pixel_count = label_crop.GetHeight()*label_crop.GetWidth()*label_crop.GetDepth()
      mean_pixel=statFilter.GetSum() / pixel_count
      #print('mean_pixel :'+str(mean_pixel))
      #print('min_ratio :'+str(self.min_ratio))
      if mean_pixel<self.min_ratio:
      #if statFilter.GetMaximum()<self.min_pixel:
        contain_label = self.drop(self.drop_ratio) # has some probabilty to contain patch with empty label
      else:
        contain_label = True
      #contain_label = True
    image_crop = roiFilter.Execute(image)

    return {'image': image_crop, 'label': label_crop}

  def drop(self,probability):
    return random.random() <= probability

class RandomNoise(object):
  """
  Randomly noise to the image in a sample. This is usually used for data augmentation.
  """
  def __init__(self):
    self.name = 'Random Noise'

  def __call__(self, sample):
    self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
    self.noiseFilter.SetMean(0)
    self.noiseFilter.SetStandardDeviation(0.1)

    print("Normalizing image...")
    image, label = sample['image'], sample['label']
    image = self.noiseFilter.Execute(image)

    return {'image': image, 'label': label}
