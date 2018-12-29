import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np

def myshow2d(img, title = None, margin=0.05, dpi=80):
   nda = sitk.GetArrayFromImage(img)
   spacing = img.GetSpacing()

   ysize = nda.shape[0]
   xsize = nda.shape[1]

   figsize = (1+margin) * ysize /dpi, (1+margin) * xsize /dpi

   fig = plt.figure(title, figsize = figsize , dpi = dpi)
   ax = fig.add_axes([margin,margin,1 - 2*margin, 1-2*margin])

   extent = (0, xsize*spacing[1], 0, ysize*spacing[0])

   t = ax.imshow(nda, extent=extent, cmap= 'gray', origin='lower')
   if(title):
       plt.title(title)
   plt.show()

def resample(image, new_spacing):
   # Output image Origin, Spacing, Size, Direction 들은 레퍼런스에서 가져옵니다.
   transform = sitk.Transform(3, sitk.sitkIdentity)
   interpolator = sitk.sitkNearestNeighbor
   resize_factor= np.array(image.GetSpacing())/np.array(new_spacing)
   new_real_shape = np.array(image.GetSize())*resize_factor
   new_shape = np.array(np.round(new_real_shape),np.uint32)
   new_size = (int(new_shape[0]),int(new_shape[1]),int(new_shape[2]))
   origin = image.GetOrigin()
   new_spacing = new_spacing
   direction = image.GetDirection()
   #return sitk.Resample(image, reference_image, transform, interpolator, default_value)
   return sitk.Resample(image,new_size , transform, interpolator, origin, new_spacing, direction)