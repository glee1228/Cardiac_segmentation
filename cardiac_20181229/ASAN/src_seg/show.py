import SimpleITK as sitk
from matplotlib import pyplot as plt


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