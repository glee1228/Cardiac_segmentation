from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import NiftiDataset
import os
import datetime
import SimpleITK as sitk
import math
import numpy as np
from tqdm import tqdm
import argparse
import sys

# select gpu devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # e.g. "0,1,2", "0,2" 

def prepare_batch(image,ijk_patch_indices):
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1],patch[2]:patch[3],patch[4]:patch[5]]
            image_batch.append(image_patch)
        image_batch = np.asarray(image_batch)
        image_batch = image_batch[:,:,:,:,np.newaxis]
        image_batches.append(image_batch)
        
    return image_batches

def evaluate():
    """evaluate the vnet model by stepwise moving along the 3D image"""
    # restore model grpah
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(FLAGS.model_path)

    # create transformations to image and labels
    transforms = [
        # NiftiDataset.Normalization(),
        NiftiDataset.StatisticalNormalization(FLAGS.sigma,FLAGS.max_sd_coe, FLAGS.min_sd_coe),
        NiftiDataset.Resample(0.25),
        NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer))      
        ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:  
        print("{}: Start evaluation...".format(datetime.datetime.now()))

        imported_meta.restore(sess, FLAGS.checkpoint_path)
        print("{}: Restore checkpoint success".format(datetime.datetime.now()))


        for file_name in os.listdir(FLAGS.data_dir):
            # ops to load data
            if FLAGS.image_filename in file_name:
                # check image data exists
                image_path = os.path.join(FLAGS.data_dir,file_name)
                image_split = file_name.split(".")
                output_filename = image_split[0]+'_output.mha'
                output_path = os.path.join(FLAGS.output_dir,output_filename)
                if not os.path.exists(image_path):
                    print("{}: Image file not found at {}".format(datetime.datetime.now(),image_path))
                    break
                else:
                    print("{}: Evaluating image at {}".format(datetime.datetime.now(),image_path))
                    print("{}: Output image at {}".format(datetime.datetime.now(),output_path))

                    # read image file
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(image_path)
                    image = reader.Execute()

                    # preprocess the image and label before inference
                    image_tfm = image

                    # create empty label in pair with transformed image
                    label_tfm = sitk.Image(image_tfm.GetSize(),sitk.sitkUInt32)
                    label_tfm.SetOrigin(image_tfm.GetOrigin())
                    label_tfm.SetDirection(image.GetDirection())
                    label_tfm.SetSpacing(image_tfm.GetSpacing())

                    sample = {'image':image_tfm, 'label': label_tfm}

                    for transform in transforms:
                        sample = transform(sample)

                    image_tfm, label_tfm = sample['image'], sample['label']

                    # create empty softmax image in pair with transformed image
                    softmax_tfm = sitk.Image(image_tfm.GetSize(),sitk.sitkFloat32)
                    softmax_tfm.SetOrigin(image_tfm.GetOrigin())
                    softmax_tfm.SetDirection(image.GetDirection())
                    softmax_tfm.SetSpacing(image_tfm.GetSpacing())

                    # convert image to numpy array
                    image_np = sitk.GetArrayFromImage(image_tfm)
                    image_np = np.asarray(image_np,np.float32)

                    label_np = sitk.GetArrayFromImage(label_tfm)
                    label_np = np.asarray(label_np,np.int32)

                    softmax_np = sitk.GetArrayFromImage(softmax_tfm)
                    softmax_np = np.asarray(softmax_np,np.float32)

                    # unify numpy and sitk orientation
                    image_np = np.transpose(image_np,(2,1,0))
                    label_np = np.transpose(label_np,(2,1,0))
                    softmax_np = np.transpose(softmax_np,(2,1,0))

                    # a weighting matrix will be used for averaging the overlapped region
                    weight_np = np.zeros(label_np.shape)

                    # prepare image batch indices
                    inum = int(math.ceil((image_np.shape[0]-FLAGS.patch_size)/float(FLAGS.stride_inplane))) + 1
                    jnum = int(math.ceil((image_np.shape[1]-FLAGS.patch_size)/float(FLAGS.stride_inplane))) + 1
                    knum = int(math.ceil((image_np.shape[2]-FLAGS.patch_layer)/float(FLAGS.stride_layer))) + 1

                    patch_total = 0
                    ijk_patch_indices = []
                    ijk_patch_indicies_tmp = []

                    for i in range(inum):
                        for j in range(jnum):
                            for k in range (knum):
                                if patch_total % FLAGS.batch_size == 0:
                                    ijk_patch_indicies_tmp = []

                                istart = i * FLAGS.stride_inplane
                                if istart + FLAGS.patch_size > image_np.shape[0]: #for last patch
                                    istart = image_np.shape[0] - FLAGS.patch_size
                                iend = istart + FLAGS.patch_size

                                jstart = j * FLAGS.stride_inplane
                                if jstart + FLAGS.patch_size > image_np.shape[1]: #for last patch
                                    jstart = image_np.shape[1] - FLAGS.patch_size
                                jend = jstart + FLAGS.patch_size

                                kstart = k * FLAGS.stride_layer
                                if kstart + FLAGS.patch_layer > image_np.shape[2]: #for last patch
                                    kstart = image_np.shape[2] - FLAGS.patch_layer
                                kend = kstart + FLAGS.patch_layer

                                ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                                if patch_total % FLAGS.batch_size == 0:
                                    ijk_patch_indices.append(ijk_patch_indicies_tmp)

                                patch_total += 1

                    batches = prepare_batch(image_np,ijk_patch_indices)

                    # acutal segmentation
                    for i in tqdm(range(len(batches))):
                        batch = batches[i]
                        [pred, softmax] = sess.run(['predicted_label/prediction:0','softmax/softmax:0'], feed_dict={'images_placeholder:0': batch})
                        istart = ijk_patch_indices[i][0][0]
                        iend = ijk_patch_indices[i][0][1]
                        jstart = ijk_patch_indices[i][0][2]
                        jend = ijk_patch_indices[i][0][3]
                        kstart = ijk_patch_indices[i][0][4]
                        kend = ijk_patch_indices[i][0][5]
                        label_np[istart:iend,jstart:jend,kstart:kend] += pred[0,:,:,:]
                        softmax_np[istart:iend,jstart:jend,kstart:kend] += softmax[0,:,:,:,1]
                        weight_np[istart:iend,jstart:jend,kstart:kend] += 1.0

                    print("{}: Evaluation complete".format(datetime.datetime.now()))
                    # eliminate overlapping region using the weighted value
                    label_np = np.rint(np.float32(label_np)/np.float32(weight_np) + 0.01)
                    softmax_np = softmax_np/np.float32(weight_np)

                    # convert back to sitk space
                    label_np = np.transpose(label_np,(2,1,0))
                    softmax_np = np.transpose(softmax_np,(2,1,0))

                    # convert label numpy back to sitk image
                    label_tfm = sitk.GetImageFromArray(label_np)
                    label_tfm.SetOrigin(image_tfm.GetOrigin())
                    label_tfm.SetDirection(image.GetDirection())
                    label_tfm.SetSpacing(image_tfm.GetSpacing())

                    softmax_tfm = sitk.GetImageFromArray(softmax_np)
                    softmax_tfm.SetOrigin(image_tfm.GetOrigin())
                    softmax_tfm.SetDirection(image.GetDirection())
                    softmax_tfm.SetSpacing(image_tfm.GetSpacing())

                    # resample the label back to original space
                    resampler = sitk.ResampleImageFilter()
                    # save segmented label
                    writer = sitk.ImageFileWriter()

                    resampler.SetInterpolator(1)
                    resampler.SetOutputSpacing(image.GetSpacing())
                    resampler.SetSize(image.GetSize())
                    resampler.SetOutputOrigin(image.GetOrigin())
                    resampler.SetOutputDirection(image.GetDirection())

                    print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
                    label = resampler.Execute(label_tfm)
                    writer.SetFileName(output_path)
                    writer.Execute(label)
                    print("{}: Save evaluate label at {} success".format(datetime.datetime.now(),output_path))

                    #print("{}: Resampling probability map back to original image space...".format(datetime.datetime.now()))
                    #prob = resampler.Execute(softmax_tfm)
                    #prob_path = os.path.join(FLAGS.output_dir,'probability_vnet.mha')
                    #writer.SetFileName(prob_path)
                    #writer.Execute(prob)
                    #print("{}: Save evaluate probability map at {} success".format(datetime.datetime.now(),prob_path))

def main(argv=None):
    evaluate()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/pirl/Downloads/cardiac/data/test/CHD',
                        help='Directory of evaluation data.')

    parser.add_argument('--output_dir', type=str, default='/home/pirl/Downloads/cardiac/data/output/CHD',
                        help='Directory where to write checkpoint')

    parser.add_argument('--image_filename', type=str, default='_dia.mha',
                        help='Image filename')

    parser.add_argument('--model_path', type=str, default='/home/pirl/Downloads/cardiac/data/model/CHD_dia/checkpoint.meta',
                        help='Path to saved models')

    parser.add_argument('--checkpoint_path', type=str, default='/home/pirl/Downloads/cardiac/data/model/CHD_dia/checkpoint',
                        help='Directory of saved checkpoints')

    parser.add_argument('--patch_size', type=int, default=192,
                        help='Size of a data patch')

    parser.add_argument('--patch_layer', type=int, default=64,
                        help='Number of layers in data patch')

    parser.add_argument('--stride_inplane', type=int, default=128,
                        help='Stride size in 2D plane')

    parser.add_argument('--stride_layer', type=int, default=16,
                        help='Stride size in layer direction')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Setting batch size (currently only accept 1)')

    parser.add_argument('--max_sd_coe', type=float, default=3,
                        help='maximum standard deviation coefficient')

    parser.add_argument('--min_sd_coe', type=float, default=0.5,
                        help='minimum standard deviation coefficient')

    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Standard deviation')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


