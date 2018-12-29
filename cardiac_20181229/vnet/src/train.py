from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import NiftiDataset
import os
import VNet
import math
import datetime
import sys
import argparse

# select gpu devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # e.g. "0,1,2", "0,2"

def placeholder_inputs(input_batch_shape, output_batch_shape):

    images_placeholder = tf.placeholder(tf.float32, shape=input_batch_shape, name="images_placeholder")
    labels_placeholder = tf.placeholder(tf.int32, shape=output_batch_shape, name="labels_placeholder")

    return images_placeholder, labels_placeholder

def dice_coe(output, target, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):

    inse = tf.reduce_sum(tf.multiply(output,target), axis=axis)

    if loss_type == 'jaccard':
        l = tf.reduce_sum(tf.multiply(output,output), axis=axis)
        r = tf.reduce_sum(tf.multiply(target,target), axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknown loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (tf.constant(2.0) * tf.cast(inse,dtype=tf.float32) + tf.constant(smooth)) / (tf.cast(l + r, dtype=tf.float32) + tf.constant(smooth))

    dice = tf.reduce_mean(dice)
    return dice

def train():
    """Train the Vnet model"""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()


        # patch_shape(batch_size, height, width, depth, channels)

        input_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1)
        output_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1)
        
        images_placeholder, labels_placeholder = placeholder_inputs(input_batch_shape,output_batch_shape)

        for batch in range(FLAGS.batch_size):
            images_log = tf.cast(images_placeholder[batch:batch+1,:,:,:,0], dtype=tf.uint8)
            labels_log = tf.cast(tf.scalar_mul(255,labels_placeholder[batch:batch+1,:,:,:,0]), dtype=tf.uint8)

            #tf.summary.image("image", tf.transpose(images_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
            #tf.summary.image("label", tf.transpose(labels_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
            #tf.summary.image("label2",tf.transpose(tf.cast(labels_placeholder[batch:batch+1,:,:,:,1]),dtype=tf.uint8))

        # Get images and labels
        train_data_dir = os.path.join(FLAGS.data_dir)

        test_data_dir = os.path.join(FLAGS.data_dir,'testing')
        # support multiple image input, but here only use single channel, label file should be a single file with different classes

        # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
        with tf.device('/cpu:0'):
            # create transformations to image and labels
            trainTransforms = [
                NiftiDataset.StatisticalNormalization(FLAGS.sigma,FLAGS.max_sd_coe, FLAGS.min_sd_coe),
                # NiftiDataset.Normalization(),
                NiftiDataset.Resample((0.5,0.5,0.5)),
                NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel,FLAGS.min_ratio),
                NiftiDataset.RandomNoise()
                ]

            TrainDataset = NiftiDataset.NiftiDataset(
                data_dir=train_data_dir,
                image_filename=FLAGS.image_filename,
                label_filename=FLAGS.label_filename,
                transforms=trainTransforms,
                train=True
                )

            trainDataset = TrainDataset.get_dataset()
            trainDataset = trainDataset.shuffle(buffer_size=5)
            trainDataset = trainDataset.batch(FLAGS.batch_size)

        train_iterator = trainDataset.make_initializable_iterator()
        next_element_train = train_iterator.get_next()

        #next_element_test = test_iterator.get_next()

        # Initialize the model
        with tf.name_scope("vnet"):
            model = VNet.VNet(
                num_classes=4, # binary for 2
                keep_prob=0.8, # default 1
                num_channels=16, # default 16 
                num_levels=4,  # default 4
                num_convolutions=(1,2,3,3), # default (1,2,3,3), size should equal to num_levels
                bottom_convolutions=3, # default 3
                activation_fn="prelu") # default relu

            logits = model.network_fn(images_placeholder)


        # # Exponential decay learning rate
        train_batches_per_epoch = math.ceil(TrainDataset.data_size/FLAGS.batch_size)
        decay_steps = train_batches_per_epoch*FLAGS.decay_steps

        with tf.name_scope("learning_rate"):
            learning_rate = FLAGS.init_learning_rate
            learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate,
                 global_step,
                 decay_steps,
                 FLAGS.decay_factor,
                 staircase=True)
        #tf.summary.scalar('learning_rate', learning_rate)

        # softmax op for probability layer
        with tf.name_scope("softmax"):
            softmax_op = tf.nn.softmax(logits,name="softmax")

        # Op for calculating loss
        with tf.name_scope("cross_entropy"):
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.squeeze(labels_placeholder,
                squeeze_dims=[4])))
        #tf.summary.scalar('loss',loss_op)

        with tf.name_scope("weighted_cross_entropy"):
            class_weights = tf.constant([1.0, 1.0])

            # deduce weights for batch samples based on their true label
            onehot_labels = tf.one_hot(tf.squeeze(labels_placeholder,squeeze_dims=[4]),depth = 2)

            weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
            # compute your (unweighted) softmax cross entropy loss
            unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.squeeze(labels_placeholder, squeeze_dims = [4]))
            #labels = tf.squeeze(labels_placeholder, squeeze_dims = [4])
            # apply the weights, relying on broadcasting of the multiplication
            weighted_loss = unweighted_loss * weights
            # reduce the result to get your final loss
            weighted_loss_op = tf.reduce_mean(weighted_loss)
                
        #tf.summary.scalar('weighted_loss',weighted_loss_op)

        # Argmax Op to generate label from logits
        with tf.name_scope("predicted_label"):
            pred = tf.argmax(logits, axis=4 , name="prediction")

        for batch in range(FLAGS.batch_size):
            pred_log = tf.cast(tf.scalar_mul(255,pred[batch:batch+1,:,:,:]), dtype=tf.uint8)
            #tf.summary.image("pred", tf.transpose(pred_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

        # Accuracy of model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.expand_dims(pred,-1), tf.cast(labels_placeholder,dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #tf.summary.scalar('accuracy', accuracy)


        # Dice Similarity, currently only for binary segmentation
        with tf.name_scope("dice"):
            # sorensen = dice_coe(tf.expand_dims(softmax_op[:,:,:,:,1],-1),tf.cast(labels_placeholder,dtype=tf.float32), loss_type='sorensen')
            # jaccard = dice_coe(tf.expand_dims(softmax_op[:,:,:,:,1],-1),tf.cast(labels_placeholder,dtype=tf.float32), loss_type='jaccard')
            #print(labels_placeholder[:,:,:,:,0])

            sorensen = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0],depth=4),dtype=tf.float32), loss_type='sorensen', axis=[1,2,3,4])
            jaccard = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0],depth=4),dtype=tf.float32), loss_type='jaccard', axis=[1,2,3,4])
            sorensen_loss = 1. - sorensen
            jaccard_loss = 1. - jaccard
            #print(sorensen)
        #tf.summary.scalar('sorensen', sorensen)
        #tf.summary.scalar('jaccard', jaccard)
        #tf.summary.scalar('sorensen_loss', sorensen_loss)
        #tf.summary.scalar('jaccard_loss',jaccard_loss)

        # Training Op
        with tf.name_scope("training"):
            # optimizer
            if FLAGS.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.init_learning_rate)
            elif FLAGS.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.init_learning_rate)
            elif FLAGS.optimizer == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.init_learning_rate, momentum=FLAGS.momentum)
            elif FLAGS.optimizer == "nesterov_momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.init_learning_rate, momentum=FLAGS.momentum, use_nesterov=True)
            else:
                sys.exit("Invalid optimizer");

            # loss function
            if (FLAGS.loss_function == "xent"):
                loss_fn = loss_op
            elif(FLAGS.loss_function == "weight_xent"):
                loss_fn = weighted_loss_op
            elif(FLAGS.loss_function == "sorensen"):
                loss_fn = sorensen_loss
            elif(FLAGS.loss_function == "jaccard"):
                loss_fn = jaccard_loss
            else:
                sys.exit("Invalid loss function");

            train_op = optimizer.minimize(
                loss = loss_fn,
                global_step=global_step)

        # # epoch checkpoint manipulation
        start_epoch = tf.get_variable("start_epoch", shape=[1], initializer= tf.zeros_initializer,dtype=tf.int32)
        start_epoch_inc = start_epoch.assign(start_epoch+1)

        # saver
        summary_op = tf.summary.merge_all()
        checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir ,"checkpoint")
        print("Setting up Saver...")
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.4

        # training cycle
        with tf.Session(config=config) as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            print("{}: Start training...".format(datetime.datetime.now()))

            # summary writer for tensorboard
            #train_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            #test_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)

            # restore from checkpoint
            if FLAGS.restore_training:
                # check if checkpoint exists
                if os.path.exists(checkpoint_prefix+"-latest"):
                    print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),FLAGS.checkpoint_dir))
                    latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir,latest_filename="checkpoint-latest")
                    saver.restore(sess, latest_checkpoint_path)

            print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval()[0]))
            print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(sess, global_step)))

            # loop over epochs
            for epoch in np.arange(start_epoch.eval(), FLAGS.epochs):
                # initialize iterator in each new epoch
                sess.run(train_iterator.initializer)

                sess.run(test_iterator.initializer)
                print("{}: Epoch {} starts".format(datetime.datetime.now(),epoch+1))

                # training phase
                while True:
                    try:

                        [image, label] = sess.run(next_element_train)
                        image = image[:,:,:,:,np.newaxis]
                        label = label[:,:,:,:,np.newaxis]
                        model.is_training = True;
                        sess.run([train_op], feed_dict={images_placeholder: image, labels_placeholder: label})

                        #train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

                    except tf.errors.OutOfRangeError:
                        start_epoch_inc.op.run()
                        # print(start_epoch.eval())
                        # save the model at end of each epoch training
                        print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,FLAGS.checkpoint_dir))
                        if not (os.path.exists(FLAGS.checkpoint_dir)):
                            os.makedirs(FLAGS.checkpoint_dir,exist_ok=True)
                        saver.save(sess, checkpoint_prefix,
                            latest_filename="checkpoint-latest")
                        print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))
                        break
                
                # testing phase
                print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
                # while True:
                #     try:
                #         [image, label] = sess.run(next_element_test)
                #
                #         #image = image[:,:,:,:,np.newaxis]
                #         #label = label[:,:,:,:,np.newaxis]
                #
                #         model.is_training = False;
                #         #loss, summary = sess.run([loss_op, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                #         #test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
                #
                #     except tf.errors.OutOfRangeError:
                #         break

        # close tensorboard summary writer
        #train_summary_writer.close()
        #test_summary_writer.close()


def main(argv=None):
    if not FLAGS.restore_training:
        # clear log directory
        #if tf.gfile.Exists(FLAGS.log_dir):
        #    tf.gfile.DeleteRecursively(FLAGS.log_dir)
        #tf.gfile.MakeDirs(FLAGS.log_dir)

        # clear checkpoint directory
        if tf.gfile.Exists(FLAGS.checkpoint_dir):
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

        # clear model directory
        #if tf.gfile.Exists(FLAGS.model_dir):
        #    tf.gfile.DeleteRecursively(FLAGS.model_dir)
        #tf.gfile.MakeDirs(FLAGS.model_dir)

    train()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/pirl/Downloads/cardiac/data/train/CHD',
                    help='Directory of stored data.')

    parser.add_argument('--checkpoint_dir',type=str, default='/home/pirl/Downloads/cardiac/data/model/CHD_dia',
                    help='Directory where to write checkpoint')

    parser.add_argument('--image_filename', type=str, default='_dia.mha',
                    help='Image filename')

    parser.add_argument('--label_filename',type=str, default='_dia_M.mha',
                    help='label_filename')

    parser.add_argument('--roi_seg', type=str, default='seg',
                        help='ROI or SEG Mode choose')

    parser.add_argument('--batch_size',type=int, default=1,
                    help='Size of batch')

    parser.add_argument('--patch_size',type=int, default=192,
                    help='Size of a data patch')

    parser.add_argument('--patch_layer',type=int, default=64,
                    help='Number of layers in data patch')

    parser.add_argument('--epochs',type=int, default=100,
                    help='Number of epochs for training')

    parser.add_argument('--init_learning_rate',type=float, default=0.02,
                    help='Initial learning rate')

    parser.add_argument('--decay_factor',type=float, default=0.07,
                    help='Exponential decay learning rate factor')

    parser.add_argument('--decay_steps',type=int, default=1000,
                    help='Number of epoch before applying one learning rate decay')

    parser.add_argument('--display_step',type=int, default=10,
                    help='Display and logging interval (train steps)')

    parser.add_argument('--save_interval',type=int, default=1,
                    help='Checkpoint save interval (epochs)')

    parser.add_argument('--restore_training',type=bool, default=True,
                    help='Restore training from last checkpoint')

    parser.add_argument('--drop_ratio',type=float, default=0.1,
                    help='Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1')

    parser.add_argument('--min_ratio',type=float, default=0.2,
                    help='Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1')

    parser.add_argument('--min_pixel',type=int, default=500,
                    help='Minimum non-zero pixels in the cropped label')

    parser.add_argument('--shuffle_buffer_size',type=int, default=5,
                    help='Number of elements used in shuffle buffer')

    parser.add_argument('--loss_function',type=str,default='sorensen',
                    help='Loss function used in optimization (xent, weight_xent, sorensen, jaccard)')

    parser.add_argument('--optimizer',type=str,default='adam',
                    help='Optimization method (sgd, adam, momentum, nesterov_momentum)')

    parser.add_argument('--momentum',type=float,default=0.5,
                    help='Momentum used in optimization')

    parser.add_argument('--max_sd_coe',type=float,default=3,
                    help='maximum standard deviation coefficient')

    parser.add_argument('--min_sd_coe',type=float,default=0.5,
                    help='minimum standard deviation coefficient')

    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Standard deviation')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)
    print('************ HYPER PARAMETETR ***************')
    print('max_sd_coe : ',FLAGS.max_sd_coe, 'min_sd_coe : ', FLAGS.min_sd_coe)
    print('EPOCHS : ', FLAGS.epochs)
    print('init learning rate :', FLAGS.init_learning_rate)
    print('drop_ratio :', FLAGS.drop_ratio)
    print('patch_size :', FLAGS.patch_size)
    print('patch_layer :', FLAGS.patch_layer)
    print('sigma :', FLAGS.sigma)

    print('*********************************************')