import os
import numpy as np
import argparse
from model import average_dice_coef, average_dice_coef_loss, load_model
from preprocessing import MyDataFlow, gen_data, Preprocess
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
from keras.optimizers import Adam
from tensorpack.dataflow.common import MapData, MapDataComponent
from tensorpack.dataflow import PrefetchData
from scipy import ndimage

os.environ['CUDA_VISIBLE_DEVICES']= '0'
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/home/pirl/Downloads/cardiac/data/train/CHD',
                    help='Directory of stored data.')
parser.add_argument('--checkpoint_dir',type=str, default='/home/pirl/Downloads/cardiac/data/model/CHD_dia',
                    help='Directory where to write checkpoint')
parser.add_argument('--image_filename', type=str, default='_dia.mha',
                    help='Image filename')
parser.add_argument('--label_filename',type=str, default='_dia_M.mha',
                    help='label_filename')
parser.add_argument('--chd_hcmp', type=str, default='chd',
                    help='chd or hcmp')
parser.add_argument('--task_detail',type=str,default='dia')
parser.add_argument('--epochs',type=int, default=1,
                    help='Training Epochs')
parser.add_argument('--init_learning_rate',type=float, default=0.02,
                    help='Initial learning rate')


FLAGS, unparsed = parser.parse_known_args()
if FLAGS.chd_hcmp != "chd" and FLAGS.chd_hcmp != 'hcmp':
    raise NotImplementedError('choose "chd" or "hcmp" , got {}'.format(FLAGS.chd_hcmp))
if FLAGS.task_detail != "dia" and FLAGS.task_detail !='sys' and FLAGS.task_detail !='A' and FLAGS.task_detail !='N':
    raise NotImplementedError('choose "dia" or "sys" or "A" or "N"')
process = Preprocess()

print('data directory :',FLAGS.data_dir)
print('checkpoint directory :',FLAGS.checkpoint_dir)

train_data_dir = FLAGS.data_dir
image_filelist = []
for file in os.listdir(train_data_dir+'/image/'):
    if FLAGS.image_filename in file:
        image_filelist.append(os.path.join(train_data_dir,'image',file))
print(image_filelist)

num_labels = 3
num_channels = 1
input_shape = (None,None,None,num_channels)
output_shape = (None,None,None,num_labels)
model = load_model(input_shape=input_shape,
                    num_labels=num_labels,
                    base_filter=32,
                    depth_size=3,
                    se_res_block=True,
                    se_ratio=16,
                    last_relu=False)
print(model)

model.compile(optimizer=Adam(lr=1e-4), loss=average_dice_coef_loss, metrics=[average_dice_coef])


df = MyDataFlow(train_data_dir, FLAGS.image_filename, FLAGS.label_filename, shuffle=True)
df = MapDataComponent(df, process.simple_preprocess_img,index=0)
df = MapDataComponent(df, process.sample_z_norm, index=0)
df = MapDataComponent(df, process.simple_preprocess_mask, index=1)
df = MapData(df, process.resize_whole)
df = MapData(df, process.data_aug)
df = PrefetchData(df, 2, 1)

gen_train = gen_data(df)

cb_early_stopping = EarlyStopping(monitor='loss', patience=57)
cbs = list()
cbs.append(ModelCheckpoint('{}/checkpoint_{}_{}.h5'.format(FLAGS.checkpoint_dir,FLAGS.chd_hcmp,FLAGS.task_detail), save_best_only=True, monitor='loss', period=1))
cbs.append(CSVLogger('{}/checkpoint.log'.format(FLAGS.checkpoint_dir), append=True))
#cbs.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001))
cbs.append(TensorBoard(log_dir='{}'.format(FLAGS.checkpoint_dir), histogram_freq=0,
                            write_graph=True, write_images=True, update_freq='epoch'))

history = model.fit_generator(generator=gen_train,
                    steps_per_epoch=len(image_filelist),
                    epochs=FLAGS.epochs,
                    max_queue_size=1,
                    workers=1,
                    use_multiprocessing=False,
                    callbacks=cbs)
