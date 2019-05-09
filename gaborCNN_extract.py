"""
Created on Fri Apr 26 12:08:59 2019

@author: abhi
"""
import numpy as np
from scipy.io import loadmat 
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from sklearn.model_selection import train_test_split
from functions import *
from keras.callbacks import ModelCheckpoint

folder='/home/abhi_daiict/Desktop/Hyperspectral_dataset/GaborCNN'
def dtst(folder):  
    #img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/GaborCNN/img.mat')
    #img=img['Clip_Feature_class'] 
    img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/GaborCNN/D1.mat')
    img=img['m'] 
    #img=np.load('/home/abhi/Desktop/Hyperspectral Datasets/GaborCNN/Gabor/band1.npy')
    #gtr=img[:,:,12]   
    img=img[:,:,0:12]       
    ignored_labels = [0] 
    gtr=np.load('/home/abhi/Desktop/Hyperspectral Datasets/GaborCNN/gtr.npy')
    return img,gtr,ignored_labels

LABEL_VALUES=['Tm2','Bs2','Bw1','De1','Ds1','Dw1','Dw2','Fv1','Sv2','Fv2',
'Fw1','Iw1','NAD','S','Ss1','Sv1','Sw1','Sw2','W','De2']

# Filter bank for Gabor filter
kernels = []
for theta in range(4):
    #Prepare filter bank as per the gabor formula
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)                
results = []
kernel_params = []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        results.append((kernel, [power(img1, kernel) for img1 in images]))
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
plt.gray()

for label, img1, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img1)
    ax.set_title(label, fontsize=9)
    ax.axis('off')
for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
plt.show()

'''
The training part of the image stack starts from here
'''
img, gtr, IGNORED_LABELS = dtst(folder)
N_CLASSES = 20
N_BANDS = 12
#gt=gt_change(gtr)

N=0.8 #The amount of training samples that need to be randomly selected
#Here I am taking 100% of the samples as I have not made the boxes that are validation sets each one at a time
results = []
SAMPLE_PERCENTAGE = N
train_gt, test_gt = sample_gt(gtr, SAMPLE_PERCENTAGE)
    
import argparse
dataset_names= 'ip'
parser = argparse.ArgumentParser(description="Run deep learning experiments")
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--folder', type=str)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--restore', type=str, default=None)
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=int, default=10)
group_dataset.add_argument('--sampling_mode', type=str)
group_dataset.add_argument('--train_set', type=str, default=None)
group_dataset.add_argument('--test_set', type=str, default=None)
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int)
group_train.add_argument('--patch_size', type=int)
group_train.add_argument('--lr', type=float)
group_train.add_argument('--class_balancing', action='store_true')
group_train.add_argument('--batch_size', type=int)
group_test = parser.add_argument_group('Test')
group_test.add_argument('--test_stride', type=int, default=1)
group_test.add_argument('--inference', type=str, default=None, nargs='?')
args = parser.parse_args()
SAMPLE_PERCENTAGE = args.training_sample / 100
DATASET = args.dataset
MODEL = args.model
N_RUNS = args.runs
PATCH_SIZE = args.patch_size
FOLDER = args.folder
EPOCH = args.epoch
SAMPLING_MODE = args.sampling_mode
CHECKPOINT = args.restore
LEARNING_RATE = args.lr
CLASS_BALANCING = args.class_balancing
TRAIN_GT = args.train_set
TEST_GT = args.test_set
INFERENCE = args.inference
TEST_STRIDE = args.test_stride

MODEL = 'nn'
hyperparams = vars(args)
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
model,model_name, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
      
# Split train set in train/val        
train_gt, val_gt = sample_gt(train_gt, N)
train_loader = dtimport(img, train_gt, **hyperparams)
val_loader = dtimport(img, val_gt, **hyperparams)
train_steps=train_loader.__len__()
valid_steps=val_loader.__len__()    

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model_dir = './checkpoints/'
filename = '/history.txt'
checkpoint_name = "_epoch{epoch:02d}_{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_acc',save_best_only=True, 
                             save_weights_only=True,verbose=0, mode='max')
#cb_hist = SaveHistory(filename)    
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
history=model.fit_generator(train_loader, epochs=hyperparams['epoch'],
                            steps_per_epoch = train_steps,validation_data=val_loader, 
                            validation_steps=valid_steps,callbacks=[checkpoint])
probabilities = test(model, img, hyperparams)
prediction = np.argmax(probabilities, axis=-1)+1
run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
results.append(run_results)
show_results(run_results, label_values=LABEL_VALUES)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''
hyperparams = vars(args)
# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
'''