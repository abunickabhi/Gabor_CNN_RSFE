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
from functions import get_model, test, save_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.models import Model
from keras.layers import Lambda
from keras.layers import Reshape
from keras.callbacks import Callback
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers import BatchNormalization
from keras import optimizers, regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  *
from keras.layers.core import Lambda

img=loadmat('/home/viral/Desktop/Gabor/img.mat')
img=img['Clip_Feature_class'] 
img=img.astype('float32')
indx=img.shape
im=img/np.max(img)
gt=img[:,:,12]
img=img[:,:,0:12]
img1=img[:,:,5]
plt.imshow(gt,cmap='gnuplot2')
images=img1
image=img1
image_names='Gabor'

# These functions are taken from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i

def power(image, kernel):
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

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


from scipy.signal import convolve2d
res = convolve2d(img1,kernels[1:], mode='valid') 
plt.show(res); # title('response') Book figure
'''
The training part of the image stack starts from here
'''
folder='/home/abhi_daiict/Desktop/Hyperspectral_dataset/GaborCNN'
def dtst(folder):  
    img=loadmat('/home/viral/Desktop/Gabor/img.mat')
    img=img['Clip_Feature_class'] 
    #gt=img[:,:,12]
    img=img[:,:,0:12]   
    ignored_labels = [0] 
    gtr=np.load('/home/viral/Desktop/Gabor/gtr.npy')
    return img,gtr,ignored_labels
    
LABEL_VALUES=['Tm2','Bs2','Bw1','De1','Ds1','Dw1''Dw2','Fv1','Sv2','Fv2',
'Fw1','Iw1','NAD','S','Ss1','Sv1','Sw1','Sw2','W','De2']

 
img, gtr, IGNORED_LABELS = dtst(folder)
N_CLASSES = 20
N_BANDS = 12

N=0.8 #The amount of training samples that need to be randomly selected
#Here I am taking 100% of the samples as I have not made the boxes that are validation sets each one at a time
results = []
SAMPLE_PERCENTAGE = N
train_gt, test_gt = sample_gt(gtr, SAMPLE_PERCENTAGE)
    
MODEL = 'nn'

hyperparams = vars(args)
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
model,model_name, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
      
# Split train set in train/val        
train_gt, val_gt = sample_gt(train_gt, N)
train_loader = HyperX(img, train_gt, **hyperparams)
val_loader = HyperX(img, val_gt, **hyperparams)
train_steps=train_loader.__len__()
valid_steps=val_loader.__len__()   

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model_dir = './checkpoints/'
filename = '/history.txt'
checkpoint_name = "_epoch{epoch:02d}_{val_acc:.2f}.h5"
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_acc',save_best_only=True, save_weights_only=True,verbose=0, mode='max')
#cb_hist = SaveHistory(filename)    
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
history=model.fit_generator(train_loader, epochs=hyperparams['epoch'], steps_per_epoch = train_steps,validation_data=val_loader, validation_steps=valid_steps,callbacks=[checkpoint])
probabilities = test(model, img, hyperparams)
prediction = np.argmax(probabilities, axis=-1)+1
run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
results.append(run_results)
show_results(run_results, label_values=LABEL_VALUES)
color_prediction = convert_to_color(prediction)
display_predictions(color_prediction, gt=convert_to_color(test_gt), caption="Prediction vs. test ground truth")


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

