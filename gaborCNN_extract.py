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
img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/GaborCNN/img.mat')
img=img['Clip_Feature_class'] 
img=img.astype('float32')
indx=img.shape
im=img/np.max(img)
gt=img[:,:,12]
img=img[:,:,0:12]
plt.imshow(gt,cmap='gnuplot2')
images=img
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
        results.append((kernel, [power(img, kernel) for img in images]))
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
plt.gray()

for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
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
folder='/home/abhi_daiict/Desktop/Hyperspectral_dataset/GaborCNN'
def dtst(folder):  
    img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/GaborCNN/img.mat')
    img=img['Clip_Feature_class'] 
    gt=img[:,:,12]
    img=img[:,:,0:12]   
    ignored_labels = [0] 
    return img,gt,ignored_labels

'''    
All the class labels
1	Tm2
6	Bs2
7	Bw1
8	De1
10	Ds1
12	Dw1
13	Dw2
14	Fv1
15	Sv2
16	Fv2
17	Fw1
22	Iw1
23	NAD
25	S
26	Ss1
28	Sv1
29	Sw1
30	Sw2
32	W
33	De2
'''

img, gt, IGNORED_LABELS = dtst(folder)
N_CLASSES = 20
N_BANDS = 12

N=1. #The amount of training samples that need to be randomly selected
#Here I am taking 100% of the samples as I have not made the boxes that are validation sets each one at a time
results = []
SAMPLE_PERCENTAGE = N
train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE)
    
MODEL = 'nn'
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
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_acc',save_best_only=True, save_weights_only=True,verbose=0, mode='max')
cb_hist = SaveHistory(filename)    
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
history=model.fit_generator(train_loader, epochs=hyperparams['epoch'], steps_per_epoch = train_steps,validation_data=val_loader, validation_steps=valid_steps,callbacks=[checkpoint])
probabilities = test(model, img, hyperparams)
prediction = np.argmax(probabilities, axis=-1)+1
run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
results.append(run_results)
show_results(run_results, label_values=LABEL_VALUES)


'''
hyperparams = vars(args)

# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
'''