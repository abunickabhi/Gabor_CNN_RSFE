# GaborCNN_Surendranagar

![Desertification Classes](https://github.com/abunickabhi/GaborCNN_Surendranagar/blob/master/class.png)
![Thresholded Image representation](https://github.com/abunickabhi/GaborCNN_Surendranagar/blob/master/gt.png)

Gabor feature extraction and contextual CNN on multispectral surendranagar data.

The region over Surendranagar has R, G, B, and NIR images and the samples of train and test are selected from it via random sampling of the points.

The Spatial Resolution of the dataset is 56m and has 4 bands. The Convolutional Neural Networks are designed to fit on Multispectral bands. 

First, there is a loss of information due to reduced spectra and less ranging values. Gabor filter is a linear filter used for texture analysis. Since multispectral data also have intraclass variability, extracting the texture component is essential to get accurate input representation.

