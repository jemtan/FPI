# Code for Foreign Patch Interpolation (FPI) Submission for MOOD 2020

### Abstract:
In medical imaging, outliers can contain hypo/hyper-intensities, minor deformations, or completely different anatomy altogether. To detect these irregularities it is helpful to learn the features present in both normal and abnormal images. However this is difficult because of the wide range of possible abnormalities and also the number of ways that normal anatomy can vary naturally. As such, we leverage the natural variations in normal anatomy to create a range of synthetic abnormalities. Specifically, the same patch region is extracted from two independent samples and is replaced with an interpolation between both patches. The interpolation factor, patch size, and patch location are randomly sampled from uniform distributions. A wide residual encoder decoder is trained to give a pixel-wise prediction of the patch and its interpolation factor. This encourages the network to learn what features to expect normally and to identify where foreign patterns have been introduced. However, optimization must be done carefully to avoid overfitting to the self-supervised task because generalization is essential for outlier detection with this type of approach. The outlier score is derived directly from the estimate of the interpolation factor. Meanwhile the pixel-wise output allows for pixel- and subject- level predictions using the same model.

### Key files:  
self_sup_task.py - FPI operation used to create self-supervised task
fpiSubmit.py - training/testing loops
models/wide_residual_network.py - network architecture  
readData.py - data processing, reading/writing volumes
train_simple.py - run training and save models  
pred_simple.py - generate predictions on test data 
eval_simple.py - calculate score (average precision) 


#### Run training with:
```
python train_simple.py -i <input_dir> -o <output_dir> -d <brain/abdom>
```
Then place desired models into restore_dir/brain and restore_dir/abdom respectively


#### Generate predictions with:
```
python pred_simple.py -i <input_dir> -o <output_dir> -m <sample/pixel> -d <brain/abdom>
```

#### Calculate score with:
```
python eval_simple.py -l <label_dir> -o <output_dir> -m <sample/pixel> -d <brain/abdom>
```
Note that there are no test samples provided, but you can create your own and provide the labels in label_dir. Follow the format of the toy test samples provided by the MOOD challenge.


#### Disclaimer
This is an initial release which uses very old packages. These will be updated soon to make the code more accessible. If you prefer, you can use the code in self_sup_task.py to train your own architecture.
This repository includes code from the respository linked below. In particular, the network architecture is built on top of their wide resnet implementation. The linked repository is a great outlier detection method also based on a self-supervised task. If you are interested in this topic I definitely recommend you check out their paper/code!
```
https://github.com/izikgo/AnomalyDetectionTransformations
```

### Author Information
```
Jeremy Tan, Benjamin Hou, James Batten, Huaqi Qiu, and Bernhard Kainz.: Foreign Patch Interpolation. Medical Out-of-Distribution Analysis Challenge at MICCAI. (2020)

j.tan17@imperial.ac.uk
```


