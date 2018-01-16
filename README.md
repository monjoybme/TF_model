# Simple utils and model for experimenting with your own images using Tensorflow (e.g. MNIST,CIFAR10). 

Written for myself. I will be gald if you find it useful too.

Code for resizing images in a folder and split them into train/test directories for further TFrecord creation.

Model.py - a simple CNN model for working with TFrecods files containing images and label.
To make model work on yor own images you need:

# 1. Prepare images for traning, i.e. resize to a unified size and split it randomly to train/test datasets.
```
Resize_and_slit.py
```
Input to the script are row images in the folder structure like this ```../image_class/images_in_class.jpg```
<br /> ==> the script creates two separate folders ```/TRAIN_DIR/``` and ```/VALIDATION_DIR/```
<br /> ==> the script split resized imagaes randomly to 80% to ```TRAIN_DIR```, 20% to ```VALIDATION_DIR```
<br /> ==> the script resizes images to a defined size

# 2. Create train/test TFrecords files from prepared images.
```
build_image_data.py
```
<br />==> Create labels.txt and string by string write classes there (example included).
<br /> ==> Change `'train_directory'` value to your addrs to ```/TRAIN_DIR/``` and ```'validation_directory'``` to ```/VALIDATION_DIR/```
<br />==> script creates two TFRecords files ```/output/train-00000-of-00001 and /output/validation-00000-of-00001```

# 3. Train the CNN model
```
Model.py
```
<br /> ==> You need to change addrs to your TFRecords files and size parameters of images you created in previous steps.
<br /> ==> update size of the first fully connected layer 'fc_1'. Remember every max_pool layer makes tensor size twice less.
<br /> Because net uses two pooling layers with kernel=2 stride=2 and padding of convolutional layers = SAME
<br />==> you need to divide height and width of your input imagers by 4 times as long as you have two max_pool layers.
<br />If images have size of h = 100, w = 100 ==> fc_1 will have flat size 25x25x64, where 64 is number of filters
in the previous convolutional layer (can be changed too).

Note: model checkpoints are saved in the checkpoins directory for the further restore and predict.




Disclamer: big part of the code presented here is based on Google's proprietary scripts. Just combined and adapted for convinience. Use for you pleasure. Dont harm cats and dogs.
