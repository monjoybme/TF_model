# Simple utils and model for experimenting with your own images on Tensorflow (MNIST,CIFAR10 etc.)

Code for resizing images in a folder and split them into train/test directories for further TFrecord creation.

Model.py - a simple CNN model for working with TFrecods files containing images and label. To make model work on yor own images you need:

1. Resize_and_slit.py - resize the images to any height/width. The script uses as raw a directory with images in form ../image_class/images_in_class.jpg ==> script creates two separate folders /TRAIN_DIR/ and /VALIDATION_DIR/ ==> split resized imagaes randomly to 80% to TRAIN_DIR, 20% to VALIDATION_DIR.

2. Create TFrecords. Use build_image_data.py (provided by Google). Create labels.txt and string by string write classes there (example included). Change 'train_directory' value to your addrs to /TRAIN_DIR/ and 'validation_directory' to /VALIDATION_DIR/ ==> script creates two TFRecords files /output/train-00000-of-00001 and /output/validation-00000-of-00001

3. Use Model.py to train your net. You need to change addrs to your TFRecords files and size parameters of images. Also you need to update size of the first fully connected layer 'fc_1' to fit your size. Because net uses two pooling layers with kernel=2 stride=2 and padding of convolutional layers = SAME ==> you need to divide height and width by 4 times. If images have size of h = 100, w = 100 ==> fc_1 will have flat size 25x25x64, where 64 is number of filters in the previous convolutional layer (can be changed too).


