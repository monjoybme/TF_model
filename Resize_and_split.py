# 2017 Pavel Krolevets @ Shanghai Jiao Tong University.
# ==============================================================================
"""
This code is the preparation of your own images to feed them into build_image_data.py to create TFrecords file.
First this script resize img, then split them randomly to 80% train folder and 20% to test folder. All of the raw folder
 structure preserved. Split images easy to use with build_image_data.py (provided by facenet dev.team) to create
 TFRecords files for feeding.
"""

import os
import sys
from random import shuffle
from pathlib import Path
import shutil
from PIL import Image

new_size = 160 # dimentions to resize raw imgs - means 200x200. Smaller images will be stretched.

imageFolder_raw = '/home/pavelkrolevets/Working/TF_facenet/data/raw/' # adr to raw imgs
imageFolder_train = '/home/pavelkrolevets/Working/TF_facenet/data/TRAIN_DIR/' # adr to train folder to store resized imgs
imageFolder_test = '/home/pavelkrolevets/Working/TF_facenet/data/VALIDATION_DIR/'# adr to test folder to store resized imgs

def resize(folder, fileName, Size):
    filePath = os.path.join(folder, fileName)
    im = Image.open(filePath)
    w, h  = im.size
    newIm = im.resize((Size, Size))
    # overrider orginal, you can save a copy
    newIm.save(filePath)

def bulkResize(Folder, factor):
    imgExts = ["png", "bmp", "jpg"]
    for path, dirs, files in os.walk(Folder):
        for fileName in files:
            ext = fileName[-3:].lower()
            if ext not in imgExts:
                continue

            resize(path, fileName, factor)

# split data to train and test

dir_list=[x[0] for x in os.walk(imageFolder_train)]
dir_list = dir_list[1:]
just_dirs = next(os.walk(imageFolder_raw))[1]
print(just_dirs)
thefile = open('/home/pavelkrolevets/Working/TF_facenet/data/labels.txt', 'w')
for item in just_dirs:
  thefile.write("%s\n" % item)

for i in just_dirs:
    f = []
    imgExts = ["png", "bmp", "jpg"]
    for path, dirs, files in os.walk(imageFolder_raw+i):
        for fileName in files:
            ext = fileName[-3:].lower()
            if ext not in imgExts:
                continue
            f.append(fileName)
            shuffle(f) # DONT FORGET TO SHUFFLE :-)
        train_files = f[0:int(0.8 * len(f))] # 80% to train
        test_files = f[int(0.8 * len(f)):] # the rest 20% to test
        print(train_files, '\n', test_files)

    # creating folders
    if not os.path.exists(imageFolder_train+i+'/'):
        os.makedirs(imageFolder_train+i+'/')
    if not os.path.exists(imageFolder_test+i+'/'):
        os.makedirs(imageFolder_test+i+'/')

    #copying files
    for file in train_files:
        print(imageFolder_raw + i + '/' + file)
        print(imageFolder_train+i+'/')
        shutil.copy2(imageFolder_raw+i+'/'+file, imageFolder_train+i+'/')
    for file in test_files:
        print(imageFolder_raw + i + '/' + file)
        print(imageFolder_test + i + '/')
        shutil.copy2(imageFolder_raw + i + '/' + file, imageFolder_test + i + '/')

# resize
bulkResize(imageFolder_train, new_size)
bulkResize(imageFolder_test, new_size)
