#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import shutil

# process training folders
train_filenames = os.listdir('dataset/train')
train_cat = filter(lambda x:x[:3] == 'cat', train_filenames)
train_dog = filter(lambda x:x[:3] == 'dog', train_filenames)

def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
        print('delete: ', dirname)
    os.mkdir(dirname)
    print('make dir: ', dirname)

os.mkdir('dataset/train/cat')
os.mkdir('dataset/train/dog')

for filename in train_cat:
    shutil.move('dataset/train/'+filename, 'dataset/train/cat/')

for filename in train_dog:
    shutil.move('dataset/train/'+filename, 'dataset/train/dog/')

# process testing folder
test_filenames = os.listdir('dataset/test')
os.mkdir('dataset/test/test')
for filename in test_filenames:
    shutil.move('dataset/test/'+filename, 'dataset/test/test')
