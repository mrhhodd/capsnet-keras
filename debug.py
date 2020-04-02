# This file is here for prototyping purposes

# import os
# from random import shuffle
# import numpy as np
# from keras.preprocessing.image import load_img, img_to_array
# from keras.utils import to_categorical
# import math

# def load_data(rootdir):
#     class_map = {"NORMAL":0, "CNV":1, "DME":2, "DRUSEN":3}
#     data_sets = { "train": {},  "test": {},  "val": {} }
#     for data_set_name, data_set in data_sets.items():
#         f = []
#         for (dirpath, dirnames, filenames) in os.walk(os.path.join(rootdir, data_set_name)):
#             f.extend([os.path.join(dirpath, filename) for filename in filenames])
#         np.random.shuffle(f)
#         data_set['x'] = []
#         data_set['y'] = []
#         for file_path in f:
#             img_arr = img_to_array(load_img(file_path))
#             data_set['x'].append(img_arr.reshape(img_arr.shape).astype('float32') / 255)
#             class_no = class_map[os.path.basename(file_path).split("-")[0]]
#             data_set['y'].append(to_categorical(class_no, 4))

#     with open('out.txt', 'w') as f:
#         f.write(str(out))    
#         # f.write(str(data_sets))    


# def generate_new_database(rootdir, newdir, new_size_h, new_size_w, max_examples=None):
#     import cv2
#     os.makedirs(newdir, exist_ok=True)
#     class_map = {"NORMAL":0, "CNV":1, "DME":2, "DRUSEN":3}
#     data_sets = { "train": {},  "test": {},  "val": {} }
#     for d in ["train", "test", "val"]:
#         os.makedirs(os.path.join(newdir, d), exist_ok=True)
#     f = []
#     f_filtred = []
#     for (dirpath, dirnames, filenames) in os.walk(rootdir):
#         f.extend([os.path.join(dirpath, filename) for filename in filenames])
#     np.random.shuffle(f)
#     max = max_examples if max_examples else len(f)
#     test_no = int(0.2*max) if 0.2*max < 100 else 100
#     val_no = int(0.2*max) if 0.2*max < 100 else 100
#     i = 0
#     for file_path in f:
#         if i > max_examples:
#             break
#         image = cv2.imread(file_path)
#         if image.shape[0]/image.shape[1] > 1.1 or image.shape[0]/image.shape[1]<0.9:
#             continue
#         f_filtred.append(file_path)
#         i+=1
#     i = 0
#     np.random.shuffle(f_filtred)
#     for file_path in f_filtred:
#         if i > max_examples:
#             break
#         image = cv2.imread(file_path)
#         new_image = cv2.resize(image, (new_size_h, new_size_w))
#         if i<test_no:
#             data_set_name = "test"
#         elif i<test_no+val_no:
#             data_set_name = "val"
#         else:
#             data_set_name = "train"
#         cv2.imwrite(os.path.join(newdir, data_set_name, os.path.basename(file_path)), new_image)
#         i+=1
    
# def resize_images(rootdir, newdir, new_size_h, new_size_w, max_examples=None):
#     import cv2
#     os.makedirs(newdir, exist_ok=True)
#     class_map = {"NORMAL":0, "CNV":1, "DME":2, "DRUSEN":3}
#     data_sets = { "train": {},  "test": {},  "val": {} }
#     for data_set_name, data_set in data_sets.items():
#         f = []
#         for (dirpath, dirnames, filenames) in os.walk(os.path.join(rootdir, data_set_name)):
#             f.extend([os.path.join(dirpath, filename) for filename in filenames])
#         np.random.shuffle(f)
#         max = len(f) 
#         if max_examples and data_set_name=="train":
#             max = max_examples
#         os.makedirs(os.path.join(newdir, data_set_name), exist_ok=True)
#         i = 0
#         for file_path in f:
#             if max_examples and i == max:
#                 break
#             image = cv2.imread(file_path, 1)
#             # if image.shape[0] != image.shape[1]:
#             #     continue
#             new_image = cv2.resize(image, (new_size_h, new_size_w))
#             cv2.imwrite(os.path.join(newdir, data_set_name, os.path.basename(file_path)), new_image)
#             i+=1

# # load_data("/home/hod/mag/OCT2017_")
# load_data("/home/hod/mag/64x64_oct2017")
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/256x256_oct2017", 256, 256)
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/128x128__oct2017", 128, 128)
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/128x128__oct2017_1k_examples", 128, 128, 1000)
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/32x32_oct2017", 32, 32)
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/32x32_oct2017_10k_examples", 32, 32, 10000)
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/64x64_oct2017_10k_examples", 64, 64, 10000)
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/64x64_oct2017_4k_examples", 64, 64, 4000)
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/64x64_oct2017_1k_examples", 64, 64, 1000)
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/64x64_oct2017_1k_examples_same_dim", 64, 64, 1000)
# generate_new_database("/home/hod/mag/OCT2017_", "/home/hod/mag/64x64_oct2017_1.2k_examples_same_dim_mixed", 64, 64, 1200)
# resize_images("/home/hod/mag/OCT2017_", "/home/hod/mag/128x128_oct2017_10k_examples", 128, 128, 10000)

from capsulenet import CapsNet, train
from data_generators import DataGen

cn = CapsNet(input_shape=[128,128,1])

data_gen = DataGen(
    batch_size=2, 
    data_dir="/home/hod/mag/data/OCT2017_preprocessed_128x128", 
    target_size=(128,128), 
    color_mode="grayscale", 
    training_seed=123)
# train(network=cn, data_gen=data_gen, save_dir="/home/hod/mag/results/OCT2017_preprocessed_128x128", epochs=1)
# cn.test()