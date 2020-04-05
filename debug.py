# This file is here for prototyping purposes

from capsulenet import CapsNet, train
from data_generators import DataGen

cn = CapsNet(input_shape=(32,32,1))

data_gen = DataGen(
    batch_size=32, 
    data_dir="/home/hod/mag/data/OCT2017_preprocessed_128x128",
    target_size=(128,128),
    validation_split=0.1)
# train(network=cn, data_gen=data_gen, save_dir="/home/hod/mag/results/OCT2017_preprocessed_128x128", epochs=1)
# cn.test()