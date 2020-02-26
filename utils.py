import numpy as np
from matplotlib import pyplot as plt
import pandas
import os

def load_data(rootdir, balance_data=False):
    data_sets = { "train": [],  "test": [],  "val": [] }

    if balance_data:
        classes = ["NORMAL", "CNV", "DME", "DRUSEN"]
        # test_val_size = 1000
        test_val_size = 12
        all_files = {}
        for class_name in classes:
            all_files[class_name] = []
        for (dirpath, _, filenames) in os.walk(rootdir):
            for class_name in classes:
                all_files[class_name].extend([os.path.join(dirpath, filename) for filename in filenames if class_name in filename])
        # min_count = min([len(data_set) for data_set_name, data_set in all_files.items()])
        min_count = 124
        val_count = int(test_val_size/len(classes))
        test_count = int(test_val_size/len(classes))
        train_count = min_count - val_count - test_count
        for class_name, files in all_files.items():
            np.random.shuffle(files)
            data_sets["val"].extend(files[:val_count])
            data_sets["test"].extend(files[val_count:val_count+test_count])
            data_sets["train"].extend(files[val_count+test_count:val_count+test_count+train_count])

    else:
        for data_set_name, data_set in data_sets.items():
            files = []
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(rootdir, data_set_name)):
                files.extend([os.path.join(dirpath, filename) for filename in filenames])
            np.random.shuffle(files)
            data_set.extend(files)

    return data_sets


def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()