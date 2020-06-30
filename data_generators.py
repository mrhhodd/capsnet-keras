from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import numpy as np

class DataGen():
    def __init__(self, batch_size, data_dir, target_size, validation_split):
        data_gen = ImageDataGenerator(
            preprocessing_function=lambda input: (input - np.mean(input))/np.var(input),
            validation_split=validation_split
        )
        self.training_generator = data_gen.flow_from_directory(
            directory=data_dir, 
            target_size=target_size, 
            color_mode="grayscale", 
            batch_size=batch_size,
            subset="training"
            )
        self.validation_generator = data_gen.flow_from_directory(
            directory=data_dir, 
            target_size=target_size, 
            color_mode="grayscale", 
            batch_size=batch_size,
            subset="validation"
            )