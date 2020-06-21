from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

class DataGen():
    def __init__(self, batch_size, data_dir, target_size, validation_split):
        data_gen = ImageDataGenerator(
            preprocessing_function=lambda input: (input - 127.0)/128.0,
            validation_split=validation_split
        )
        self.training_generator = data_gen.flow_from_directory(
            directory=data_dir, 
            target_size=target_size, 
            color_mode="grayscale", 
            batch_size=batch_size,
            subset="training",
            save_to_dir=data_dir,
            save_prefix="testing",
            save_format="jpg",
            )
        self.validation_generator = data_gen.flow_from_directory(
            directory=data_dir, 
            target_size=target_size, 
            color_mode="grayscale", 
            batch_size=batch_size,
            subset="validation"
            )