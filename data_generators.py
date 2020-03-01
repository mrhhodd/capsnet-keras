from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

class DataGen():
    def __init__(self, batch_size, data_dir, target_size, color_mode, training_seed):
        data_gen = ImageDataGenerator()
        self.training_generator = data_gen.flow_from_directory(
            directory=Path(data_dir, "train"), 
            target_size=target_size, 
            color_mode=color_mode, 
            batch_size=batch_size, 
            seed=training_seed
            )
        self.validation_generator = data_gen.flow_from_directory(
            directory=Path(data_dir, "val"),
            target_size=target_size, 
            color_mode=color_mode, 
            batch_size=batch_size, 
            seed=training_seed
            )
        self.test_generator = data_gen.flow_from_directory(
            directory=Path(data_dir, "test"), 
            target_size=target_size, 
            color_mode=color_mode, 
            batch_size=batch_size, 
            seed=training_seed
            )