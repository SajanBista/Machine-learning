import numpy as np
from keras.preprocessing import image
import os

# Validate file path
file_path = 'dataset/single_prediction/cat_or_dog_1.jpg'

if os.path.exists(file_path):
    test_image = image.load_img(file_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    print("Image loaded successfully!")
else:
    print(f"File not found at {file_path}")
