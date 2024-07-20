import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image
from stable_diffusion.stable_diffusion import StableDiffusion, get_models
from tqdm import tqdm


MAX_TEXT_LEN = 77
img_height = 512
img_width = 512

# Custom dataset loader
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_dataset(dataset_path):
    def parse_function(file_path):
        image = load_and_preprocess_image(file_path)
        caption = tf.strings.split(tf.strings.regex_replace(file_path, dataset_path + '/', ''), '.')[0]
        return image, caption

    dataset = tf.data.Dataset.list_files(dataset_path + '/*.jpg')
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(0, half) / half)
    args = np.outer(timesteps, freqs)
    embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    return tf.convert_to_tensor(embedding, dtype=tf.float32)


# Example usage
epochs = 1
learning_rate = 1e-5
batch_size = 4
num_steps = 50


dataset_path = '/content/drive/MyDrive/4x4_grid_images'
train_dataset = load_dataset(dataset_path, batch_size=batch_size)

trainer = StableDiffusion(img_height, img_width, jit_compile=False, download_weights=False)
trainer.fine_tune(epochs, learning_rate, train_dataset, batch_size, num_steps=100)