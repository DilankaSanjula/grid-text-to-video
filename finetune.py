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
        prefix_length = len(dataset_path) + 1  # +1 to remove the trailing slash
        caption = tf.strings.substr(file_path, prefix_length, -1)  # Remove the prefix
        caption = tf.strings.split(caption, '.')[0]  # Remove file extension
        caption = tf.strings.regex_replace(caption, '_', ' ')  # Replace underscores with spaces
        return image, caption

    # List files in sorted order
    file_paths = tf.io.gfile.glob(dataset_path + '/*.jpg')
    file_paths = sorted(file_paths)

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
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
batch_size = 1
num_steps = 50



dataset_path = 'webvid10m_dataset/4x4_grid_images'
pickle_file_path = 'model_cache/text_encoder_cache.pkl'

os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)

train_dataset = load_dataset(dataset_path)

for image, caption in train_dataset.take(17):
    print("Image shape:", image.numpy().shape)
    print("Caption:", caption.numpy().decode('utf-8'))

trainer = StableDiffusion(img_height, img_width, jit_compile=False, download_weights=False)
trainer.fine_tune(epochs, learning_rate, train_dataset, batch_size, num_steps=5)