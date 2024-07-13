import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt
from stable_diffusion.stable_diffusion import get_models
from stable_diffusion.diffusion_model import UNetModel

img_height = 512
img_width = 512
MAX_TEXT_LEN = 77  # Define MAX_TEXT_LEN according to your dataset

text_encoder, diffusion_model, decoder, encoder = get_models(img_height, img_width)

# Print the summary of each model
print("Text Encoder Model Summary:")
text_encoder.summary()

print("\nDiffusion Model Summary:")
diffusion_model.summary()

print("\nDecoder Model Summary:")
decoder.summary()

print("\nEncoder Model Summary:")
encoder.summary()

