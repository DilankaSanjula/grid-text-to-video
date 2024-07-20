import os
import numpy as np
from tqdm import tqdm
import math
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from .autoencoder_kl import Decoder, Encoder
from .diffusion_model import UNetModel
from .clip_encoder import CLIPTextTransformer
from .clip_tokenizer import SimpleTokenizer
from .constants import _UNCONDITIONAL_TOKENS, _ALPHAS_CUMPROD, PYTORCH_CKPT_MAPPING
from PIL import Image

MAX_TEXT_LEN = 77
mixed_precision.set_global_policy('mixed_float16')

class StableDiffusion:
    def __init__(self, img_height=512, img_width=512, jit_compile=False, download_weights=True):
        self.img_height = img_height
        self.img_width = img_width
        self.tokenizer = SimpleTokenizer()

        text_encoder, diffusion_model, decoder, encoder = get_models(img_height, img_width, download_weights=download_weights)
        # text_encoder == Clip Model
        self.text_encoder = text_encoder
        self.diffusion_model = diffusion_model
        self.decoder = decoder
        self.encoder = encoder

        if jit_compile:
            self.text_encoder.compile(jit_compile=True)
            self.diffusion_model.compile(jit_compile=True)
            self.decoder.compile(jit_compile=True)
            self.encoder.compile(jit_compile=True)

        self.dtype = tf.float16
        #self.dtype = tf.float32
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.dtype = tf.float16

    def generate(
        self,
        prompt,
        negative_prompt=None,
        batch_size=1,
        num_steps=50,
        unconditional_guidance_scale=7.5,
        temperature=1,
        seed=None,
        input_image=None,
        input_mask=None,
        input_image_strength=0.5,
    ):  
        print('\n')
        print('\n')
        print("------------------------------------------------------------------------")
        # Tokenize prompt - Uses SimpleTokenizer
        input_tokens = self.tokenizer.encode(prompt)
        print("Tokenizing....")
        print(f"tokenized_prompt_shape : {len(input_tokens)}")
        print("------------------------------------------------------------------------")

        # Checks whether the prompt text length is too long
        assert len(input_tokens) < 77, "Prompt is too long (should be < 77 tokens)"

        # Fills tokenized prompt list to 77, convert to numpy array, repeat to use batch
        phrase = input_tokens + [49407] * (77 - len(input_tokens))
        phrase = np.array(phrase)[None].astype("int32")
        phrase = np.repeat(phrase, batch_size, axis=0)
        print("Filling Tokens to match token length (77)....")
        print(f"tokenized_phrase_shape : {phrase.shape}")
        print("------------------------------------------------------------------------")

        # Encode prompt tokens (and their positions) into a "context vector"
        pos_ids = np.array(list(range(77)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis=0)

        # Predict using CLIP Model - context
        context = self.text_encoder.predict_on_batch([phrase, pos_ids])
        print("CLIP model context prediction...")
        print(f"tokenized_context_shape : {context.shape}")
        print("------------------------------------------------------------------------")
 
        unconditional_tokens = _UNCONDITIONAL_TOKENS
        unconditional_tokens = np.array(unconditional_tokens)[None].astype("int32")
        unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)

        # Predict using CLIP Model - unconditional context
        unconditional_context = self.text_encoder.predict_on_batch(
            [unconditional_tokens, pos_ids]
        )
        print("CLIP model uncoditionall context prediction...")
        print(f"tokenized_unconditional_context_shape : {unconditional_tokens.shape}")
        print("------------------------------------------------------------------------")

        timesteps = np.arange(1, 1000, 1000 // num_steps)
        input_img_noise_t = timesteps[ int(len(timesteps)*input_image_strength) ]

        input_image_tensor = None

        latent, alphas, alphas_prev = self.get_starting_parameters(
            timesteps, batch_size, seed , input_image=input_image_tensor, input_img_noise_t=input_img_noise_t
        )
        print("Getting starting params for diffusion process...")
        starting_params = {"latent_shape": latent.shape, "alphas": alphas, "alpha_prev": alphas_prev}
        print(f"Starting params : {starting_params}")
        print("------------------------------------------------------------------------")

        if input_image is not None:
            timesteps = timesteps[: int(len(timesteps)*input_image_strength)]

        print("Diffusion model...")
        
        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar:
            progbar.set_description(f"{index:3d} {timestep:3d}")
            e_t = self.get_model_output(
                latent,
                timestep,
                context,
                unconditional_context,
                unconditional_guidance_scale,
                batch_size,
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            latent, pred_x0 = self.get_x_prev_and_pred_x0(
                latent, e_t, index, a_t, a_prev, temperature, seed
            )

        print(f"latent_shape: {latent.shape}")
        print("------------------------------------------------------------------------")

        print("Decoding...")
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        print("------------------------------------------------------------------------")

        return np.clip(decoded, 0, 255).astype("uint8")

    def timestep_embedding(self, timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1),dtype=self.dtype)

    def add_noise(self, x , t , noise=None ):
        batch_size,w,h = x.shape[0] , x.shape[1] , x.shape[2]
        if noise is None:
            noise = tf.random.normal((batch_size,w,h,4), dtype=self.dtype)
        sqrt_alpha_prod = _ALPHAS_CUMPROD[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - _ALPHAS_CUMPROD[t]) ** 0.5

        return  sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

    def get_starting_parameters(self, timesteps, batch_size, seed,  input_image=None, input_img_noise_t=None):
        n_h = self.img_height // 8
        n_w = self.img_width // 8
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        if input_image is None:
            latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
        else:
            latent = self.encoder(input_image)
            latent = tf.repeat(latent , batch_size , axis=0)
            latent = self.add_noise(latent, input_img_noise_t)
        return latent, alphas, alphas_prev

    def get_model_output(
        self,
        latent,
        t,
        context,
        unconditional_context,
        unconditional_guidance_scale,
        batch_size,
    ):
        timesteps = np.array([t])
        t_emb = self.timestep_embedding(timesteps)
        t_emb = np.repeat(t_emb, batch_size, axis=0)
        unconditional_latent = self.diffusion_model.predict_on_batch(
            [latent, t_emb, unconditional_context]
        )
        latent = self.diffusion_model.predict_on_batch([latent, t_emb, context])
        return unconditional_latent + unconditional_guidance_scale * (
            latent - unconditional_latent
        )

    def get_x_prev_and_pred_x0(self, x, e_t, index, a_t, a_prev, temperature, seed):
        sigma_t = 0
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
        noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0

    def load_weights_from_pytorch_ckpt(self , pytorch_ckpt_path):
        import torch
        pt_weights = torch.load(pytorch_ckpt_path, map_location="cpu")
        for module_name in ['text_encoder', 'diffusion_model', 'decoder', 'encoder' ]:
            module_weights = []
            for i , (key , perm ) in enumerate(PYTORCH_CKPT_MAPPING[module_name]):
                w = pt_weights['state_dict'][key].numpy()
                if perm is not None:
                    w = np.transpose(w , perm )
                module_weights.append(w)
            getattr(self, module_name).set_weights(module_weights)
            print("Loaded %d weights for %s"%(len(module_weights) , module_name))


    def fine_tune(self, epochs, learning_rate, train_dataset, batch_size=1, num_steps=50, accumulation_steps=8):
        print("batch_size", batch_size)
        # Set mixed precision policy
        mixed_precision.set_global_policy('mixed_float16')

        # Freeze the text encoder, diffusion model, and encoder
        self.text_encoder.trainable = False
        self.diffusion_model.trainable = False
        self.encoder.trainable = False
        self.decoder.trainable = False

        self.decoder.summary()

        optimizer = mixed_precision.LossScaleOptimizer(
            keras.optimizers.Adam(learning_rate=learning_rate),
            dynamic=True
        )
        mse_loss = keras.losses.MeanSquaredError()

        print("Starting fine-tuning")

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            accumulated_loss = 0
            accumulated_gradients = None

            for step, (images, captions) in enumerate(train_dataset.batch(batch_size)):
                print(step)
                # Ensure captions are strings
                captions = [caption.numpy().decode('utf-8') if isinstance(caption.numpy(), bytes) else str(caption.numpy()) for caption in captions]

                # Tokenize and pad the captions
                input_tokens = [self.tokenizer.encode(caption) for caption in captions]
                input_tokens = keras.preprocessing.sequence.pad_sequences(input_tokens, maxlen=MAX_TEXT_LEN, padding='post')
                input_tokens = np.array(input_tokens)

                pos_ids = np.array(list(range(MAX_TEXT_LEN)))[None].astype("int32")
                pos_ids = np.repeat(pos_ids, batch_size, axis=0)

                # Perform inference outside the gradient tape
                context = self.text_encoder.predict([input_tokens, pos_ids])

                # Ensure the shape is compatible before reshaping
                expected_shape = (batch_size, self.img_height, self.img_width, 3)
                if images.shape[1:] != expected_shape[1:]:
                    raise ValueError(f"Unexpected image shape {images.shape}, expected {expected_shape}")

                # Reshape images to match the expected input shape of the encoder
                images = tf.reshape(images, expected_shape)
                print(f"Images shape after reshaping: {images.shape}")

                # Perform inference outside the gradient tape
                start_time = time.time()
                latent = self.encoder.predict(images)
                end_time = time.time()
                print(f"Encoder Inference Time: {end_time - start_time} seconds")

                # Generate the timestep embeddings (t_emb) for the current step
                timesteps = np.arange(1, 1000, 1000 // num_steps)
                input_img_noise_t = timesteps[int(len(timesteps) * 0.5)]
                latent, alphas, alphas_prev = self.get_starting_parameters(timesteps, batch_size, None, input_image=None, input_img_noise_t=input_img_noise_t)

                start_time = time.time()
                # Diffusion process (multiple steps)
                for index, timestep in enumerate(timesteps[::-1]):
                    print("Diffusion steps", index)
                    t_emb = self.timestep_embedding(np.array([timestep]))
                    t_emb = np.repeat(t_emb, batch_size, axis=0)

                    unconditional_latent = self.diffusion_model.predict([latent, t_emb, context])
                    latent = self.diffusion_model.predict([latent, t_emb, context])
                    e_t = unconditional_latent + 1.0 * (latent - unconditional_latent)
                    a_t, a_prev = alphas[index], alphas_prev[index]
                    latent, pred_x0 = self.get_x_prev_and_pred_x0(latent, e_t, index, a_t, a_prev, 1.0, None)

                end_time = time.time()
                print(f"Diffusion Inference Time: {end_time - start_time} seconds")

                outputs = self.decoder.predict(latent)
                loss = mse_loss(images, outputs) / accumulation_steps  # Scale loss

                print(loss)

                # Use tf.stop_gradient to prevent gradients from being calculated
                loss = tf.stop_gradient(loss)

                if step % 500 == 0:
                    os.makedirs('models', exist_ok=True)
                    encoder_save_path = os.path.join('models', f'encoder_epoch_{epoch + 1}_step_{step + 1}.h5')
                    decoder_save_path = os.path.join('models', f'decoder_epoch_{epoch + 1}_step_{step + 1}.h5')

                    print(f"Step {step}, Loss: {tf.reduce_mean(loss).numpy()}")

                    self.encoder.save(encoder_save_path)
                    self.decoder.save(decoder_save_path)

        # Save the encoder and decoder models at the end of each epoch
        os.makedirs('models', exist_ok=True)
        encoder_save_path = os.path.join('models', f'encoder_epoch_{epoch + 1}.h5')
        decoder_save_path = os.path.join('models', f'decoder_epoch_{epoch + 1}.h5')

        self.encoder.save(encoder_save_path)
        self.decoder.save(decoder_save_path)
        print(f"Models saved: {encoder_save_path}, {decoder_save_path}")

        print("Fine-tuning complete")



def get_models(img_height, img_width, download_weights=True):
    n_h = img_height // 8
    n_w = img_width // 8

    # Create text encoder
    input_word_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype="int32")
    input_pos_ids = keras.layers.Input(shape=(MAX_TEXT_LEN,), dtype="int32")
    embeds = CLIPTextTransformer()([input_word_ids, input_pos_ids])
    text_encoder = keras.models.Model([input_word_ids, input_pos_ids], embeds)

    # Creation diffusion UNet
    context = keras.layers.Input((MAX_TEXT_LEN, 768))
    t_emb = keras.layers.Input((320,))
    latent = keras.layers.Input((n_h, n_w, 4))
    unet = UNetModel()
    diffusion_model = keras.models.Model(
        [latent, t_emb, context], unet([latent, t_emb, context])
    )

    # Create decoder
    latent = keras.layers.Input((n_h, n_w, 4))
    decoder = Decoder()
    decoder = keras.models.Model(latent, decoder(latent))

    inp_img = keras.layers.Input((img_height, img_width, 3))
    encoder = Encoder()
    encoder = keras.models.Model(inp_img, encoder(inp_img))

    if download_weights:
        text_encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/text_encoder.h5",
            file_hash="d7805118aeb156fc1d39e38a9a082b05501e2af8c8fbdc1753c9cb85212d6619",
        )
        diffusion_model_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
            file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
        )
        decoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
            file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
        )

        encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/divamgupta/stable-diffusion-tensorflow/resolve/main/encoder_newW.h5",
            file_hash="56a2578423c640746c5e90c0a789b9b11481f47497f817e65b44a1a5538af754",
        )

        text_encoder.load_weights(text_encoder_weights_fpath)
        diffusion_model.load_weights(diffusion_model_weights_fpath)
        decoder.load_weights(decoder_weights_fpath)
        encoder.load_weights(encoder_weights_fpath)


    else:
        # Load weights from local directory
        if os.path.exists('models'):
            local_weights_dir = 'models'

        if os.path.exists('/content/drive/MyDrive/models'):
            local_weights_dir = '/content/drive/MyDrive/models'
            
        text_encoder_weights_fpath = os.path.join(local_weights_dir, 'text_encoder.h5')
        diffusion_model_weights_fpath = os.path.join(local_weights_dir, 'diffusion_model.h5')
        decoder_weights_fpath = os.path.join(local_weights_dir, 'decoder.h5')
        encoder_weights_fpath = os.path.join(local_weights_dir, 'encoder_newW.h5')

        # Ensure that all weight files exist in the local directory
        for weight_file in [text_encoder_weights_fpath, diffusion_model_weights_fpath, decoder_weights_fpath, encoder_weights_fpath]:
            if not os.path.exists(weight_file):
                raise FileNotFoundError(f"Weight file {weight_file} not found in {local_weights_dir}")
    return text_encoder, diffusion_model, decoder , encoder
