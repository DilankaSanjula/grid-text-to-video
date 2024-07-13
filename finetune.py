from tensorflow import keras
from stable_diffusion.stable_diffusion import StableDiffusion
from stable_diffusion.stable_diffusion import get_models
from PIL import Image

img_height = 512
img_width = 512

text_encoder, diffusion_model, decoder, encoder = get_models(img_height, img_width)
#train_dataset, 

def fine_tune(epochs, learning_rate):
        # Compile the models for training
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        text_encoder.compile(optimizer=optimizer, loss='mse')
        diffusion_model.compile(optimizer=optimizer, loss='mse')
        decoder.compile(optimizer=optimizer, loss='mse')
        encoder.compile(optimizer=optimizer, loss='mse')
        
        print("came here")
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            # for step, (images, captions) in enumerate(train_dataset):
            #     with tf.GradientTape(persistent=True) as tape:
            #         # Tokenize captions
            #         tokens = [self.tokenizer.encode(c) + [49407] * (MAX_TEXT_LEN - len(self.tokenizer.encode(c))) for c in captions]
            #         tokens = np.array(tokens).astype("int32")

            #         # Forward pass through the models
            #         encoded_text = self.text_encoder(tokens)
            #         latents = self.encoder(images)
            #         predicted_latents = self.diffusion_model([latents, encoded_text])
            #         decoded_images = self.decoder(predicted_latents)

            #         # Calculate loss
            #         loss = tf.reduce_mean(tf.square(images - decoded_images))

            #     # Apply gradients
            #     gradients = tape.gradient(loss, self.text_encoder.trainable_variables)
            #     optimizer.apply_gradients(zip(gradients, self.text_encoder.trainable_variables))

            #     gradients = tape.gradient(loss, self.diffusion_model.trainable_variables)
            #     optimizer.apply_gradients(zip(gradients, self.diffusion_model.trainable_variables))

            #     gradients = tape.gradient(loss, self.decoder.trainable_variables)
            #     optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

            #     gradients = tape.gradient(loss, self.encoder.trainable_variables)
            #     optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

            #     if step % 100 == 0:
            #         print(f"Step {step}: Loss = {loss.numpy()}")

            # print(f"Epoch {epoch + 1} completed. Loss = {loss.numpy()}")


fine_tune(epochs=1, learning_rate=1e-4)