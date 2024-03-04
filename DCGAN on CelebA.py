from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

import os
import zipfile
import glob
import imageio





# Define the path to the zip file and the extraction directory
drive_zip_path = "/content/drive/MyDrive/archive_2.zip"
extract_path = "/content/celeb_dataset"

# Unzip the CelebA dataset
with zipfile.ZipFile(drive_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Check the contents of the unzipped directory
celeb_images = glob.glob(os.path.join(extract_path, "img_align_celeba", "img_align_celeba", "*"))

# Display the number of images in the CelebA dataset
print("Number of CelebA images:", len(celeb_images))

import tensorflow as tf
from PIL import Image
import glob
import numpy as np
import os
from tensorflow.keras import layers

# Define the paths
celeb_images = glob.glob("/content/celeb_dataset/img_align_celeba/img_align_celeba/*")
print("Number of CelebA images:", len(celeb_images))

# Batch size and buffer size
batch_size = 32  # Adjust this based on your available memory
BUFFER_SIZE = 10000

# Shuffle and create a TensorFlow dataset
np.random.shuffle(celeb_images)
celeb_images = celeb_images[:BUFFER_SIZE]

# Load and preprocess images in batches
celeb_images_dataset = []

for i in range(0, len(celeb_images), batch_size):
    batch = celeb_images[i:i + batch_size]
    batch_data = [np.array(Image.open(img).resize((32, 32))) for img in batch]
    celeb_images_dataset.append(np.array(batch_data))

celeb_images_np = np.concatenate(celeb_images_dataset)

# Normalize the images to the range [-1, 1]
celeb_images_np = (celeb_images_np - 127.5) / 127.5

# Set BUFFER_SIZE based on the number of CelebA images
BUFFER_SIZE = len(celeb_images_np)

# Set BATCH_SIZE
BATCH_SIZE = 32

# Check if BUFFER_SIZE is greater than zero
if BUFFER_SIZE <= 0:
    raise ValueError("BUFFER_SIZE must be greater than zero.")

# Batch and shuffle the data
celeb_dataset = tf.data.Dataset.from_tensor_slices(celeb_images_np).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)






# Generator model
import time
import matplotlib.pyplot as plt


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 512)))
    assert model.output_shape == (None, 4, 4, 512)

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 3)

    return model

# Discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Generator and Discriminator instances
generator = make_generator_model()
discriminator = make_discriminator_model()

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss function
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Checkpoint setup
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Training parameters
EPOCHS = 200
noise_dim = 100
num_examples_to_generate = 16

# Seed for generating fixed images for GIF
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go
        print(f'Time for epoch {epoch + 1} is {time.time()-start} sec')

        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    print(f'Time for epoch {epoch + 1} is {time.time()-start} sec')
    generate_and_save_images(generator, epochs, seed)

# Generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, :] + 1) / 2.0)  # Denormalize for display
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# Train the model
train(celeb_dataset, EPOCHS)

# Display a single image using the epoch number
def display_image(epoch_no):
    return Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

# Create an animated GIF
anim_file = 'dcgan_celeba.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Display the animated GIF
from IPython.display import HTML
HTML('<img src="dcgan_celeba.gif">')