import dnnlib.tflib as tflib

import moviepy.editor

import numpy as np

import pickle
import argparse
import re
import os


def load_Gs(model_path):
    with open(model_path, "rb") as file:
        _, _, Gs = pickle.load(file)
        # G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
    return Gs


def main(model_path, seed=1000):
    # Initialize TensorFlow and load the model:
    tflib.init_tf()
    Gs = load_Gs(model_path)
    # Get the model name (for naming the video)
    model_name = re.search('([\w-]+).pkl', model_path).group(1)
    # Set the seed:
    rnd = np.random.RandomState(seed)
    # Generate 3 random latents of shape (1, 512):
    latents_a = rnd.randn(1, Gs.input_shape[1])
    latents_b = rnd.randn(1, Gs.input_shape[1])
    latents_c = rnd.randn(1, Gs.input_shape[1])

    def circ_generator(latents_interpolate):
        radius = 40.0
        latents_axis_x = (latents_a - latents_b).flatten() / np.linalg.norm(latents_a - latents_b)
        latents_axis_y = (latents_a - latents_c).flatten() / np.linalg.norm(latents_a - latents_c)
        latents_x = np.sin(np.pi * 2.0 * latents_interpolate) * radius
        latents_y = np.cos(np.pi * 2.0 * latents_interpolate) * radius
        latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
        return latents

    def mse(x, y):
        return (np.square(x - y)).mean()

    def generate_from_generator_adaptive(gen_func):
        max_step = 1.0
        current_pos = 0.0

        change_min = 10.0
        change_max = 11.0

        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

        current_latent = gen_func(current_pos)
        current_image = Gs.run(current_latent, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
        array_list = []

        video_length = 1.0
        while(current_pos < video_length):
            array_list.append(current_image)

            lower = current_pos
            upper = current_pos + max_step
            current_pos = (upper + lower) / 2.0

            current_latent = gen_func(current_pos)
            current_image = images = Gs.run(current_latent, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
            # Calculate the MSE with respect to the last image
            current_mse = mse(array_list[-1], current_image)
            # There is a range of min and max values that the MSE must lay in:
            while current_mse < change_min or current_mse > change_max:
                if current_mse < change_min:
                    lower = current_pos
                    current_pos = (upper + lower) / 2.0

                if current_mse > change_max:
                    upper = current_pos
                    current_pos = (upper + lower) / 2.0


                current_latent = gen_func(current_pos)
                current_image = images = Gs.run(current_latent, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
                current_mse = mse(array_list[-1], current_image)
            print(current_pos, current_mse)
        return array_list

    frames = generate_from_generator_adaptive(circ_generator)
    frames = moviepy.editor.ImageSequenceClip(frames, fps=30)

    # Make the save dir:
    save_path = './results/interpolation_videos/{}/'.format(model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Generate video.
    mp4_file = save_path + 'seed_{}-circular.mp4'.format(seed)
    mp4_codec = 'libx265'
    mp4_bitrate = '15M'
    mp4_fps = 30

    frames.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)

def parse():
    parser = argparse.ArgumentParser(
        description="Circular interpolation with respect to a point."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained model.",
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed.",
        default=1000
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    main(
        model_path=args.model_path,
        seed=args.seed
    )
