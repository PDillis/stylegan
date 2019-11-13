# From Cyril Diagne: http://cyrildiagne.com/, with my modifications:

import os
import pickle
import argparse
from argparse import RawTextHelpFormatter
import re

import numpy as np
import scipy

import PIL.Image
import moviepy.editor

import dnnlib
import dnnlib.tflib as tflib
import config


def load_Gs(model_path):
    with open(model_path, "rb") as file:
        _G, _D, Gs = pickle.load(file)
        # G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
    return Gs


def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(
        list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype
    )
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid


def generate_interpolation_video(
    save_path,
    Gs,
    cols,
    rows,
    image_shrink=1,
    image_zoom=1,
    duration_sec=30.0,
    smoothing_sec=3.0,
    mp4_fps=30,
    mp4_codec="libx264",
    mp4_bitrate="5M",
    seed=1000,
    minibatch_size=8,
):

    # Save the video as ./results/interpolation_videos/model_name/seed-#-slerp.mp4:
    mp4 = save_path + "seed_{}-slerp.mp4".format(seed)
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(seed)

    print("Generating latent vectors...")
    grid_size = [cols, rows]
    # [frame, image, channel, component]:
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(
        all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode="wrap"
    )
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        # Get the images (with labels = None)
        images = Gs.run(
            latents,
            None,
            minibatch_size=minibatch_size,
            num_gpus=1,
            out_mul=127.5,
            out_add=127.5,
            out_shrink=image_shrink,
            out_dtype=np.uint8,
            truncation_psi=0.7,
            randomize_noise=False,
        )
        grid = create_image_grid(images, grid_size).transpose(1, 2, 0)  # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)  # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor  # pip install moviepy

    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.write_videofile(mp4, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)


def generate_style_transfer_video(
    save_path,
    mp4_file,
    size,
    Gs,
    style_ranges,
    dst_seeds=[700, 198],  # Source A in Figure 3
    image_shrink=1,
    image_zoom=1,
    duration_sec=30.0,
    smoothing_sec=3.0,
    mp4_fps=30,
    mp4_codec="libx264",
    mp4_bitrate="5M",
    seed=1000,
    minibatch_size=8,
):
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(seed)

    width = size
    height = size

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    synthesis_kwargs = dict(
        output_transform=fmt, truncation_psi=0.7, minibatch_size=minibatch_size
    )

    shape = [num_frames] + Gs.input_shape[1:]  # [frame, image, channel, component]
    src_latents = random_state.randn(*shape).astype(np.float32)  # Source B in Figure 3
    src_latents = scipy.ndimage.gaussian_filter(
        src_latents, smoothing_sec * mp4_fps, mode="wrap"
    )
    src_latents /= np.sqrt(np.mean(np.square(src_latents)))

    dst_latents = np.stack(
        np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds
    )

    src_dlatents = Gs.components.mapping.run(
        src_latents, None
    )  # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(
        dst_latents, None
    )  # [seed, layer, component]
    src_images = Gs.components.synthesis.run(
        src_dlatents, randomize_noise=False, **synthesis_kwargs
    )
    dst_images = Gs.components.synthesis.run(
        dst_dlatents, randomize_noise=False, **synthesis_kwargs
    )

    canvas = PIL.Image.new("RGB", (width * (len(dst_seeds) + 1), height * 2), "white")

    for col, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, "RGB"), ((col + 1) * height, 0))

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        src_image = src_images[frame_idx]
        canvas.paste(PIL.Image.fromarray(src_image, "RGB"), (0, height))

        for col, dst_image in enumerate(list(dst_images)):
            col_dlatents = np.stack([dst_dlatents[col]])
            col_dlatents[:, style_ranges[col]] = src_dlatents[
                frame_idx, style_ranges[col]
            ]
            col_images = Gs.components.synthesis.run(
                col_dlatents, randomize_noise=False, **synthesis_kwargs
            )
            for row, image in enumerate(list(col_images)):
                canvas.paste(
                    PIL.Image.fromarray(image, "RGB"),
                    ((col + 1) * height, (row + 1) * width),
                )
        return np.array(canvas)

    # Generate video.
    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    video_clip.write_videofile(
        mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate
    )

# From Snowy Halcy: https://github.com/halcy/stylegan/blob/master/Stylegan-Generate-Encode.ipynb
# with my modifications:
def generate_circular_interpolation_video(
    save_path,
    Gs,
    mp4_file,
    mp4_fps=30,
    mp4_codec="libx264",
    mp4_bitrate="5M",
    seed=1000,
):
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
    frames = moviepy.editor.ImageSequenceClip(frames, fps=mp4_fps)
    frames.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)


def main(
    model_path,
    seed,
    size,
    dst_seeds=42,
    cols=3,
    rows=2,
    duration=30.0,
    random=False,
    coarse=False,
    middle=False,
    fine=False,
    circular=False,
    generate_all=False,
):

    tflib.init_tf()

    # Load the Generator (stable version):
    print("Loading network from {}...".format(model_path))
    Gs = load_Gs(model_path=model_path)

    # Let's get the model name (without the .pkl):
    model_name = re.search("([\w-]+).pkl", model_path).group(1)
    # By default, the video will be saved in the ./stylegan/results/ subfolder
    save_path = "./{}/interpolation_videos/{}/seed_{}/".format(config.result_dir,
                                                          model_name,
                                                          seed)

    # If save_path doesn't exist, create it:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Different types of style ranges
    # Coarse styles: range(0, 4), Middle styles: range(4, 8), Fine: range(8, n)
    # where n will be defined, depending on size (max 18). This is because
    # there are 2 AdaIN blocks per layer (2 in 4x4, 2 in 8x8, ...)
    n = int(2 * (np.log2(size) - 1))
    # Depending on the input by the user:
    if generate_all:
        random = coarse = middle = fine = circular = True
    if random:
        generate_interpolation_video(
            save_path=save_path,
            Gs=Gs,
            cols=cols,
            rows=rows,
            duration_sec=duration,
            image_shrink=1,
            image_zoom=1,
            smoothing_sec=3.0,
            mp4_fps=30,
            mp4_codec="libx264",
            mp4_bitrate="5M",
            seed=seed,
            minibatch_size=8,
        )
    if coarse:
        style_ranges = [range(0, 4)] * len(dst_seeds)
        mp4_file = save_path + "seed_{}-style_mixing_coarse.mp4".format(seed)

        generate_style_transfer_video(
            save_path=save_path,
            mp4_file=mp4_file,
            size=size,
            Gs=Gs,
            style_ranges=style_ranges,
            dst_seeds=dst_seeds,
            image_shrink=1,
            image_zoom=1,
            duration_sec=duration,
            smoothing_sec=3.0,
            mp4_fps=30,
            mp4_codec="libx264",
            mp4_bitrate="5M",
            seed=seed,
            minibatch_size=8,
        )
    if middle:
        style_ranges = [range(4, 8)] * len(dst_seeds)
        mp4_file = save_path + "seed_{}-style_mixing_middle.mp4".format(seed)

        generate_style_transfer_video(
            save_path=save_path,
            mp4_file=mp4_file,
            size=size,
            Gs=Gs,
            style_ranges=style_ranges,
            dst_seeds=dst_seeds,
            image_shrink=1,
            image_zoom=1,
            duration_sec=duration,
            smoothing_sec=3.0,
            mp4_fps=30,
            mp4_codec="libx264",
            mp4_bitrate="5M",
            seed=seed,
            minibatch_size=8,
        )
    if fine:
        style_ranges = [range(8, n)] * len(dst_seeds)
        mp4_file = save_path + "seed_{}-style_mixing_fine.mp4".format(seed)

        generate_style_transfer_video(
            save_path=save_path,
            mp4_file=mp4_file,
            size=size,
            Gs=Gs,
            style_ranges=style_ranges,
            dst_seeds=dst_seeds,
            image_shrink=1,
            image_zoom=1,
            duration_sec=duration,
            smoothing_sec=3.0,
            mp4_fps=30,
            mp4_codec="libx264",
            mp4_bitrate="5M",
            seed=seed,
            minibatch_size=8,
        )
    if circular:
        mp4_file = save_path + 'seed_{}-circular.mp4'.format(seed)
        print("Generating circular interpolation video! Frames are being generated (please be patient):\n")
        generate_circular_interpolation_video(
            save_path=save_path,
            Gs=Gs,
            mp4_file=mp4_file,
            mp4_fps=30,
            mp4_codec="libx264",
            mp4_bitrate="5M",
            seed=1000,
        )


def parse():
    parser = argparse.ArgumentParser(
        description="Interpolation videos with a trained StyleGAN. Three types of videos can be generated:\n\
\t *Interpolation between random latent vectors (--random) {--rows --cols} \n\
\t *Style transfer interpolation videos (--coarse --middle --fine) {--dst_seeds} \n\
\t *Circular interpolation video (--circular)\n\
You can generate all the videos by simply adding the --generate_all flag. Have fun!",
        formatter_class=RawTextHelpFormatter
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
    parser.add_argument(
        "--dst_seeds",
        nargs="+",
        type=int,
        help="Source A seeds. These will be fixed. Input like so: --dst_seed 42 56 1...",
        default=42
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Size of generated images.",
        required=True
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="Columns in the random interpolation video (optional).",
        default=3,
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Rows in the random interpolation video (optional).",
        default=2,
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Add flag if you wish to generate the random interpolation video.",
        default=False,
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Add flag if you wish to generate the coarse styles interpolation video.",
        default=False,
    )
    parser.add_argument(
        "--middle",
        action="store_true",
        help="Add flag if you wish to generate the middle styles interpolation video.",
        default=False,
    )
    parser.add_argument(
        "--fine",
        action="store_true",
        help="Add flag if you wish to generate the fine styles interpolation video.",
        default=False,
    )
    parser.add_argument(
        "--circular",
        action="store_true",
        help="Add flag if you wish to generate the circular interpolation video.",
        default=False
    )
    parser.add_argument(
        "--generate_all",
        action="store_true",
        help="Add flag if you wish to generate all the interpolation videos.",
        default=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    main(
        model_path=args.model_path,
        seed=args.seed,
        dst_seeds=args.dst_seeds,
        size=args.size,
        random=args.random,
        cols=args.cols,
        rows=args.rows,
        coarse=args.coarse,
        middle=args.middle,
        fine=args.fine,
        circular=args.circular,
        generate_all=args.generate_all,
    )
