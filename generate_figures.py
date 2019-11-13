# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Modified script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import re
import os
import pickle

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib
import config

import argparse
from urllib.parse import urlparse

# ----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.

url_ffhq = (
    "https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ"
)  # karras2019stylegan-ffhq-1024x1024.pkl
url_celebahq = (
    "https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf"
)  # karras2019stylegan-celebahq-1024x1024.pkl
url_bedrooms = (
    "https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF"
)  # karras2019stylegan-bedrooms-256x256.pkl
url_cars = (
    "https://drive.google.com/uc?id=1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3"
)  # karras2019stylegan-cars-512x384.pkl
url_cats = (
    "https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ"
)  # karras2019stylegan-cats-256x256.pkl

synthesis_kwargs = dict(
    output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
    minibatch_size=8,
    truncation_psi=0.7,
)

_Gs_cache = dict()


def load_Gs(path_or_url):
    # Check if the input is a URL:
    if bool(urlparse(path_or_url).netloc):
        if path_or_url not in _Gs_cache:
            # If so, we will store the model in the cache:
            with dnnlib.util.open_url(path_or_url, cache_dir=config.cache_dir) as f:
                _G, _D, Gs = pickle.load(f)
            _Gs_cache[path_or_url] = Gs
        return _Gs_cache[path_or_url]
    # Else, it's a path to the local file:
    else:
        with open(path_or_url, "rb") as file:
            _G, _D, Gs = pickle.load(file)
        return Gs


# ----------------------------------------------------------------------------
# Figures 2, 3, 10, 11, 12: Multi-resolution grid of uncurated result images.


def draw_uncurated_result_figure(png, Gs, cx, cy, cw, ch, rows, lods, seed):
    # Print the name of the png to know that it's being generated in the cmd:
    print(png)
    latents = np.random.RandomState(seed).randn(
        sum(rows * 2 ** lod for lod in lods), Gs.input_shape[1]
    )
    images = Gs.run(latents, None, **synthesis_kwargs)  # [seed, y, x, rgb]

    canvas = PIL.Image.new(
        "RGB", (sum(cw // 2 ** lod for lod in lods), ch * rows), "white"
    )
    image_iter = iter(list(images))
    for col, lod in enumerate(lods):
        for row in range(rows * 2 ** lod):
            image = PIL.Image.fromarray(next(image_iter), "RGB")
            image = image.crop((cx, cy, cx + cw, cy + ch))
            image = image.resize((cw // 2 ** lod, ch // 2 ** lod), PIL.Image.ANTIALIAS)
            canvas.paste(
                image, (sum(cw // 2 ** lod for lod in lods[:col]), row * ch // 2 ** lod)
            )
    canvas.save(png)


# ----------------------------------------------------------------------------
# Figure 3: Style mixing.


def draw_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    # Print the name of the png to know that it's being generated in the cmd:
    print(png)
    src_latents = np.stack(
        np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds
    )
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

    canvas = PIL.Image.new(
        "RGB", (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), "white"
    )
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, "RGB"), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, "RGB"), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(
            row_dlatents, randomize_noise=False, **synthesis_kwargs
        )
        for col, image in enumerate(list(row_images)):
            canvas.paste(
                PIL.Image.fromarray(image, "RGB"), ((col + 1) * w, (row + 1) * h)
            )
    canvas.save(png)


# ----------------------------------------------------------------------------
# Figure 4: Noise detail.


def draw_noise_detail_figure(png, Gs, w, h, num_samples, seeds):
    # Print the name of the png to know that it's being generated in the cmd:
    print(png)
    canvas = PIL.Image.new("RGB", (w * 3, h * len(seeds)), "white")
    for row, seed in enumerate(seeds):
        latents = np.stack(
            [np.random.RandomState(seed).randn(Gs.input_shape[1])] * num_samples
        )
        images = Gs.run(latents, None, **synthesis_kwargs)
        canvas.paste(PIL.Image.fromarray(images[0], "RGB"), (0, row * h))
        for i in range(4):
            crop = PIL.Image.fromarray(images[i + 1], "RGB")
            # The crop will be at the center of the image (change it if you wish
            # to see the stochastic noise in other areas of the generated images)
            crop = crop.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
            crop = crop.resize((w // 2, h // 2), PIL.Image.NEAREST)
            canvas.paste(crop, (w + (i % 2) * w // 2, row * h + (i // 2) * h // 2))
        diff = np.std(np.mean(images, axis=3), axis=0) * 4
        diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)
        canvas.paste(PIL.Image.fromarray(diff, "L"), (w * 2, row * h))
    canvas.save(png)


# ----------------------------------------------------------------------------
# Figure 5: Noise components.


def draw_noise_components_figure(png, Gs, w, h, seeds, noise_ranges, flips):
    # Print the name of the png to know that it's being generated in the cmd:
    print(png)
    Gsc = Gs.clone()
    noise_vars = [
        var
        for name, var in Gsc.components.synthesis.vars.items()
        if name.startswith("noise")
    ]
    noise_pairs = list(zip(noise_vars, tflib.run(noise_vars)))  # [(var, val), ...]
    latents = np.stack(
        np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds
    )
    all_images = []
    for noise_range in noise_ranges:
        tflib.set_vars(
            {
                var: val * (1 if i in noise_range else 0)
                for i, (var, val) in enumerate(noise_pairs)
            }
        )
        range_images = Gsc.run(latents, None, randomize_noise=False, **synthesis_kwargs)
        range_images[flips, :, :] = range_images[flips, :, ::-1]
        all_images.append(list(range_images))

    canvas = PIL.Image.new("RGB", (w * 2, h * 2), "white")
    for col, col_images in enumerate(zip(*all_images)):
        canvas.paste(
            PIL.Image.fromarray(col_images[0], "RGB").crop((0, 0, w // 2, h)),
            (col * w, 0),
        )
        canvas.paste(
            PIL.Image.fromarray(col_images[1], "RGB").crop((w // 2, 0, w, h)),
            (col * w + w // 2, 0),
        )
        canvas.paste(
            PIL.Image.fromarray(col_images[2], "RGB").crop((0, 0, w // 2, h)),
            (col * w, h),
        )
        canvas.paste(
            PIL.Image.fromarray(col_images[3], "RGB").crop((w // 2, 0, w, h)),
            (col * w + w // 2, h),
        )
    canvas.save(png)


# ----------------------------------------------------------------------------
# Figure 8: Truncation trick.


def draw_truncation_trick_figure(png, Gs, w, h, seeds, psis):
    # Print the name of the png to know that it's being generated in the cmd:
    print(png)
    latents = np.stack(
        np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds
    )
    dlatents = Gs.components.mapping.run(latents, None)  # [seed, layer, component]
    dlatent_avg = Gs.get_var("dlatent_avg")  # [component]

    canvas = PIL.Image.new("RGB", (w * len(psis), h * len(seeds)), "white")
    for row, dlatent in enumerate(list(dlatents)):
        row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(
            psis, [-1, 1, 1]
        ) + dlatent_avg
        row_images = Gs.components.synthesis.run(
            row_dlatents, randomize_noise=False, **synthesis_kwargs
        )
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, "RGB"), (col * w, row * h))
    canvas.save(png)


# ----------------------------------------------------------------------------
# Main program.


def main(
    model_path,
    size=512,
    uncurated=False,
    style_mixing=False,
    noise_detail=False,
    noise_components=False,
    trunc_trick=False,
    generate_all=False,
):
    # Initialize TensorFlow
    tflib.init_tf()
    # Get the Generator (stable version):
    Gs = load_Gs(model_path)
    # Get the file name of the model:
    model_name = re.search("([\w-]+).pkl", model_path).group(1)
    # Create the save dir:
    save_path = "./results/paper_images/{}/".format(model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # If desired, all the images will be generated:
    if generate_all:
        uncurated = style_mixing = noise_detail = noise_components = trunc_trick = True
    # Else, the user will define what specific figures to generate:
    if uncurated:
        save_name = save_path + "figure02-uncurated.png"
        # We assume the generated images are square, so there is no need to crop the black padding (as in the cars model):
        draw_uncurated_result_figure(
            save_name,
            Gs,
            cx=0,
            cy=0,
            cw=size,
            ch=size,
            rows=2,
            lods=[0, 1, 2, 2, 3, 3],
            seed=5,
        )
    if style_mixing:
        # Source B; modify these accordingly to your model
        src_seeds = [639, 701, 321369, 615, 2268]
        # Source A; ibidem
        dst_seeds = [888, 829, 1898, 147, 1614, 845]
        # Different types of style ranges
        # Coarse styles: range(0, 4), Middle styles: range(4, 8), Fine: range(8, n)
        # where n will be defined, depending on size (max 18). This is because
        # there are 2 AdaIN blocks per layer (2 in 4x4, 2 in 8x8, ...)
        n = int(2 * (np.log2(size) - 1))
        # We assume that size > 32, otherwise, why use StyleGAN?
        dic_styles = dict(
            all=[range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, n)], # 3 + 2 + 1 = 6 = len(dst_seeds)
            coarse=[range(0, 4)] * len(dst_seeds),
            middle=[range(4, 8)] * len(dst_seeds),
            fine=[range(8, n)] * len(dst_seeds),
        )
        for k in dic_styles.keys():
            save_name = save_path + "figure03-style-mixing-{}.png".format(k)
            draw_style_mixing_figure(
                save_name,
                Gs,
                w=size,
                h=size,
                src_seeds=src_seeds,
                dst_seeds=dst_seeds,
                style_ranges=dic_styles[k],
            )
    if noise_detail:
        save_name = save_path + "figure04-noise-detail.png"
        draw_noise_detail_figure(
            save_name, Gs, w=size, h=size, num_samples=100, seeds=[1157, 1012]
        )
    if noise_components:
        save_name = save_path + "figure05-noise-components.png"
        # Same as the style mixing figure, there are 2 sources of noise B
        # per layer in the Synthesis network, so we will add noise on all
        # (range(0, 18)), no noise (range(0, 0)), only in the fine layers
        # (range(8, 18)) and only on the coarse layers (range(0, 8))
        n = int(2 * (np.log2(size) - 1))
        # We assume size > 32, otherwise, why use StyleGAN?
        noise_ranges=[range(0, n), range(0, 0), range(8, n), range(0, 8)]
        draw_noise_components_figure(
            save_name,
            Gs,
            w=size,
            h=size,
            seeds=[1967, 1555],
            noise_ranges=noise_ranges,
            flips=[1],
        )
    if trunc_trick:
        save_name = save_path + "figure08-truncation-trick.png"
        draw_truncation_trick_figure(
            save_name,
            Gs,
            w=size,
            h=size,
            seeds=[91, 388, 1839, 47, 9],
            psis=[1, 0.7, 0.5, 0.2, 0, -0.2, -0.5, -0.7, -1],
        )


def parse():
    parser = argparse.ArgumentParser(
        description="Generate images with a locally trained pkl file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model.",
        required=True
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Size of each generated image.",
        required=True
    )
    parser.add_argument(
        "--uncurated",
        action="store_true",
        help="Generate Figure 2 from the paper.",
        default=False,
    )
    parser.add_argument(
        "--style_mixing",
        action="store_true",
        help="Generate Figure 3 from the paper.",
        default=False,
    )
    parser.add_argument(
        "--noise_detail",
        action="store_true",
        help="Generate Figure 4 from the paper.",
        default=False,
    )
    parser.add_argument(
        "--noise_components",
        action="store_true",
        help="Generate Figure 5 from the paper.",
        default=False,
    )
    parser.add_argument(
        "--trunc_trick",
        action="store_true",
        help="Generate Figure 8 from the paper.",
        default=False,
    )
    parser.add_argument(
        "--generate_all",
        action="store_true",
        help="Generate all the figures.",
        default=False,
    )
    args = parser.parse_args()
    return args


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse the arguments
    args = parse()
    # Run the main program:
    main(
        model_path=args.model_path,
        size=args.size,
        uncurated=args.uncurated,
        style_mixing=args.style_mixing,
        noise_detail=args.noise_detail,
        noise_components=args.noise_components,
        trunc_trick=args.trunc_trick,
        generate_all=args.generate_all,
    )

# ----------------------------------------------------------------------------
