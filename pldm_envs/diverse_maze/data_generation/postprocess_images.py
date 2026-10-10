import argparse
import concurrent.futures
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pldm_envs.diverse_maze.transforms import select_transforms


def load_image(transform, expected_shape, image_path):
    """Load and transform one PNG, failing with its exact path on corruption."""
    try:
        with Image.open(image_path) as image:
            image.load()
            array = np.asarray(transform(image), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(f"Unable to read rendered image: {image_path}") from exc

    if array.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected image shape for {image_path}: "
            f"expected {expected_shape}, got {array.shape}"
        )
    return array


def save_progress(progress_path, next_episode):
    """Atomically record the first episode that has not been converted yet."""
    temporary_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    temporary_path.write_text(f"{next_episode}\n", encoding="utf-8")
    os.replace(temporary_path, progress_path)


def remove_output_files(paths):
    for path in paths:
        if path.exists():
            path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Combine rendered PNG observations into a memory-mapped NumPy array"
    )
    parser.add_argument("--data_path", required=True, help="Dataset directory")
    parser.add_argument(
        "--save_path",
        default=None,
        help="Output directory (defaults to --data_path)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Number of PNG decoding threads",
    )
    parser.add_argument(
        "--batch_episodes",
        type=int,
        default=32,
        help="Number of complete episodes decoded before each output write",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing final or partial images.npy and rebuild it",
    )
    parser.add_argument(
        "--quick_debug",
        action="store_true",
        help="Convert at most three episodes into a separate debug output",
    )
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num_workers must be a positive integer")
    if args.batch_episodes < 1:
        parser.error("--batch_episodes must be a positive integer")

    data_path = Path(args.data_path).resolve()
    save_path = Path(args.save_path).resolve() if args.save_path else data_path
    save_path.mkdir(parents=True, exist_ok=True)

    metadata_path = data_path / "metadata.pt"
    proprio_path = data_path / "data.p"
    input_image_path = data_path / "images"
    for required_path in (metadata_path, proprio_path, input_image_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required dataset path is missing: {required_path}")

    config = torch.load(metadata_path, weights_only=False)
    splits = torch.load(proprio_path, weights_only=False)
    episode_lengths = np.fromiter(
        (len(split["observations"]) for split in splits),
        dtype=np.int64,
        count=len(splits),
    )
    del splits

    total_episodes = len(episode_lengths)
    if total_episodes == 0:
        raise RuntimeError(f"Dataset contains no episodes: {proprio_path}")

    if args.quick_debug:
        total_episodes = min(total_episodes, 3)
        episode_lengths = episode_lengths[:total_episodes]

    image_offsets = np.zeros(total_episodes + 1, dtype=np.int64)
    np.cumsum(episode_lengths, out=image_offsets[1:])
    total_images = int(image_offsets[-1])

    img_size = tuple(config["img_size"])
    expected_shape = (total_images, *img_size)
    output_name = "images.debug.npy" if args.quick_debug else "images.npy"
    partial_name = "images.debug.partial.npy" if args.quick_debug else "images.partial.npy"
    progress_name = (
        "images.debug.partial.progress" if args.quick_debug else "images.partial.progress"
    )
    output_path = save_path / output_name
    partial_path = save_path / partial_name
    progress_path = save_path / progress_name

    if args.overwrite:
        remove_output_files((output_path, partial_path, progress_path))

    if output_path.exists():
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape != expected_shape or existing.dtype != np.uint8:
            raise RuntimeError(
                f"Existing {output_path} has shape {existing.shape} and dtype "
                f"{existing.dtype}; expected {expected_shape} and uint8. "
                "Use --overwrite to rebuild it."
            )
        print(f"Output is already complete: {output_path} ({existing.shape})")
        return

    if partial_path.exists() != progress_path.exists():
        raise RuntimeError(
            f"Incomplete resume state: expected both {partial_path} and "
            f"{progress_path}. Use --overwrite to rebuild from episode 0."
        )

    if partial_path.exists():
        next_episode = int(progress_path.read_text(encoding="utf-8").strip())
        if not 0 <= next_episode <= total_episodes:
            raise RuntimeError(
                f"Invalid resume episode {next_episode} in {progress_path}"
            )
        output = np.lib.format.open_memmap(partial_path, mode="r+")
        if output.shape != expected_shape or output.dtype != np.uint8:
            raise RuntimeError(
                f"Partial output has shape {output.shape} and dtype {output.dtype}; "
                f"expected {expected_shape} and uint8. Use --overwrite to rebuild it."
            )
        print(f"Resuming conversion at episode {next_episode}/{total_episodes}")
    else:
        next_episode = 0
        save_progress(progress_path, next_episode)
        output = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.uint8,
            shape=expected_shape,
        )

    print(
        f"Combining {total_images:,} PNGs from {total_episodes:,} episodes "
        f"into {output_path}"
    )
    print(
        f"Output shape: {expected_shape}; decoding workers: {args.num_workers}; "
        f"batch episodes: {args.batch_episodes}"
    )

    image_transform = select_transforms(config["env"])
    load_one = partial(load_image, image_transform, img_size)

    batch_starts = range(next_episode, total_episodes, args.batch_episodes)
    total_batches = (
        total_episodes - next_episode + args.batch_episodes - 1
    ) // args.batch_episodes

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            for batch_start in tqdm(
                batch_starts,
                total=total_batches,
                desc="Combining image batches",
                disable=not sys.stdout.isatty(),
            ):
                batch_end = min(
                    batch_start + args.batch_episodes,
                    total_episodes,
                )
                image_paths = [
                    input_image_path / f"{episode_idx}_{timestep}.png"
                    for episode_idx in range(batch_start, batch_end)
                    for timestep in range(int(episode_lengths[episode_idx]))
                ]
                images = list(executor.map(load_one, image_paths))
                write_start = int(image_offsets[batch_start])
                write_end = int(image_offsets[batch_end])
                output[write_start:write_end] = np.stack(images)
                output.flush()
                save_progress(progress_path, batch_end)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted safely. Re-run the same command to resume from "
            f"{progress_path}.",
            file=sys.stderr,
        )
        raise

    output.flush()
    del output
    os.replace(partial_path, output_path)
    progress_path.unlink()
    print(f"Finished writing {output_path}")


if __name__ == "__main__":
    main()
