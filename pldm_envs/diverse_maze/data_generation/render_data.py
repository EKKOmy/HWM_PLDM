from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import torch
import argparse

from pldm_envs.diverse_maze import ant_draw
from pldm_envs.diverse_maze import maze_draw


def save_image(array, path):
    """Save a PNG atomically so interruption cannot leave a corrupt file."""
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        image = Image.fromarray(np.uint8(array))
        image.save(temporary_path, format="png")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def image_is_valid(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert proprio to images")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--workers_num", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--save_replace", action="store_true")
    parser.add_argument(
        "--verify_existing",
        action="store_true",
        help="Verify existing PNGs and re-render missing or corrupt files",
    )
    parser.add_argument("--quick_debug", action="store_true")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_image_path = data_path / "images"
    output_image_path.mkdir(parents=True, exist_ok=True)
    proprio_path = data_path / "data.p"

    config_path = data_path / "metadata.pt"
    config = torch.load(config_path, weights_only=False)

    # all_splits = d4rl_ds.splits
    all_splits = torch.load(proprio_path, weights_only=False)

    total_splits = len(all_splits)
    print(f"{total_splits=}")
    if args.workers_num < 1:
        parser.error("--workers_num must be a positive integer")
    if not 0 <= args.worker_id < args.workers_num:
        parser.error("--worker_id must be in [0, --workers_num)")

    # Give every worker a contiguous, non-overlapping slice.  The first
    # `remainder` workers process one additional split, so no dataset is
    # skipped when its size is not divisible by the worker count.
    splits_per_worker, remainder = divmod(total_splits, args.workers_num)
    start = args.worker_id * splits_per_worker + min(args.worker_id, remainder)
    end = start + splits_per_worker + int(args.worker_id < remainder)
    print(
        f"worker {args.worker_id}/{args.workers_num}: "
        f"splits {start}:{end} ({end - start} total)"
    )
    indices = np.arange(start, end)

    if "diverse" in config["env"]:
        # retrieve map metadata for generating custom environments
        map_metadata_path = data_path / "train_maps.pt"
        map_metadata = torch.load(map_metadata_path, weights_only=False)
    else:
        env = ant_draw.load_environment(config["env"])
        drawer = maze_draw.create_drawer(env, env.name)

    for split_idx in indices:
        split = all_splits[split_idx]
        map_idx = split["map_idx"]
        last_image = None
        split_drawer = None if "diverse" in config["env"] else drawer

        for img_idx, obs in tqdm(enumerate(split["observations"])):
            image_path = output_image_path / f"{split_idx}_{img_idx}.png"
            if image_path.exists() and not args.save_replace:
                if not args.verify_existing or image_is_valid(image_path):
                    continue
                print(f"Re-rendering corrupt image: {image_path}")

            if split_drawer is None:
                env = ant_draw.load_environment(
                    name=f"{config['env']}_{map_idx}",
                    map_key=map_metadata[map_idx],
                )
                split_drawer = maze_draw.create_drawer(env, env.name)
            last_image = split_drawer.render_state(obs)
            save_image(last_image, image_path)

            if args.quick_debug and img_idx > 10:
                break

        # save one image per split per layout for visualization
        image_folder = data_path / f"layout_{map_idx}"
        if os.path.exists(image_folder):
            img_sample_path = image_folder / "image_sample.png"
            if not img_sample_path.exists() or args.save_replace:
                if last_image is None:
                    if split_drawer is None:
                        env = ant_draw.load_environment(
                            name=f"{config['env']}_{map_idx}",
                            map_key=map_metadata[map_idx],
                        )
                        split_drawer = maze_draw.create_drawer(env, env.name)
                    last_image = split_drawer.render_state(split["observations"][-1])
                save_image(last_image, img_sample_path)

        if args.quick_debug and split_idx > 10:
            break


if __name__ == "__main__":
    main()
