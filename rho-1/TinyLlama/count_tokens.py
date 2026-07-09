"""
count_tokens.py  —  count tokens in a PackedDataset .bin directory and
report the exact max_steps needed for a single clean pass.

Usage
-----
    python count_tokens.py --data_dir data/openwebmath

    # override training hyper-params if yours differ from the defaults:
    python count_tokens.py --data_dir data/openwebmath \
        --block_size 2048 --micro_batch_size 16 \
        --grad_accum 4 --world_size 8 --num_workers 8
"""

import argparse
import glob
import os
import struct

# ---------------------------------------------------------------------------
# Header layout written by PackedDatasetBuilder._write_chunk
# (tl_packed_dataset.py)
#
#   bytes  0-6  : magic      b"LITPKDS"  (7 bytes)
#   bytes  7-14 : version    uint64 LE   (8 bytes)
#   byte   15   : dtype_code uint8       (1 byte)
#   bytes 16-23 : chunk_size uint64 LE   (8 bytes)
#   -----------------------------------------------
#   HDR_SIZE = 24 bytes total
# ---------------------------------------------------------------------------
HDR_MAGIC = b"LITPKDS"

DTYPE_ITEMSIZE = {
    1: 1,  # uint8
    2: 1,  # int8
    3: 2,  # int16
    4: 4,  # int32
    5: 8,  # int64
    6: 4,  # float32
    7: 8,  # float64
    8: 2,  # uint16  ← TinyLlama uses this (vocab_size < 65500)
}


def read_header(path: str) -> tuple[int, int]:
    """Return (chunk_size, dtype_itemsize) from a single .bin file."""
    with open(path, "rb") as f:
        magic = f.read(7)
        if magic != HDR_MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}, expected {HDR_MAGIC!r}")
        _version   = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        chunk_size = struct.unpack("<Q", f.read(8))[0]

    if dtype_code not in DTYPE_ITEMSIZE:
        raise ValueError(f"{path}: unknown dtype code {dtype_code}")

    return chunk_size, DTYPE_ITEMSIZE[dtype_code]


def verify_file_size(path: str, chunk_size: int, dtype_itemsize: int) -> bool:
    """Check that the file is exactly HDR_SIZE + chunk_size * itemsize bytes.

    A truncated file (e.g. from a crashed write_reminder call) would have
    fewer bytes than expected and would silently produce wrong token counts
    if not caught here.
    """
    expected = 24 + chunk_size * dtype_itemsize
    actual   = os.path.getsize(path)
    return actual == expected


def count_tokens(
    data_dir: str,
    block_size: int,
    micro_batch_size: int,
    grad_accum: int,
    world_size: int,
    num_workers: int,
) -> None:

    filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if not filenames:
        raise RuntimeError(f"No .bin files found in {data_dir!r}")

    total_files = len(filenames)
    print(f"\n{'='*60}")
    print(f"  Directory : {os.path.abspath(data_dir)}")
    print(f"  .bin files found : {total_files:,}")
    print(f"{'='*60}")

    # --- read header from first file as the reference ---
    ref_chunk_size, ref_itemsize = read_header(filenames[0])
    print(f"\n  chunk_size (from first file) : {ref_chunk_size:,} tokens")
    print(f"  dtype itemsize               : {ref_itemsize} bytes")

    # --- validate every file ---
    bad_files = []
    wrong_header = []
    for path in filenames:
        try:
            chunk_size, itemsize = read_header(path)
        except ValueError as e:
            wrong_header.append((path, str(e)))
            continue

        if chunk_size != ref_chunk_size or itemsize != ref_itemsize:
            wrong_header.append((path, f"chunk_size={chunk_size}, itemsize={itemsize}"))
            continue

        if not verify_file_size(path, chunk_size, itemsize):
            expected = 24 + chunk_size * itemsize
            actual   = os.path.getsize(path)
            bad_files.append((path, expected, actual))

    if wrong_header:
        print(f"\n  WARNING: {len(wrong_header)} file(s) have inconsistent headers:")
        for path, msg in wrong_header[:5]:
            print(f"    {path}: {msg}")
        if len(wrong_header) > 5:
            print(f"    ... and {len(wrong_header)-5} more")

    if bad_files:
        print(f"\n  WARNING: {len(bad_files)} file(s) have unexpected sizes "
              f"(possibly truncated from a crashed write_reminder call):")
        for path, expected, actual in bad_files[:5]:
            print(f"    {path}: expected {expected:,} bytes, got {actual:,}")
        if len(bad_files) > 5:
            print(f"    ... and {len(bad_files)-5} more")

    # --- shard truncation (mirrors PackedDataset.__iter__ line 47) ---
    #
    #   num_shards   = world_size * num_workers
    #   usable_files = (total_files // num_shards) * num_shards
    #
    num_shards    = world_size * num_workers
    usable_files  = (total_files // num_shards) * num_shards
    dropped_files = total_files - usable_files

    # --- block and token accounting ---
    packed_block_size = block_size + 1          # tokens per block in the file
    blocks_per_file   = ref_chunk_size // packed_block_size
    total_blocks      = usable_files * blocks_per_file
    total_raw_tokens  = usable_files * ref_chunk_size   # every token stored
    total_sup_tokens  = total_blocks * block_size       # supervised positions

    # --- max_steps for exactly one pass ---
    samples_per_step  = micro_batch_size * grad_accum * world_size
    tokens_per_step   = samples_per_step * block_size
    max_steps         = total_blocks // samples_per_step
    leftover_blocks   = total_blocks % samples_per_step

    print(f"\n--- Token accounting ---")
    print(f"  block_size (supervised)       : {block_size:,}")
    print(f"  packed_block_size (in file)   : {packed_block_size:,}")
    print(f"  blocks_per_file               : {blocks_per_file:,}")
    print(f"  usable files (shard-aligned)  : {usable_files:,} / {total_files:,}  "
          f"(dropped {dropped_files})")
    print(f"  total blocks                  : {total_blocks:,}")
    print(f"  total raw tokens  (all files) : {total_raw_tokens/1e9:.4f} B  "
          f"({total_raw_tokens:,})")
    print(f"  total supervised tokens       : {total_sup_tokens/1e9:.4f} B  "
          f"({total_sup_tokens:,})")

    print(f"\n--- Training schedule ---")
    print(f"  world_size          : {world_size}")
    print(f"  num_workers         : {num_workers}")
    print(f"  num_shards          : {num_shards}  (world_size × num_workers)")
    print(f"  micro_batch_size    : {micro_batch_size}")
    print(f"  grad_accum          : {grad_accum}")
    print(f"  samples / step      : {samples_per_step:,}")
    print(f"  tokens  / step      : {tokens_per_step:,}")
    print(f"\n  *** max_steps (exact single pass) : {max_steps:,} ***")
    print(f"      leftover blocks after last step : {leftover_blocks:,}  "
          f"(never seen)")
    print(f"      tokens seen                     : "
          f"{max_steps * tokens_per_step / 1e9:.4f} B")

    # --- repeat warning ---
    old_max_steps = 14_700
    if max_steps < old_max_steps:
        over_pct = (old_max_steps - max_steps) / max_steps * 100
        print(f"\n  REPEAT WARNING: your previous --max_steps={old_max_steps:,} "
              f"exceeds the dataset by {over_pct:.1f}%.")
        print(f"  Use --max_steps {max_steps} in your training command.")
    elif max_steps > old_max_steps:
        under_pct = (max_steps - old_max_steps) / max_steps * 100
        print(f"\n  NOTE: dataset has {under_pct:.1f}% more steps than the old "
              f"--max_steps={old_max_steps:,}.")
        print(f"  Use --max_steps {max_steps} to see all data.")

    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count tokens in a PackedDataset .bin directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",          type=str, default="data/openwebmath")
    p.add_argument("--block_size",        type=int, default=2048)
    p.add_argument("--micro_batch_size",  type=int, default=16)
    p.add_argument("--grad_accum",        type=int, default=4)
    p.add_argument("--world_size",        type=int, default=8)
    p.add_argument("--num_workers",       type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count_tokens(
        data_dir        = args.data_dir,
        block_size      = args.block_size,
        micro_batch_size= args.micro_batch_size,
        grad_accum      = args.grad_accum,
        world_size      = args.world_size,
        num_workers     = args.num_workers,
    )