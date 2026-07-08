import argparse
import glob
import os
import struct

HDR_MAGIC = b"LITPKDS"

DTYPE_ITEMSIZE = {
    1: 1, 2: 1, 3: 2, 4: 4,
    5: 8, 6: 4, 7: 8, 8: 2,
}

def read_header(path):
    with open(path, "rb") as f:
        if f.read(7) != HDR_MAGIC:
            raise ValueError("bad magic")
        f.read(8)
        dtype_code = struct.unpack("<B", f.read(1))[0]
        chunk_size = struct.unpack("<Q", f.read(8))[0]
    return chunk_size, DTYPE_ITEMSIZE[dtype_code]

def verify_file(path, chunk, size):
    return os.path.getsize(path) == 24 + chunk * size


def main(args):
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.bin")))
    if not files:
        raise RuntimeError("No .bin files")

    ref_chunk, ref_size = read_header(files[0])

    valid_files = []
    for f in files:
        try:
            chunk, size = read_header(f)
            if chunk == ref_chunk and size == ref_size and verify_file(f, chunk, size):
                valid_files.append(f)
        except:
            continue

    total_files = len(valid_files)

    num_shards = args.world_size * args.num_workers
    usable_files = (total_files // num_shards) * num_shards

    valid_files = valid_files[:usable_files]

    total_tokens = usable_files * ref_chunk

    packed_block = args.block_size + 1
    blocks_per_file = ref_chunk // packed_block
    total_blocks = usable_files * blocks_per_file

    samples_per_step = (
        args.micro_batch_size *
        args.grad_accum *
        args.world_size
    )

    tokens_per_step = samples_per_step * args.block_size
    max_steps = total_blocks // samples_per_step

    print("\n--- MATCHED TO TRAINING ---")
    print(f"total files        : {total_files:,}")
    print(f"usable files       : {usable_files:,}")
    print(f"dropped files      : {total_files - usable_files:,}")
    print(f"total tokens       : {total_tokens/1e9:.4f} B")
    print(f"total blocks       : {total_blocks:,}")
    print(f"tokens / step      : {tokens_per_step:,}")
    print(f"\n*** max_steps      : {max_steps:,} ***\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--micro_batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--world_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    main(args)