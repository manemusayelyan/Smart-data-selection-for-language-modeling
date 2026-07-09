from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from datasets import load_dataset

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt.tokenizer import Tokenizer


def prepare_full(process_id, num_processes, tokenizer_path, destination_path, chunk_size):

    tokenizer = Tokenizer(tokenizer_path)

    dataset = load_dataset(
        "open-web-math/open-web-math",
        split="train",
        streaming=True
    ).shard(num_shards=num_processes, index=process_id)

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"train_openwebmath_{process_id}",
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for example in tqdm(dataset):

        text = example["text"]
        tokens = tokenizer.encode(text)

        builder.add_array(np.array(tokens, dtype=builder.dtype))

    # builder.write_reminder()

# --------------------------------------------------------------------------------------------------

def prepare(tokenizer_path, destination_path, chunk_size=2049*1024):
    destination_path.mkdir(parents=True, exist_ok=True)
    num_processes = 16
    processes = []

    for i in range(num_processes):

        p = Process(
            target=prepare_full,
            args=(i, num_processes, tokenizer_path, destination_path, chunk_size)
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":

    prepare(
        tokenizer_path=Path("checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"),
        destination_path=Path("data/openwebmath")
    )