import os
import glob
import time
import random
import math
import datetime
import argparse
import contextlib
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM

from lit_gpt.packed_dataset import PackedDataset


def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=120),
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_dataloader(
    data_dir: str,
    batch_size: int,
    block_size: int,
    rank: int,
    world_size: int,
    seed: int = 42,
    shuffle: bool = True,
) -> DataLoader:
    filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if not filenames:
        raise RuntimeError(f"No .bin files found in {data_dir}")

    random.seed(seed)
    random.shuffle(filenames)

    dataset = PackedDataset(
        filenames,
        n_chunks=8,
        block_size=block_size + 1,  # 2048 supervised positions need 2049 tokens
        shuffle=shuffle,
        seed=seed + rank,
        num_processes=world_size,
        process_rank=rank,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
    )


def is_main(rank: int) -> bool:
    return rank == 0


def get_lr(
    it: int,
    warmup_steps: int,
    stable_steps: int,
    max_steps: int,
    learning_rate: float,
    min_lr: float,
) -> float:

    # warmup
    if it < warmup_steps:
        return learning_rate * it / warmup_steps

    # stable
    if it < warmup_steps + stable_steps:
        return learning_rate

    # decay
    decay_steps = max_steps - warmup_steps - stable_steps
    decay_it = it - warmup_steps - stable_steps

    decay_ratio = decay_it / decay_steps
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)


def compute_token_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)


def top_ratio_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"selection ratio must be in (0, 1], got {ratio}")

    flat_scores = scores.reshape(-1)
    k = max(1, math.ceil(flat_scores.numel() * ratio))

    if k == flat_scores.numel():
        return torch.ones_like(scores, dtype=torch.bool)

    threshold = torch.topk(flat_scores, k=k, sorted=False).values.min()
    return scores >= threshold

class TinyLlamaCPTTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.local_rank, self.rank, self.world_size = setup_ddp()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.is_main = is_main(self.rank)

        self.grad_accum = args.gradient_accumulation_steps

        global_batch_tokens = (
            args.micro_batch_size
            * args.block_size
            * self.grad_accum
            * self.world_size
        )

        if self.is_main:
            print(f"[global batch tokens] : {global_batch_tokens:,}")
            print(f"[world size]           : {self.world_size}")
            print(f"[grad accum steps]     : {self.grad_accum}")

        self.out_dir = Path(args.out_dir)
        if self.is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.is_main:
            print(f"[model] loading from {args.checkpoint_dir}")

        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.local_rank},
        )

        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        self.model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            gradient_as_bucket_view=True,
        )

        ref_checkpoint_dir = args.reference_checkpoint_dir or args.checkpoint_dir
        if self.is_main:
            print(f"[reference model] loading from {ref_checkpoint_dir}")

        self.reference_model = AutoModelForCausalLM.from_pretrained(
            ref_checkpoint_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.local_rank},
        )
        self.reference_model.config.use_cache = False
        self.reference_model.eval()
        self.reference_model.requires_grad_(False)

        if self.is_main:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"[model] total parameters: {n_params:,}")

        param_dict = {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        if self.is_main:
            print(
                f"[optim] decayed tensors    : {len(decay_params):,}  "
                f"({sum(p.numel() for p in decay_params):,} params)"
            )
            print(
                f"[optim] non-decayed tensors: {len(nodecay_params):,}  "
                f"({sum(p.numel() for p in nodecay_params):,} params)"
            )

        self.optimizer = AdamW(
            optim_groups,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=1e-8,
            foreach=False
        )

        initial_lr = get_lr(
            0,
            args.warmup_steps,
            args.stable_steps,
            args.max_steps,
            args.learning_rate,
            args.min_lr,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = initial_lr

        self.train_loader = create_dataloader(
            data_dir=args.train_data_dir,
            batch_size=args.micro_batch_size,
            block_size=args.block_size,
            rank=self.rank,
            world_size=self.world_size,
            seed=args.seed,
            shuffle=True,
        )

        self.iter_num = 0
        self.step_count = 0

    def _save_checkpoint(self):
        if not self.is_main:
            return

        ckpt_dir = self.out_dir / f"step-{self.step_count}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model.module.save_pretrained(ckpt_dir)
        torch.save(
            {
                "iter_num": self.iter_num,
                "step_count": self.step_count,
                "optimizer": self.optimizer.state_dict(),
            },
            ckpt_dir / "trainer_state.pt",
        )

        print(f"[ckpt] saved -> {ckpt_dir}")

    def train(self):
        args = self.args
        grad_accum = args.gradient_accumulation_steps
        max_iters = args.max_steps * grad_accum

        self.model.train()
        self.reference_model.eval()
        self.optimizer.zero_grad(set_to_none=True)

        train_iter = iter(self.train_loader)

        t0 = time.perf_counter()
        tokens_since_log = 0

        loss_num = torch.zeros(1, device=self.device)
        loss_denom = torch.zeros(1, device=self.device)

        while self.iter_num < max_iters:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch = batch.to(self.device, non_blocking=True)

            expected_seq_len = args.block_size + 1
            if batch.ndim != 2 or batch.size(1) != expected_seq_len:
                raise RuntimeError(
                    f"Expected batch shape [B, {expected_seq_len}], got {tuple(batch.shape)}"
                )

            is_accumulating = (self.iter_num + 1) % grad_accum != 0
            sync_ctx = self.model.no_sync() if is_accumulating else contextlib.nullcontext()

            with sync_ctx:
                labels = batch[:, 1:].contiguous()

                out = self.model(input_ids=batch)
                logits = out.logits[:, :-1, :].contiguous()
                token_loss = compute_token_ce_loss(logits, labels)
                del out, logits

                with torch.no_grad():
                    ref_out = self.reference_model(input_ids=batch)
                    ref_logits = ref_out.logits[:, :-1, :].contiguous()
                    ref_token_loss = compute_token_ce_loss(ref_logits, labels)

                    ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
                    ref_probs = ref_log_probs.exp()
                    ref_entropy = -(ref_probs * ref_log_probs).sum(dim=-1)
                    del ref_out, ref_logits

                high_loss_mask = top_ratio_mask(ref_token_loss, args.rm_loss_topk_ratio)
                high_entropy_mask = top_ratio_mask(ref_entropy, args.rm_entropy_topk_ratio)
                final_mask = high_loss_mask & high_entropy_mask

                final_mask = final_mask.to(token_loss.dtype)
                selected_tokens = final_mask.sum()
                masked_loss_sum = (token_loss * final_mask).sum()
                loss = masked_loss_sum / selected_tokens.clamp_min(1.0)

                (loss / grad_accum).backward()

                n_tokens = batch.size(0) * (batch.size(1) - 1)
                loss_num += masked_loss_sum.detach()
                loss_denom += selected_tokens.detach()
                tokens_since_log += n_tokens

            if not is_accumulating:
                next_step = self.step_count + 1
                lr = get_lr(
                    next_step,
                    args.warmup_steps,
                    args.stable_steps,
                    args.max_steps,
                    args.learning_rate,
                    args.min_lr,
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    args.grad_clip,
                )
                grad_norm_value = float(grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.step_count = next_step

                if self.step_count % args.log_interval == 0:
                    dist.all_reduce(loss_num, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_denom, op=dist.ReduceOp.SUM)

                    loss_avg = loss_num / loss_denom.clamp_min(1.0)

                    if self.is_main:
                        elapsed = time.perf_counter() - t0
                        global_tokens = tokens_since_log * self.world_size
                        tok_per_s = global_tokens / elapsed

                        print(
                            f"step {self.step_count:>6}/{args.max_steps} | "
                            f"loss {loss_avg.item():.4f} | "
                            f"lr {lr:.2e} | "
                            f"grad_norm {grad_norm_value:.2f} | "
                            f"tok/s {tok_per_s:,.0f}"
                        )

                    loss_num.zero_()
                    loss_denom.zero_()
                    tokens_since_log = 0
                    t0 = time.perf_counter()

                if self.step_count % args.save_interval == 0:
                    dist.barrier()
                    self._save_checkpoint()
                    dist.barrier()

            self.iter_num += 1

        dist.barrier()

        if self.is_main:
            print("[train] done - saving final checkpoint")
            self._save_checkpoint()

        dist.barrier()
        cleanup_ddp()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TinyLlama continual pretraining (DDP)")

    p.add_argument("--checkpoint_dir", type=str, default="/mnt/weka/mmusayelyan/tinyllama_checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T")
    p.add_argument("--reference_checkpoint_dir", type=str, default="/mnt/weka/mmusayelyan/tinyllama_ct/step-13952")
    p.add_argument("--train_data_dir", type=str, default="/mnt/weka/mmusayelyan/data/prep_datasets/owm")
    p.add_argument("--out_dir", type=str, default="/mnt/weka/mmusayelyan/rho-1")

    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--micro_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)

    p.add_argument("--learning_rate", type=float, default=8e-5)
    p.add_argument("--min_lr", type=float, default=8e-6)

    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)

    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--stable_steps", type=int, default=10862)
    p.add_argument("--max_steps", type=int, default=13952)

    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=6976)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rm_loss_topk_ratio", type=float, default=0.6)
    p.add_argument("--rm_entropy_topk_ratio", type=float, default=0.6)

    return p.parse_args()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    args = parse_args()
    trainer = TinyLlamaCPTTrainer(args)
    trainer.train()
