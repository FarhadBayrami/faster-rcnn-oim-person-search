"""scripts/train_joint.py — Stage 3: joint end-to-end fine-tuning."""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import PRWDataset, get_train_transforms, get_val_transforms, collate_fn
from models import PersonSearchNet
from utils import load_config, seed_everything, save_checkpoint, load_checkpoint, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/joint.yaml")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_ds = PRWDataset(cfg["data"]["root"], split="train",
                          transforms=get_train_transforms())
    loader   = DataLoader(train_ds,
                          batch_size=cfg["data"]["batch_size"],
                          shuffle=True,
                          num_workers=cfg["data"]["num_workers"],
                          collate_fn=collate_fn)

    model = PersonSearchNet(cfg).to(device)

    # Optional warm-start
    ckpt_path = cfg["model"].get("checkpoint")
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # Load detector + reid weights selectively
        model.detector.load_state_dict(state.get("detector", state.get("model", {})), strict=False)
        print(f"Warm-started from {ckpt_path}")

    oc = cfg["optimizer"]
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=oc["lr"], momentum=oc["momentum"], weight_decay=oc["weight_decay"],
    )
    sc = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=sc["step_size"], gamma=sc["gamma"]
    )

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb_logs"))

    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for step, (images, targets) in enumerate(loader):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = loss_dict["loss_total"]

            optimizer.zero_grad()
            losses.backward()
            if cfg["training"].get("clip_grad_norm"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"]["clip_grad_norm"]
                )
            optimizer.step()

            epoch_loss += losses.item()
            global_step += 1
            if step % cfg["training"]["log_interval"] == 0:
                print(f"Epoch {epoch} [{step}/{len(loader)}]  loss={losses.item():.4f}")
                for k, v in loss_dict.items():
                    writer.add_scalar(f"joint/{k}", v.item(), global_step)

        scheduler.step()
        avg = epoch_loss / len(loader)

        state = {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch":     epoch,
            "best_metric": avg,
        }
        save_checkpoint(state, os.path.join(save_dir, "last_joint.pth"))
        if avg < best_loss:
            best_loss = avg
            save_checkpoint(state, os.path.join(save_dir, "best_model.pth"))

    writer.close()
    print("Joint training complete.")


if __name__ == "__main__":
    main()
