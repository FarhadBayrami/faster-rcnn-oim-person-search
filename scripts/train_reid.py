"""scripts/train_reid.py — Stage 2: train Re-ID head with OIM loss on GT crops."""

import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import scipy.io as sio
import numpy as np

import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ReIDEmbedder, OIMLoss
from utils import load_config, seed_everything, save_checkpoint, get_device


class PRWCropDataset(Dataset):
    """Yields (crop_tensor, pid) pairs from GT bounding boxes."""

    def __init__(self, root: str, split: str = "train"):
        frame_mat = sio.loadmat(os.path.join(root, f"frame_{split}.mat"))
        key = [k for k in frame_mat if not k.startswith("_")][0]
        frame_names = [str(f[0]).strip() for f in frame_mat[key].squeeze()]

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self.samples = []  # (img_path, box, pid)
        for fname in frame_names:
            ann = os.path.join(root, "annotations", fname.replace(".jpg", ".mat"))
            img = os.path.join(root, "frames", fname)
            if not os.path.exists(ann):
                continue
            mat = sio.loadmat(ann)
            data = mat.get("box_new", mat.get("anno_file"))
            if data is None:
                continue
            data = data.squeeze()
            if data.ndim == 1:
                data = data[np.newaxis]
            for row in data:
                pid = int(row[4])
                if pid > 0:   # only labelled identities
                    x1, y1, w, h = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    self.samples.append((img, (x1, y1, x1+w, y1+h), pid - 1))  # 0-indexed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box, pid = self.samples[idx]
        img  = Image.open(img_path).convert("RGB")
        crop = img.crop(box)
        return self.transform(crop), pid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/reid.yaml")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_ds = PRWCropDataset(cfg["data"]["root"], split="train")
    loader   = DataLoader(train_ds,
                          batch_size=cfg["data"]["batch_size"],
                          shuffle=True,
                          num_workers=cfg["data"]["num_workers"],
                          drop_last=True)

    mc = cfg["model"]
    embedder = ReIDEmbedder(embedding_dim=mc["embedding_dim"]).to(device)
    oim      = OIMLoss(
        num_features=mc["oim_num_features"],
        num_pids=482,
        scalar=mc["oim_scalar"],
        momentum=mc["oim_momentum"],
    ).to(device)

    oc = cfg["optimizer"]
    optimizer = torch.optim.Adam(
        list(embedder.parameters()),
        lr=oc["lr"], weight_decay=oc["weight_decay"],
    )
    sc = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sc["milestones"], gamma=sc["gamma"]
    )

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb_logs"))

    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        embedder.train()
        epoch_loss = 0.0
        for step, (crops, pids) in enumerate(loader):
            crops = crops.to(device)
            pids  = pids.to(device)

            embs = embedder(crops)
            loss = oim(embs, pids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            if step % cfg["training"]["log_interval"] == 0:
                print(f"Epoch {epoch} [{step}/{len(loader)}]  loss={loss.item():.4f}")
                writer.add_scalar("reid/loss", loss.item(), global_step)

        scheduler.step()
        avg = epoch_loss / len(loader)
        writer.add_scalar("reid/epoch_loss", avg, epoch)

        state = {"model": embedder.state_dict(), "oim": oim.state_dict(),
                 "optimizer": optimizer.state_dict(), "epoch": epoch}
        save_checkpoint(state, os.path.join(save_dir, "last_reid.pth"))
        if avg < best_loss:
            best_loss = avg
            save_checkpoint(state, os.path.join(save_dir, "best_reid.pth"))

    writer.close()
    print("Re-ID training complete.")


if __name__ == "__main__":
    main()
