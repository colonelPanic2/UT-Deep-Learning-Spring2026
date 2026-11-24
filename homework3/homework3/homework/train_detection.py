import argparse
from pathlib import Path

import torch

from .datasets.road_dataset import load_data
from .metrics import DetectionMetric
from .models import load_model, save_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("drive_data"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("detector").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    segmentation_loss = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 5.0, 5.0], device=device)
    )
    depth_loss = torch.nn.L1Loss()
    loaders = {
        split: load_data(
            args.data_dir / split, transform_pipeline="default", shuffle=split == "train",
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        for split in ("train", "val")
    }
    metric = DetectionMetric()

    for epoch in range(args.epochs):
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            metric.reset()
            total_loss = 0.0
            total_samples = 0
            with torch.set_grad_enabled(training):
                for batch in loader:
                    images = batch["image"].to(device)
                    track = batch["track"].to(device)
                    depth = batch["depth"].to(device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits, predicted_depth = model(images)
                    loss = segmentation_loss(logits, track) + depth_loss(predicted_depth, depth)
                    if training:
                        loss.backward()
                        optimizer.step()
                    metric.add(logits.argmax(dim=1), track, predicted_depth, depth)
                    total_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)
            result = metric.compute()
            print(
                f"Epoch {epoch + 1}/{args.epochs} {split}: "
                f"loss={total_loss / max(total_samples, 1):.4f} "
                f"iou={result['iou']:.4f} "
                f"depth_mae={result['abs_depth_error']:.4f} "
                f"lane_depth_mae={result['tp_depth_error']:.4f}"
            )

    save_model(model)
