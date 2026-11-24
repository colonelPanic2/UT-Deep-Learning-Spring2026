import argparse
from pathlib import Path

import torch

from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric
from .models import load_model, save_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("classification_data"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("classifier").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    loaders = {
        "train": load_data(
            args.data_dir / "train", transform_pipeline="aug", shuffle=True,
            batch_size=args.batch_size, num_workers=args.num_workers,
        ),
        "val": load_data(
            args.data_dir / "val", transform_pipeline="default",
            batch_size=args.batch_size, num_workers=args.num_workers,
        ),
    }
    metric = AccuracyMetric()

    for epoch in range(args.epochs):
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            metric.reset()
            total_loss = 0.0
            total_samples = 0
            with torch.set_grad_enabled(training):
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    if training:
                        loss.backward()
                        optimizer.step()
                    metric.add(logits.argmax(dim=1), labels)
                    total_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)
            accuracy = metric.compute()["accuracy"]
            print(
                f"Epoch {epoch + 1}/{args.epochs} {split}: "
                f"loss={total_loss / max(total_samples, 1):.4f} "
                f"accuracy={accuracy:.4f}"
            )

    save_model(model)
