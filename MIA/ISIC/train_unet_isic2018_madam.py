"""
U-Net training and inference for ISIC2018 skin lesion segmentation with MAdam.

Same setup as the Adam variant but replaces Adam + DiceCELoss with:
  - Two separate losses: Dice (task 0) and CE (task 1)
  - MAdam preconditioning with a Dirichlet-sampled preference ray each step
  - Adam as the base optimizer

Dataset: Dataset5143_ISIC2018 (binary segmentation: background vs lesion)
"""

import os
import sys
import json
import glob
import logging
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import monai
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    Resize,
    ScaleIntensity,
)
from monai.utils import set_determinism


def _ensure_madam_core() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "madam_core" / "src"
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return
    raise ImportError("Could not locate madam_core/src")


_ensure_madam_core()

from madam_core.madam import MAdamPreconditioner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = Path(
    os.environ.get("ISIC2018_DATA_ROOT", "./data/Dataset5143_ISIC2018")
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 30
EMA_DECAY = 0.999


def build_file_lists(images_dir: Path, labels_dir: Path):
    """Pair image and label paths based on the shared ISIC ID."""
    image_paths = sorted(glob.glob(str(images_dir / "*.png")))
    pairs = []
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        label_name = basename.replace("_0000.png", ".png")
        label_path = str(labels_dir / label_name)
        if os.path.exists(label_path):
            pairs.append({"image": img_path, "label": label_path})
        else:
            logger.warning("No matching label for %s", img_path)
    logger.info("Found %d image-label pairs in %s", len(pairs), images_dir)
    return pairs


# ---------------------------------------------------------------------------
# Transforms (identical to Adam variant)
# ---------------------------------------------------------------------------
def get_train_transforms(image_size):
    img_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(spatial_size=image_size),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandRotate(range_x=0.3, prob=0.5, keep_size=True),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ])
    lbl_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=image_size, mode="nearest"),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandRotate(range_x=0.3, prob=0.5, keep_size=True, mode="nearest"),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, mode="nearest"),
    ])
    return img_transforms, lbl_transforms


def get_val_transforms(image_size):
    img_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(spatial_size=image_size),
    ])
    lbl_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=image_size, mode="nearest"),
    ])
    return img_transforms, lbl_transforms


# ---------------------------------------------------------------------------
# Dataset wrapper (identical to Adam variant)
# ---------------------------------------------------------------------------
class PairedDataset(torch.utils.data.Dataset):
    """Apply separate transforms to image and label while synchronising random state."""

    def __init__(self, data_list, img_transform, lbl_transform):
        self.data_list = data_list
        self.img_transform = img_transform
        self.lbl_transform = lbl_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        seed = np.random.randint(0, 2**32)

        monai.utils.misc.set_determinism(seed=seed)
        image = self.img_transform(item["image"])

        monai.utils.misc.set_determinism(seed=seed)
        label = self.lbl_transform(item["label"])

        monai.utils.misc.set_determinism(seed=None)

        label = (label > 0.5).float()
        return image, label


# ---------------------------------------------------------------------------
# Separate loss functions for MAdam (two "tasks": Dice and CE)
# ---------------------------------------------------------------------------
def compute_dice_loss(outputs, labels):
    """Dice loss with sigmoid (task 0)."""
    dice_fn = DiceLoss(sigmoid=True, reduction="mean")
    return dice_fn(outputs, labels)


def compute_ce_loss(outputs, labels):
    """Binary cross-entropy with logits (task 1)."""
    return F.binary_cross_entropy_with_logits(outputs, labels, reduction="mean")


# ---------------------------------------------------------------------------
# Training loop — MAdam two-phase API
# ---------------------------------------------------------------------------
def train(args):
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- data lists ----
    data_root = Path(args.data_root)
    all_train_data = build_file_lists(data_root / "imagesTr", data_root / "labelsTr")
    n_val = int(len(all_train_data) * args.val_split)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(all_train_data))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_data = [all_train_data[i] for i in train_indices]
    val_data = [all_train_data[i] for i in val_indices]
    logger.info("Train: %d, Val: %d", len(train_data), len(val_data))

    # ---- transforms ----
    train_img_tx, train_lbl_tx = get_train_transforms(args.image_size)
    val_img_tx, val_lbl_tx = get_val_transforms(args.image_size)

    # ---- datasets / loaders ----
    train_ds = PairedDataset(train_data, train_img_tx, train_lbl_tx)
    val_ds = PairedDataset(val_data, val_img_tx, val_lbl_tx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- model ----
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # ---- MAdam preconditioner + Adam base optimizer ----
    params = [p for p in model.parameters() if p.requires_grad]
    precond = MAdamPreconditioner(
        params,
        num_tasks=2,  # task 0 = Dice, task 1 = CE
        ema_decay=args.ema_decay,
        eps_damp=args.eps_damp,
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # For validation, use combined Dice+CE (equal weight) to track val loss
    val_dice_fn = DiceLoss(sigmoid=True, reduction="mean")

    # ---- output dir ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    best_epoch = -1
    start_epoch = 1
    last_checkpoint_epoch = 0

    # ---- resume from rolling latest_checkpoint.pth ----
    state_path = out_dir / "training_state.json"
    ckpt_path = out_dir / "best_model.pth"
    latest_ckpt_path = out_dir / "latest_checkpoint.pth"
    if args.resume:
        if latest_ckpt_path.exists() and state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            model.load_state_dict(torch.load(latest_ckpt_path, map_location=device, weights_only=True))
            last_checkpoint_epoch = state.get("last_checkpoint_epoch", 0)
            start_epoch = last_checkpoint_epoch + 1
            best_val_dice = state.get("best_val_dice", -1.0)
            best_epoch = state.get("best_epoch", -1)
            logger.info(
                "Resumed from %s (last_checkpoint_epoch=%d, best_epoch=%d, best_dice=%.4f). Continuing at epoch %d.",
                latest_ckpt_path, last_checkpoint_epoch, best_epoch, best_val_dice, start_epoch,
            )
        else:
            logger.warning(
                "--resume set but no latest_checkpoint.pth found in %s. Starting from scratch.",
                out_dir,
            )

    for epoch in range(start_epoch, args.num_epochs + 1):
        # --- train with MAdam ---
        model.train()
        epoch_dice_loss = 0.0
        epoch_ce_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            # Separate losses as two "tasks" for MAdam
            loss_dice = compute_dice_loss(outputs, labels)
            loss_ce = compute_ce_loss(outputs, labels)
            task_losses = [loss_dice, loss_ce]

            # Sample a preference ray from Dirichlet(1,1)
            lam = precond.sample_ray()

            # Two-phase MAdam: update FIM, backward, precondition, step
            precond.update(task_losses, lam)
            scalarized = lam[0] * loss_dice + lam[1] * loss_ce
            scalarized.backward()
            precond.precondition()
            optimizer.step()

            epoch_dice_loss += loss_dice.item()
            epoch_ce_loss += loss_ce.item()

        epoch_dice_loss /= len(train_loader)
        epoch_ce_loss /= len(train_loader)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                vl_dice = val_dice_fn(outputs, labels)
                vl_ce = compute_ce_loss(outputs, labels)
                val_loss += (vl_dice.item() + vl_ce.item())
                preds = [post_pred(p) for p in decollate_batch(outputs)]
                labels_list = decollate_batch(labels)
                dice_metric(preds, labels_list)

        val_loss /= len(val_loader)
        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        logger.info(
            "Epoch %d/%d — dice_loss: %.4f | ce_loss: %.4f | val_loss: %.4f | val_dice: %.4f",
            epoch, args.num_epochs, epoch_dice_loss, epoch_ce_loss, val_loss, val_dice,
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  -> New best model saved (dice=%.4f)", best_val_dice)

        # Rolling latest checkpoint every save_every epochs (used by --resume)
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), latest_ckpt_path)
            last_checkpoint_epoch = epoch
            logger.info("  -> Latest checkpoint saved at epoch %d", epoch)

        # Persist training state every epoch so --resume knows where to continue
        with open(state_path, "w") as f:
            json.dump({
                "last_epoch": epoch,
                "best_val_dice": best_val_dice,
                "best_epoch": best_epoch,
                "last_checkpoint_epoch": last_checkpoint_epoch,
            }, f)

    logger.info("Training complete. Best val dice: %.4f at epoch %d", best_val_dice, best_epoch)
    return out_dir / "best_model.pth"


# ---------------------------------------------------------------------------
# Test inference (identical to Adam variant)
# ---------------------------------------------------------------------------
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running test inference on device: %s", device)

    data_root = Path(args.data_root)
    test_data = build_file_lists(data_root / "imagesTs", data_root / "labelsTs")
    val_img_tx, val_lbl_tx = get_val_transforms(args.image_size)
    test_ds = PairedDataset(test_data, val_img_tx, val_lbl_tx)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    ckpt = Path(args.output_dir) / "best_model.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    logger.info("Loaded checkpoint: %s", ckpt)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = [post_pred(p) for p in decollate_batch(outputs)]
            labels_list = decollate_batch(labels)
            dice_metric(preds, labels_list)

    test_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    logger.info("Test Dice Score: %.4f", test_dice)
    return test_dice


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="MONAI U-Net for ISIC2018 segmentation with MAdam preconditioning"
    )
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                        help="Dataset5143_ISIC2018 root containing imagesTr/labelsTr/imagesTs/labelsTs")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both",
                        help="Run training, testing, or both (default: both)")
    parser.add_argument("--output-dir", type=str, default="./isic2018_unet_madam_output",
                        help="Directory for checkpoints and results")
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE),
                        help="Resize target H W (default: 256 256)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY,
                        help="EMA decay for MAdam Fisher estimates (default: 0.999)")
    parser.add_argument("--eps-damp", type=float, default=1e-6,
                        help="Curvature damping inside the MAdam denominator")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest_checkpoint.pth in --output-dir")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Overwrite latest_checkpoint.pth every N epochs (default: 20)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.image_size = tuple(args.image_size)

    if args.mode in ("train", "both"):
        train(args)

    if args.mode in ("test", "both"):
        test(args)
