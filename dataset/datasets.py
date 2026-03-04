import os
import glob
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A


def _cityscapes_labelid_to_trainid_lut() -> np.ndarray:
    """
    Build LUT: labelId (0..255) -> trainId (0..18) or 255 (ignore).
    Cityscapes 'trainId' uses 19 classes; others are ignore(255).

    This LUT matches the official cityscapesScripts mapping.
    """
    lut = np.ones((256,), dtype=np.uint8) * 255  # default ignore

    # Official Cityscapes trainId mapping (19 classes)
    # trainId: 0..18
    # labelIds that map to each trainId:
    mapping = {
        0: [7],                    # road
        1: [8],                    # sidewalk
        2: [11],                   # building
        3: [12],                   # wall
        4: [13],                   # fence
        5: [17],                   # pole
        6: [19],                   # traffic light
        7: [20],                   # traffic sign
        8: [21],                   # vegetation
        9: [22],                   # terrain
        10: [23],                  # sky
        11: [24],                  # person
        12: [25],                  # rider
        13: [26],                  # car
        14: [27],                  # truck
        15: [28],                  # bus
        16: [31],                  # train
        17: [32],                  # motorcycle
        18: [33],                  # bicycle
    }

    for train_id, label_ids in mapping.items():
        for lid in label_ids:
            lut[lid] = train_id

    return lut


class CityscapesDataset(Dataset):
    """
    Returns:
        mask_onehot: [C, H, W] float32 (C=19 by default), ignore pixels are all-zeros
        img_tensor:  [3, H, W] float32 normalized to [-1, 1]

    Notes:
    - Uses *_gtFine_labelIds.png and maps labelId -> trainId.
    - Pairs files via filename stem, not sorting.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 512),  # (H, W) recommended for Cityscapes
        num_classes: int = 19,
        augment: bool = True,
        include_ignore_channel: bool = False,
    ):
        """
        Args:
            root: path to Cityscapes root containing leftImg8bit/ and gtFine/
            split: "train" or "val"
            image_size: (H, W) output size
            num_classes: 19 for trainIds
            augment: enables horizontal flip
            include_ignore_channel: if True, output channels = 20 (extra ignore channel)
        """
        self.root = root
        self.split = split
        self.H, self.W = image_size
        self.num_classes = num_classes
        self.augment = augment
        self.include_ignore_channel = include_ignore_channel

        if self.num_classes != 19:
            raise ValueError("For Cityscapes trainId mapping, num_classes must be 19.")

        self.img_glob = os.path.join(root, "leftImg8bit", split, "*", "*_leftImg8bit.png")
        self.mask_glob = os.path.join(root, "gtFine", split, "*", "*_gtFine_labelIds.png")

        self.img_paths = sorted(glob.glob(self.img_glob))
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found. Check path/glob: {self.img_glob}")

        # Map from image stem -> mask path
        mask_paths = glob.glob(self.mask_glob)
        if len(mask_paths) == 0:
            raise FileNotFoundError(f"No masks found. Check path/glob: {self.mask_glob}")

        self.mask_by_stem = {}
        for mp in mask_paths:
            base = os.path.basename(mp)
            # e.g., frankfurt_000000_000294_gtFine_labelIds.png
            stem = base.replace("_gtFine_labelIds.png", "")
            self.mask_by_stem[stem] = mp

        # Keep only paired items
        self.pairs: List[Tuple[str, str]] = []
        for ip in self.img_paths:
            base = os.path.basename(ip)
            # e.g., frankfurt_000000_000294_leftImg8bit.png
            stem = base.replace("_leftImg8bit.png", "")
            mp = self.mask_by_stem.get(stem, None)
            if mp is not None:
                self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            raise RuntimeError("No (image, mask) pairs found. Directory structure may be wrong.")

        # LUT for labelId -> trainId
        self.lut = _cityscapes_labelid_to_trainid_lut()

        # Albumentations: resize + optional flip
        tfs = [
            A.Resize(height=self.H, width=self.W, interpolation=cv2.INTER_LINEAR),
        ]
        if self.augment:
            tfs.append(A.HorizontalFlip(p=0.5))

        self.transform = A.Compose(
            tfs,
            additional_targets={"mask": "mask"},
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # labelIds PNG is single-channel
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        augmented = self.transform(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]

        # labelId -> trainId
        mask_train = self.lut[mask]  # uint8, values in 0..18 or 255(ignore)

        # Image -> [-1, 1], CHW
        img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()

        # Mask one-hot with ignore handling
        mask_t = torch.from_numpy(mask_train.astype(np.int64))  # [H, W]
        ignore = (mask_t == 255)

        if self.include_ignore_channel:
            # 20 channels: 19 classes + 1 ignore channel
            # Set ignore pixels to class 0 temporarily for one-hot, then overwrite ignore channel
            tmp = mask_t.clone()
            tmp[ignore] = 0
            onehot = F.one_hot(tmp, num_classes=19).float()  # [H, W, 19]
            ignore_ch = ignore.float().unsqueeze(-1)         # [H, W, 1]
            mask_onehot = torch.cat([onehot, ignore_ch], dim=-1)  # [H, W, 20]
            mask_onehot = mask_onehot.permute(2, 0, 1).contiguous()
        else:
            # 19 channels, ignore pixels are all zeros
            tmp = mask_t.clone()
            tmp[ignore] = 0
            mask_onehot = F.one_hot(tmp, num_classes=19).float()  # [H, W, 19]
            mask_onehot[ignore] = 0.0
            mask_onehot = mask_onehot.permute(2, 0, 1).contiguous()

        return mask_onehot, img_tensor
    

