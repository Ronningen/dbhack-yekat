import torch
import pytorchvideo.models.resnet
import torch.nn.functional as F
import torch.nn as nn
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)
import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data


class KineticsDataModule(pytorch_lightning.LightningDataModule):
    _DATA_PATH = ""
    _CLIP_DURATION = 2  # Duration of sampled clip for each video
    _BATCH_SIZE = 8
    _NUM_WORKERS = 8  # Number of parallel processes fetching data

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self._CLIP_DURATION),
            decode_audio=False,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def train_dataloader(self):
        """
          Create the Kinetics train partition from the list of video labels
          in {self._DATA_PATH}/train.csv. Add transform that subsamples and
          normalizes the video before applying the scale, crop and flip augmentations.
          """
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45),
                                      (0.225, 0.225, 0.225)),
                            RandomShortSideScale(
                                min_size=256, max_size=320),
                            RandomCrop(244),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "train.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "random", self._CLIP_DURATION),
            transform=train_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = pytorchvideo.models.resnet.create_resnet(
            input_channel=3,  # RGB input
            model_depth=50,  # For the tutorial let's just use a 50 layer network
            model_num_class=2, # work or not
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.cross_entropy(y_hat, batch["label"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-1)


def train():
    classification_module = VideoClassificationLightningModule()
    data_module = KineticsDataModule()
    trainer = pytorch_lightning.Trainer()
    trainer.fit(classification_module, data_module)
