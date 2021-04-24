from typing import All, Dict
import argparse

import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn



class ResnetTransformer(pl.LightningModule):
    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None) -> None:
        super().__init__()

        self.args = vars(args) if args is not None else {}
        
        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequetial(*layers)

        # use the pretrained model to classify