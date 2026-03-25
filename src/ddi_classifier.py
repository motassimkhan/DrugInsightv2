import json
import os

import torch
import torch.nn as nn


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_FEATURE_METADATA_PATH = os.path.join(
    ROOT_DIR,
    'data',
    'processed',
    'feature_metadata.json',
)


def load_feature_metadata(feature_metadata_path=DEFAULT_FEATURE_METADATA_PATH):
    with open(feature_metadata_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


class DDIClassifier(nn.Module):
    def __init__(
        self,
        drug_embed_dim=256,
        extra_dim=None,
        extra_features=None,
        feature_metadata_path=DEFAULT_FEATURE_METADATA_PATH,
        dropout=0.5,
    ):
        super().__init__()

        if extra_dim is None and extra_features is not None:
            extra_dim = extra_features
        if extra_dim is None:
            metadata = load_feature_metadata(feature_metadata_path)
            extra_dim = int(metadata['extra_dim'])

        self.extra_dim = int(extra_dim)
        input_dim = drug_embed_dim * 2 + self.extra_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.prob_head = nn.Linear(128, 1)
        self.severity_head = nn.Linear(128, 3)

    def forward(self, embed_a, embed_b, extra):
        x = torch.cat([embed_a, embed_b, extra], dim=-1)
        x = self.trunk(x)
        prob = self.prob_head(x)
        severity = self.severity_head(x)
        return prob, severity
