import torch
from torch import nn
from torch.nn import functional as F

from torchmetrics import Accuracy, Precision, Recall


class Head(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def loss(self, x, y):
        raise NotImplementedError

    def get_metrics(self):
        raise NotImplementedError


class TAPEGlobalHead(Head):

    def __init__(self, **kwargs):
        super().__init__()
        self.backbone_output_idx = 0
        self.in_features = kwargs["backbone_hidden_size"]
        self.hidden_features = kwargs["head_hidden_features"]

        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.fc2 = nn.Linear(self.hidden_features, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def get_metrics(self):
        return [Accuracy(),
                Precision(),
                Recall()]


class TAPEClsHead(TAPEGlobalHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = x[self.backbone_output_idx][:, 0, :]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class TAPEAvgHead(TAPEGlobalHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = torch.mean(x[self.backbone_output_idx], dim=1, keepdim=False)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class TAPEAttnHead(TAPEGlobalHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_layer = nn.Linear(self.in_features, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.attention_layer(x[self.backbone_output_idx]), dim=1)
        x = torch.sum(attn_weights * x[self.backbone_output_idx], dim=1, keepdim=False)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class TAPEMHAttnHead(TAPEGlobalHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multiheadattention_layer = nn.MultiheadAttention(self.in_features, 12, batch_first=True)
        self.Q_layer = nn.Linear(self.in_features, self.in_features)
        self.K_layer = nn.Linear(self.in_features, self.in_features)
        self.V_layer = nn.Linear(self.in_features, self.in_features)

    def forward(self, x):
        x = x[self.backbone_output_idx]
        q = self.Q_layer(x)
        k = self.K_layer(x)
        v = self.V_layer(x)
        x = self.multiheadattention_layer(q, k, v)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

