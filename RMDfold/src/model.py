from torch import nn
from torch.nn.functional import cross_entropy
import torch as tr
from tqdm import tqdm
import pandas as pd
import random
import os
import numpy as np
from mamba_ssm import Mamba
from torch.utils.data import DataLoader
from dataset import SeqDataset, pad_batch
from metrics import contact_f1
from utils import mat2bp, postprocessing
from torch.amp import autocast, GradScaler
from utils import write_ct, ct2dot
from utils import dot2png, ct2svg
import time

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)
    tr.cuda.manual_seed_all(seed)
    tr.backends.cudnn.benchmark = False
    tr.backends.cudnn.deterministic = True

def getcsv(seq_names,seq_lens_list,run_time,mcc,recall,precision,f1_score,f1_post):
    df = pd.DataFrame({'names':seq_names,'seq_lens':seq_lens_list,'run_times':run_time,'mcc':mcc,'recall':recall,'precision':precision,'f1':f1_score,"f1_post":f1_post})
    df.to_csv(r"D:\RMDfold\result/archiveII.csv",index=False)
    return

def rmdfold(pretrained=False, weights=None, **kwargs):
    model = RMDfold(**kwargs)
    if pretrained:
        print("Load pretrained weights...")
        state_dict = tr.load(r"D:\RMDfold\weights/RMDfold.pt",map_location=tr.device(model.device))
        model.load_state_dict(state_dict)
    else:
        if weights is not None:
            print(f"Load weights from {weights}")
            model.load_state_dict(tr.load(weights, map_location=tr.device(model.device)))
        else:
            print("No weights provided, using random initialization")
    return model

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    @tr.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype == tr.float16:
            x = x.type(tr.float32)
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)
        return out+x

class DenseBlock1D(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, num_layers=2, dilation=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(in_channels + i * growth_rate),
                    nn.ReLU(),
                    nn.Conv1d(in_channels + i * growth_rate, growth_rate, kernel_size, dilation=dilation, padding="same"),
                )
            )
        self.conv1x1 = nn.Conv1d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(tr.cat(features, dim=1))
            features.append(out)
        out = tr.cat(features, dim=1)
        out = self.conv1x1(out)
        return out

class DenseBlock2D(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, num_layers=4, dilation=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(),
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size, dilation=dilation, padding="same"),
                )
            )
        self.conv1x1 = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)
        self.mamba = MambaLayer(in_channels, channel_token=False)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(tr.cat(features, dim=1))
            features.append(out)
        out = tr.cat(features, dim=1)
        out = self.conv1x1(out)
        out = self.mamba(out)
        return out

class RMDfold(nn.Module):
    def __init__(
        self,
        train_len=0,
        embedding_dim=4,
        device="cpu",
        negative_weight=0.1,
        lr=1e-4,
        loss_l1=0.01,
        scheduler="none",
        verbose=True,
        interaction_prior='none',
        output_th=0.5,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.class_weight = tr.tensor([negative_weight, 1.0]).float().to(device)
        self.loss_l1 = loss_l1
        self.verbose = verbose
        self.config = kwargs
        self.output_th = output_th
        mid_ch = 1
        self.interaction_prior = interaction_prior
        if interaction_prior != "none":
            mid_ch = 2

        self.build_graph(embedding_dim, mid_ch=mid_ch, **kwargs)
        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        self.scheduler_name = scheduler
        if scheduler == "plateau":
            self.scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, verbose=True
            )
        elif scheduler == "cycle":
            self.scheduler = tr.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr, steps_per_epoch=train_len, epochs=self.config["max_epochs"]
            )
        else:
            self.scheduler = None

        self.to(device)

    def build_graph(
        self,
        embedding_dim,
        kernel=3,
        filters=32,
        num_layers=2,
        dilation_desnet1d=3,
        desnet_bottleneck_factor=0.25,
        mid_ch=1,
        kernel_desnet2d=5,
        bottleneck1_desnet2d=256,
        filters_desnet2d=256,
        rank=64,
        dilation_desnet2d=3,
    ):
        pad = (kernel - 1) // 2
        self.use_restrictions = mid_ch != 1

        self.desnet1d = nn.Sequential(
            nn.Conv1d(embedding_dim, filters, kernel, padding="same"),
            DenseBlock1D(
                in_channels=filters,
                growth_rate=int(desnet_bottleneck_factor * filters),
                kernel_size=kernel,
                num_layers=num_layers + 2,
                dilation=dilation_desnet1d
            ),
            MambaLayer(filters, channel_token=False)
        )

        self.convrank1 = nn.Conv1d(
            in_channels=filters,
            out_channels=rank,
            kernel_size=kernel,
            padding=pad,
            stride=1,
        )
        self.convrank2 = nn.Conv1d(
            in_channels=filters,
            out_channels=rank,
            kernel_size=kernel,
            padding=pad,
            stride=1,
        )

        self.desnet2d = [nn.Conv2d(
            in_channels=mid_ch, out_channels=filters_desnet2d, kernel_size=7, padding="same"
        )]
        self.desnet2d += [
            DenseBlock2D(
                in_channels=filters_desnet2d,
                growth_rate=int(bottleneck1_desnet2d/4),
                kernel_size=kernel_desnet2d,
                num_layers=4,
                dilation=dilation_desnet2d
            )
        ]
        self.desnet2d = nn.Sequential(*self.desnet2d)

        self.conv2Dout = nn.Conv2d(
            in_channels=filters_desnet2d,
            out_channels=1,
            kernel_size=kernel_desnet2d,
            padding="same",
        )

    def forward(self, batch):
        x = batch["embedding"].to(self.device)
        batch_size = x.shape[0]
        L = x.shape[2]

        y = self.desnet1d(x)
        ya = self.convrank1(y)
        ya = tr.transpose(ya, -1, -2)
        yb = self.convrank2(y)

        y = ya @ yb
        yt = tr.transpose(y, -1, -2)
        y = (y + yt) / 2
        y0 = y.view(-1, L, L)

        if self.interaction_prior != "none":
            prob_mat = batch["interaction_prior"].to(self.device)
            x1 = tr.zeros([batch_size, 2, L, L]).to(self.device)
            x1[:, 0, :, :] = y0
            x1[:, 1, :, :] = prob_mat
        else:
            x1 = y0.unsqueeze(1)

        y = self.desnet2d(x1)
        # === 应用 MambaLayer ===
        y = self.conv2Dout(tr.relu(y)).squeeze(1)

        if batch["canonical_mask"] is not None:
            y = y.multiply(batch["canonical_mask"].to(self.device))
        yt = tr.transpose(y, -1, -2)
        y = (y + yt) / 2

        return y

    def loss_func(self, yhat, y):
        y = y.view(y.shape[0], -1)
        yhat= yhat
        yhat = yhat.view(yhat.shape[0], -1)
        l1_loss = tr.mean(tr.relu(yhat[y != -1]))
        yhat = yhat.unsqueeze(1)
        yhat = tr.cat((-yhat, yhat), dim=1)
        error_loss = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weight)
        loss = error_loss  + self.loss_l1 * l1_loss
        return loss





