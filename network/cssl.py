import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from torchmetrics import Accuracy

from src.metrics import D_lambda_Khan, Ds
from .wald_utilities import genMTF_MS


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, pad="same", pad_mode="reflect", bn=False, act=nn.GELU()):
        super().__init__()
        layers = []
        for idx in range(2):
            layers.append(
                nn.Conv2d(
                    n_feats,
                    n_feats,
                    kernel_size,
                    bias=bias,
                    padding=pad,
                    padding_mode=pad_mode,
                )
            )
            if bn:
                layers.append(nn.BatchNorm2d(n_feats))
            if idx == 0:
                layers.append(act)
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x)


class SDEM(nn.Module):
    def __init__(self, feature_dimension, kernel_size=7):
        super().__init__()
        self.channel_reduce = nn.Conv2d(feature_dimension, 1, kernel_size=1)
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False, padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()
        self.padding = kernel_size // 2

    def forward(self, x):
        reduced = self.channel_reduce(x)
        low_frequency = F.avg_pool2d(
            reduced,
            self.kernel_size,
            stride=1,
            padding=self.padding,
            count_include_pad=False,
        )
        high_frequency = reduced - low_frequency
        spatial_weights = self.sigmoid(self.conv(torch.cat([low_frequency, high_frequency], dim=1)))
        return x * spatial_weights + x


class AFSM(nn.Module):
    def __init__(self, feature_dimension, features_num, hidden_dim, dropout=0.05):
        super().__init__()
        self.cov_mlp = nn.Sequential(
            nn.Linear(feature_dimension, feature_dimension),
            nn.Dropout(dropout, inplace=False),
            nn.LeakyReLU(inplace=False),
            nn.Linear(feature_dimension, hidden_dim),
            nn.LeakyReLU(inplace=False),
            nn.Linear(hidden_dim, features_num),
        )

    def forward(self, x):
        _, _, height, width = x.shape
        x_flat = rearrange(x, "B C H W -> B (H W) C", H=height) / (height * width - 1)
        x_flat = x_flat - x_flat.mean(dim=-1, keepdim=True)
        cov = x_flat.transpose(-2, -1) @ x_flat
        cov_norm = torch.norm(x_flat, p=2, dim=-2, keepdim=True)
        cov_norm = cov_norm.transpose(-2, -1) @ cov_norm
        cov = cov / (cov_norm + 1e-8)
        weight = self.cov_mlp(cov).unsqueeze(-1)
        return x + weight * x


class AFRM(nn.Module):
    def __init__(self, feature_dimension):
        super().__init__()
        self.ASFM = AFSM(feature_dimension, 1, round(feature_dimension * 0.6))
        self.SDEM = SDEM(feature_dimension=feature_dimension, kernel_size=7)

    def forward(self, x):
        return self.SDEM(self.ASFM(x))


class FFTBranch(nn.Module):
    def __init__(self, in_channels, out_channels, magnitude_size, phase_size):
        super().__init__()
        self.magnitude_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=magnitude_size,
            padding=magnitude_size // 2,
            bias=False,
            padding_mode="reflect",
        )
        self.phase_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=phase_size,
            padding=phase_size // 2,
            bias=False,
            padding_mode="reflect",
        )

    def forward(self, x):
        x_fft = torch.fft.fft2(x)
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        magnitude = torch.abs(x_fft_shifted)
        phase = torch.angle(x_fft_shifted)
        adjusted = self.magnitude_conv(magnitude) * torch.exp(1j * self.phase_conv(phase))
        return torch.real(torch.fft.ifft2(torch.fft.ifftshift(adjusted, dim=(-2, -1))))


class SFCM(nn.Module):
    def __init__(self, in_channels, out_channels, magnitude_size, phase_size, fft=True, sa=True, ca=True, pad_mode="reflect"):
        super().__init__()
        self.FFT = fft
        self.SA = sa
        self.CA = ca
        if fft:
            self.fft_branch = FFTBranch(in_channels, out_channels, magnitude_size, phase_size)
        if sa:
            self.spatial_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode=pad_mode),
                SDEM(feature_dimension=out_channels, kernel_size=7),
            )
        if ca:
            self.se_layer = AFSM(out_channels, 1, round(out_channels * 0.6))

    def forward(self, x):
        spatial_output = self.spatial_branch(x) if self.SA else x
        combined_output = self.fft_branch(x) + spatial_output if self.FFT else spatial_output
        return self.se_layer(combined_output) if self.CA else combined_output


class CNNClassificationHead(nn.Module):
    def __init__(self, in_channels=32, num_classes=8):
        super().__init__()
        self.cnn_layers1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cnn_layers2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cnn_layers3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, num_classes),
        )
        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(4)

    def forward(self, fusion_features):
        features1 = self.cnn_layers1(fusion_features[0]) + self.down1(fusion_features[1])
        features2 = self.cnn_layers2(features1) + self.down2(fusion_features[2])
        features3 = self.cnn_layers3(features2).view(features2.size(0), -1)
        return self.classifier(features3)


class CSSL(pl.LightningModule):
    def __init__(
        self,
        ms_channels=8,
        dim=32,
        num_classes=8,
        task=True,
        sensor="worldview-2",
        spectral_weight=1.0,
        spatial_weight=1.0,
        learning_rate=1e-3,
        lr_step_size=50,
        lr_gamma=0.8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ms_channels = ms_channels
        self.sensor = sensor
        self.spectral_weight = spectral_weight
        self.spatial_weight = spatial_weight
        self.learning_rate = learning_rate
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.Task = task

        self.log_sigma_fusion = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))

        self.conv = nn.Sequential(
            nn.Conv2d(ms_channels + 1, dim, 3, 1, 1, bias=True, padding_mode="reflect"),
            nn.GELU(),
        )
        self.AFRM1 = nn.Sequential(AFRM(feature_dimension=dim), ResBlock(dim, 3, True))
        self.AFRM2 = nn.Sequential(AFRM(feature_dimension=dim), ResBlock(dim, 3, True))
        self.AFRM3 = nn.Sequential(AFRM(feature_dimension=dim), ResBlock(dim, 3, True))
        self.conv_out = nn.Conv2d(dim, ms_channels, 3, 1, 1, bias=True, padding_mode="reflect")

        if self.Task:
            self.classifier = CNNClassificationHead(in_channels=dim, num_classes=num_classes)

        self.Ds = Ds(ms_channels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, ms, upms, pan):
        x = self.conv(torch.cat([upms, pan], dim=1))
        x1 = self.AFRM1(x)
        x2 = self.AFRM2(x1)
        x3 = self.AFRM3(x2)
        fused_image = self.conv_out(x3) + upms
        if self.Task:
            return {"fused": fused_image, "class_logits": self.classifier([x1, x2, x3])}
        return {"fused": fused_image}

    def gradient_spatial_loss(self, fused, pan):
        sobel_x = torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
            device=pan.device,
            dtype=torch.float32,
        ) / 8
        sobel_y = torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
            device=pan.device,
            dtype=torch.float32,
        ) / 8
        fused_gray = torch.mean(fused, dim=1, keepdim=True)
        pan_grad_x = F.conv2d(pan, sobel_x, padding=1)
        pan_grad_y = F.conv2d(pan, sobel_y, padding=1)
        pan_grad_mag = torch.sqrt(pan_grad_x**2 + pan_grad_y**2 + 1e-8)
        fused_grad_x = F.conv2d(fused_gray, sobel_x, padding=1)
        fused_grad_y = F.conv2d(fused_gray, sobel_y, padding=1)
        fused_grad_mag = torch.sqrt(fused_grad_x**2 + fused_grad_y**2 + 1e-8)
        return F.l1_loss(fused_grad_mag, pan_grad_mag)

    def loss_function(self, pred_dict, ms, upms, pan, label):
        wald_ms = genMTF_MS(pred_dict["fused"], 4, self.sensor, self.ms_channels, self.device)
        spectral_loss = self.spectral_weight * F.l1_loss(wald_ms, ms)
        spatial_loss = self.spatial_weight * self.gradient_spatial_loss(pred_dict["fused"], pan)
        fusion_loss = spectral_loss + spatial_loss
        if self.Task:
            sigma_f = torch.exp(self.log_sigma_fusion)
            sigma_c = torch.exp(self.log_sigma_cls)
            cls_loss = F.cross_entropy(pred_dict["class_logits"], label)
            return fusion_loss / (2 * sigma_f**2) + self.log_sigma_fusion + cls_loss / (2 * sigma_c**2) + self.log_sigma_cls
        return fusion_loss

    def _step(self, batch, stage):
        ms, pan, upms, label = batch
        pred = self(ms, upms, pan)
        loss = self.loss_function(pred, ms, upms, pan, label)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if stage != "train":
            wald_ms = genMTF_MS(pred["fused"], 4, self.sensor, self.ms_channels, self.device)
            d_lambda = D_lambda_Khan(ms, wald_ms).mean()
            ds_value = self.Ds(pred["fused"], pan, ms).mean()
            hqnr = (1 - d_lambda) * (1 - ds_value)
            self.log("DlambdaF", d_lambda.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("Ds", ds_value.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("HQNR", hqnr.item(), on_step=False, on_epoch=True, prog_bar=False)
            if self.Task:
                acc = self.val_acc(pred["class_logits"], label)
                self.log("acc", acc.item(), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch, idx):
        return self._step(batch, "train")

    def validation_step(self, batch, idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
