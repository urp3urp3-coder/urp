import torch
import torch.nn as nn
import torch.nn.functional as F

# SMP(ResNet-34 U-Net) 있으면 사용, 없으면 기존 간단 U-Net로 폴백
try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except Exception:
    _HAS_SMP = False


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class _PlainUNet(nn.Module):
    """기존 간단 U-Net (sigmoid 제거, 로짓 반환)"""
    def __init__(self, in_c=3, out_c=1, base=64):
        super().__init__()
        self.d1 = DoubleConv(in_c, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.u1 = DoubleConv(base * 4 + base * 2, base * 2)
        self.u2 = DoubleConv(base * 2 + base, base)
        self.out = nn.Conv2d(base, out_c, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        u1 = F.interpolate(c3, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.u1(torch.cat([u1, c2], dim=1))
        u2 = F.interpolate(u1, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u2, c1], dim=1))
        return self.out(u2)  # 로짓


class UNet(nn.Module):
    """
    선행연구 반영:
      - 기본: ResNet-34 encoder + ImageNet 가중치 (SMP 사용)
      - SMP 미설치 시: 기존 간단 U-Net로 폴백
    """
    def __init__(self, in_c=3, out_c=1, base=64):
        super().__init__()
        if _HAS_SMP:
            self.model = smp.Unet(
                encoder_name='resnet34',           # ✅ 선행연구 백본
                encoder_weights='imagenet',        # ✅ ImageNet 초기화
                in_channels=in_c,
                classes=out_c,
                activation=None                    # 로짓 반환
            )
            self._use_smp = True
        else:
            self.model = _PlainUNet(in_c=in_c, out_c=out_c, base=base)
            self._use_smp = False

    def forward(self, x):
        return self.model(x)  # 로짓