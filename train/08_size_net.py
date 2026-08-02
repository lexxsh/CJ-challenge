"""
Dimension-head v2 (Deep3DBox 골격 + intrinsic-free 멀티 스케일 특징).
입력: 크롭(3xSxS) + geo벡터(G). 출력: 정렬된 (s1>=s2>=s3) cm = mean_dims + residual.
geo 특징(레일 기준, intrinsic 불변): [bw/rail, bh/rail, diag/rail, log(diag*cmpp+1), cy_norm]
"""
import torch
import torch.nn as nn

GEO_DIM = 5


def build_backbone(name="resnet18", pretrained=True):
    if name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); m.fc = nn.Identity(); return m, 512
    if name == "resnet34":
        from torchvision.models import resnet34, ResNet34_Weights
        m = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None); m.fc = nn.Identity(); return m, 512
    if name == "convnext_tiny":
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Flatten(1); return m, 768
    if name == "deit_small":
        import timm
        m = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=0)  # 풀링 feature 384
        return m, m.num_features
    raise ValueError(name)


class SizeNet2(nn.Module):
    def __init__(self, mean_dims=(33., 22., 14.), backbone="resnet18", pretrained=True, geo_dim=GEO_DIM,
                 out_mode="dims", vol_init=(0., 0., 0.)):
        super().__init__()
        bb, feat = build_backbone(backbone, pretrained)
        self.backbone = bb
        self.geo_mlp = nn.Sequential(nn.Linear(geo_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(feat + 64, 256), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(256, 3))
        self.out_mode = out_mode   # "dims"=mean+residual / "volparam"=[logV,log(s1/s2),log(s2/s3)] -> 재구성
        self.register_buffer("mean_dims", torch.tensor(mean_dims, dtype=torch.float32))
        self.register_buffer("vol_init", torch.tensor(vol_init, dtype=torch.float32))

    def forward(self, crop, geo):
        f = self.backbone(crop)
        if f.dim() > 2:
            f = f.flatten(1)
        g = self.geo_mlp(geo)
        raw = self.head(torch.cat([f, g], 1))
        if self.out_mode == "dims":
            return self.mean_dims + raw
        # volparam: 부피+비율로 재정식화 (스케일=부피, 모양=비율 분리)
        p = self.vol_init + raw
        V = torch.exp(p[:, 0]); a = torch.exp(p[:, 1]); b = torch.exp(p[:, 2])   # a=s1/s2, b=s2/s3
        s2 = (V * b / a).clamp(min=1e-6) ** (1.0 / 3.0)
        s1 = a * s2; s3 = s2 / b
        out = torch.stack([s1, s2, s3], 1)
        return torch.sort(out, dim=1, descending=True)[0]   # 정렬 보장(메트릭과 일치)
