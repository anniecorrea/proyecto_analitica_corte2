import torch
import torch.nn as nn
 
 
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
 
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))  # [B, 1, H, W]
 
 
class CBAM(nn.Module):
    """Secuencial: primero atencion de canal, luego atencion espacial sobre el resultado."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
 
    def forward(self, x):
        channel_attn = self.channel_attention(x)
        x_channel = x * channel_attn
        spatial_attn = self.spatial_attention(x_channel)
        x_out = x_channel * spatial_attn
        return x_out, channel_attn, spatial_attn
 
 
class CompuertaResidualCBAM(nn.Module):
    """y = x + gamma * (CBAM(x) - x), con gamma inicializado en 0.
    Con gamma=0, y = x exactamente: insertar CBAM no cambia nada al inicio.
    La red decide, durante el entrenamiento, cuanto usar la atencion."""
    def __init__(self, cbam):
        super().__init__()
        self.cbam = cbam
        self.gamma = nn.Parameter(torch.zeros(1))
 
    def forward(self, x):
        x_cbam, ch_attn, sp_attn = self.cbam(x)
        y = x + self.gamma * (x_cbam - x)
        return y, ch_attn, sp_attn
 