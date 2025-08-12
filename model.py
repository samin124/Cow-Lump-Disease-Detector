import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 2
CLASS_NAMES = ['Lumpy Skin', 'Normal Skin']

# CBAM modules (same as your code)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        max_ = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + max_)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, max_], dim=1)
        return self.sigmoid(self.conv(cat))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# Your model class, but CBAM initialized in __init__ (not in forward)
class EffNetCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.efficientnet_b3(pretrained=True)
        self.features = base_model.features
        in_features = base_model.classifier[1].in_features
        in_channels = 1536  # fixed for EfficientNet-B3 backbone output channels
        self.cbam = CBAM(in_channels=in_channels).to(DEVICE)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, NUM_CLASSES)
        )


    def forward(self, x):
        x = self.features(x)  # shape [B, C, H, W]
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

# Load your saved model
def load_model(path, device=DEVICE):
    model = EffNetCBAM().to(device)
    state_dict = torch.load(path, map_location=device)

    # Fix if saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Predict function from image bytes
def predict_image_bytes(model, image_bytes, device=DEVICE):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, dim=1)
        label = CLASS_NAMES[idx.item()]
        confidence = conf.item()
    return label, confidence
