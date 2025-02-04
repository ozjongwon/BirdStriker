import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torchvision.models import EfficientNet_V2_S_Weights
import torch.utils.checkpoint as checkpoint

class EfficientNetBird(nn.Module):
    def __init__(self, num_classes=200):
        super(EfficientNetBird, self).__init__()
        # Use the newer weights parameter instead of pretrained
        self.efficient_net = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Replace classifier
        in_features = self.efficient_net.classifier[1].in_features
        self.efficient_net.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Apply gradient checkpointing to the features
        #features = checkpoint.checkpoint(self.efficient_net.features, x, use_reentrant=False)
        # Process through classifier normally
        features = self.efficient_net.features(x)

        x = self.efficient_net.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.efficient_net.classifier(x)
        return x

class EnsembleModel(nn.Module):
    def __init__(self, num_models=3, num_classes=200):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([
            EfficientNetBird(num_classes) for _ in range(num_models)
        ])

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(F.softmax(model(x), dim=1))

        # Average the predictions
        return torch.mean(torch.stack(outputs), dim=0)

# 1. First define your model architecture exactly as before
model = EnsembleModel(num_models=3, num_classes=200)

# 2. Load the weights from your existing .pth file
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. Convert to TorchScript and save complete model
# dummy_input = torch.randn(32, 3, 384, 384)  # Assuming input is an image of size 384x384
# traced_model = torch.jit.trace(model, dummy_input)
# traced_model.save('model_complete.pt')

# Disable gradients for tracing
with torch.no_grad():
    # Create example input that matches your real input
    dummy_input = torch.randn(32, 3, 384, 384)  # Note: batch size 1 might work better

    # Try to trace the model
    try:
        traced_model = torch.jit.trace(model, dummy_input)

        # Test the traced model with another input
        #test_input = torch.randn(1, 3, 384, 384)
        test_output = traced_model(dummy_input)

        # traced_model(test_input)  # This should# work

        # If we get here, tracing succeeded
        traced_model.save('model_complete.pt')
        print("Model traced and saved successfully!")

    except Exception as e:
        print(f"Tracing failed: {str(e)}")
