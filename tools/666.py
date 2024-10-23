from models_zoo import summary, get_model_stat
import torch
from thop import profile
import torchvision.models as models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50().to(device)
input_tensor = torch.randn(1, 3, 224, 224).to(device)

flops, params = get_model_stat(model, input_tensor, output_path=None, device=str(device))
print(f"FLOPs: {flops}")
print(f"Parameters: {params}")

