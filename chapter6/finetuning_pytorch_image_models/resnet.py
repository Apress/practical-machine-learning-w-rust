import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
# model = torch.load("model-best.pth", map_location='cpu')
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet18_model.pt")