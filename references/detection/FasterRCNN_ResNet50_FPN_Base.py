from torchinfo import summary # https://www.geeksforgeeks.org/how-to-print-the-model-summary-in-pytorch/

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

print(f"*"*80)
print(f"Base Model Architecture")
print(f"-"*80)
print(model)
print(f"*"*80)

summary(model, input_size=(1, 3, 1333, 800), depth = 10)
