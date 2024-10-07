from torchinfo import summary # https://www.geeksforgeeks.org/how-to-print-the-model-summary-in-pytorch/
from torch import nn, load, Tensor
from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.misc import FrozenBatchNorm2d

import torch

from collections import OrderedDict

from torchvision.models.resnet import ResNet, Bottleneck

def _rename_module(module, old_name, new_name):
    module.__dict__['_modules'] = OrderedDict([(new_name, v) if k == old_name else (k, v) for k, v in module.__dict__['_modules'].items()])
    
class _DepthWiseSeparable2D(nn.Module):
    def __init__(self, inputChannel, outputChannel,stride, kernel_size = 3, padding = 1, bias = False):
        super(_DepthWiseSeparable2D, self).__init__()
        self.DepthWise = nn.Conv2d(inputChannel,inputChannel, stride=stride, kernel_size = kernel_size, padding = padding, groups = inputChannel, bias = bias)
        self.PointWise = nn.Conv2d(inputChannel, outputChannel, stride=1, kernel_size = 1, bias = bias)

    def forward(self, x):
        out = self.DepthWise(x)
        out = self.PointWise(out)

        return out
    

#Here we will replace Relu module with Gelu in "FasterRCNN_Optimized" model..............
def _Apply_GELU_(model):

    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            #replace the relu layer with gelu
            setattr(model, child_name, nn.GELU())
        else:
            _Apply_GELU_(child)

class Bottleneck_Optimized(Bottleneck):
    """
    Wrapper for `Torch Vision` `ResNet` `Bottleneck`
    """
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None):
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50_Optimized(ResNet):
    def __init__(self):
        super().__init__(
                        block = Bottleneck_Optimized,
                        layers = [3, 4, 6, 3],
                        # num_classes,
                        # zero_init_residual,
                        # groups,
                        # width_per_group,
                        # replace_stride_with_dilation,
                        norm_layer = FrozenBatchNorm2d,
                        )
    def forward(self, x: Tensor) -> Tensor:
        """
        *Overrides* `Torch-Vision` `ResNet` `forward` implementation
        """
        ...
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class FasterRCNN_Optimized(FasterRCNN):
    def __init__(self):
        backbone = _resnet_fpn_extractor(
                                        backbone = ResNet50_Optimized(),
                                        trainable_layers = 3, # trainable_layers (int): number of trainable (not frozen) layers starting from final block.
                                                              #                         Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
                                        )
        
        super().__init__(
                        backbone = backbone,
                        num_classes = 91,    # `num_classes` is independent of the backbone, could be set to any value, `91` is the value being set by the base model
                        # min_size,
                        # max_size,
                        # image_mean,
                        # image_std,
                        # rpn_anchor_generator,
                        # rpn_head,
                        # rpn_pre_nms_top_n_train,
                        # rpn_pre_nms_top_n_test,
                        # rpn_post_nms_top_n_train,
                        # rpn_post_nms_top_n_test,
                        # rpn_nms_thresh,
                        # rpn_fg_iou_thresh,
                        # rpn_bg_iou_thresh,
                        # rpn_batch_size_per_image,
                        # rpn_positive_fraction,
                        # rpn_score_thresh,
                        # box_roi_pool,
                        # box_head,
                        # box_predictor,
                        # box_score_thresh,
                        # box_nms_thresh,
                        # box_detections_per_img,
                        # box_fg_iou_thresh,
                        # box_bg_iou_thresh,
                        # box_batch_size_per_image,
                        # box_positive_fraction,
                        # bbox_reg_weights,
                        )
        # Here we're loading the pre-trained weights for the whole model `FasterRCNN + ResNet50 + FPN`
        self.load_state_dict(load("fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", weights_only=True))
        #self.load_state_dict(load("/home/ai1/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", weights_only=True))
        
        # TODO modify/fine-tune the backbone
        if True:
            # Updating backbone using depth-wise separable convolution instead of standard convolution
            #For Layer_1
            backbone.body.layer1[0].conv2=_DepthWiseSeparable2D(64, 64, kernel_size=(3, 3), stride=(1, 1)) #; _rename_module(backbone.body.layer1[0], "conv2", "DWS2")
            backbone.body.layer1[1].conv2=_DepthWiseSeparable2D(64, 64, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer1[2].conv2=_DepthWiseSeparable2D(64, 64, kernel_size=(3, 3), stride=(1, 1))
    
            #For Layer_2 
            backbone.body.layer2[0].conv2=_DepthWiseSeparable2D(128, 128, kernel_size=(3, 3), stride=(2, 2))
            backbone.body.layer2[1].conv2=_DepthWiseSeparable2D(128, 128, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer2[2].conv2=_DepthWiseSeparable2D(128, 128, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer2[3].conv2=_DepthWiseSeparable2D(128, 128, kernel_size=(3, 3), stride=(1, 1))
    
            #For Layer_3
            backbone.body.layer3[0].conv2=_DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(2, 2))
            backbone.body.layer3[1].conv2=_DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer3[2].conv2=_DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer3[3].conv2=_DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer3[4].conv2=_DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer3[5].conv2=_DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
    
            #For Layer_4
            backbone.body.layer4[0].conv2=_DepthWiseSeparable2D(512, 512, kernel_size=(3, 3), stride=(2, 2))
            backbone.body.layer4[1].conv2=_DepthWiseSeparable2D(512, 512, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer4[2].conv2=_DepthWiseSeparable2D(512, 512, kernel_size=(3, 3), stride=(1, 1))
        
        if True:
            # Updating backbone using GELU activation instead of Relu
            _Apply_GELU_(self)

        # The following snippets modifies the backbone architecture by different ways.
        
        # The following deletes an existing block
        # del backbone.body.conv1
        
        # The following modifies an existing block with another
        # from torch.nn import Conv2d # if previously imported not needed to re-import it
        # backbone.body.conv1 = Conv2d(3, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # The following adds a block within the model after an existing block
        # Working on it ...
        # from torch.nn import Conv2d # if previously imported not needed to re-import it
        # backbone_modules = list(backbone.body.named_children())
        # backbone_modules.insert(1, ("conv2", Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)))
        # backbone_modules.insert(2, ("conv3", Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)))
        # print(len(backbone_modules))
        # for mod in backbone_modules:
        #     print(mod)

        # FIXME How to add reflect this modification into backbone.body ?
        
        # FIXME The following adds to the end of body, how to inject them in between of existing layers ?
        # from torch.nn import Conv2d # if previously imported not needed to re-import it
        # backbone.body.conv2 = Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        # backbone.body.conv3 = Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)

model = FasterRCNN_Optimized()
model.eval()

print(f"*"*80)
print(f"Optimized Model Architecture")
print(f"-"*80)
print(model)
print(f"*"*80)

summary(model, input_size=(1, 3, 1333, 800), depth = 10)