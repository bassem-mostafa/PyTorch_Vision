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

class _SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(C, C//r, stride=1, kernel_size = 1, bias = False)
        self.conv2 = nn.Conv2d(C//r, C, stride=1, kernel_size = 1, bias = False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [N, C, H, W]
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.relu(self.conv1(f))
        f = self.sigmoid(self.conv2(f))
        f = f[:,:,None,None]
        # f shape: [N, C, 1, 1]

        scale = x * f
        return scale

#here we will replace fc with 1 x 1 Conv 
class _SEBlock_Pointwise(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, C//r, bias=False)
        self.fc2 = nn.Linear(C//r, C, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [N, C, H, W]
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.relu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))
        f = f[:,:,None,None]
        # f shape: [N, C, 1, 1]

        scale = x * f
        return scale


"""
def mode_fuse(model):
    #is_qat = False
    #Fusion wont be done becuase we have only FrozenBN NOT BN ........................................................
   # fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
    for module_name, module in model.backbone.body.named_children():
        #print("module_name",module_name)
        #print("Module",module)
        if "layer" in module_name:
            for m in module:
                print("m", m.conv1)
                fuse_conv_bn(m.conv1, m.bn1)
                #fuse_conv_bn(m, ['conv2', 'bn2'])
                #fuse_conv_bn(m, ['conv3','bn3'])
#...........................................................................................................................

from torch.nn.utils import fuse_conv_bn_weights

def fuse_conv_bn(conv, bn):
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True
    )

    fused_conv.weight.data = fuse_conv_bn_weights(
        conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
    )

    return fused_conv
"""

class Bottleneck_Optimized(Bottleneck):
    """
    Wrapper for `Torch Vision` `ResNet` `Bottleneck`
    """
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None):
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        # TODO Modify for optimization
        self.expansion = 4
        width = int(planes * (base_width / 64.0)) * groups
        if False:
            self.se    = _SEBlock(planes * self.expansion )
        if False:
            self.se = _SEBlock_Pointwise(planes * self.expansion)
        if True:
            self.conv2 = _DepthWiseSeparable2D(width, width, stride)   #overwrite conv2D to DepthWS
        if True:
            self.relu  = nn.GELU()    #overwrite relu to gelu
        if False:
            # Cardinality Pattern
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self.cardinality = 32
            self.conv1 = nn.Conv2d(inplanes, 2*width, kernel_size=1, stride=1, bias=False)
            self.bn1 = norm_layer(2*width)
            self.conv2 = nn.Conv2d(
                                   2*width,
                                   2*width,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=dilation,
                                   groups=self.cardinality,
                                   bias=False,
                                   dilation=dilation,
                                  )
            self.bn2 = norm_layer(2*width)
            self.conv3 = nn.Conv2d(2*width, planes * self.expansion, kernel_size=1, stride=1, bias=False)



    def forward(self, x: Tensor) -> Tensor:
        # TODO Modify for optimization
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if False:
            out = self.se(out)    #Apply Squeeze and Excitation

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50_Optimized(ResNet):
    """
    Wrapper for `Torch Vision` `ResNet`
    """
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
        # TODO Modify for optimization


        
    def forward(self, x: Tensor) -> Tensor:
        # TODO Modify for optimization
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
        #state_dict =self.load_state_dict(load("C:/Users/nhlp4620/FreeTrial/Pytorch-Vision/references/detection/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", weights_only=True))
        state_dict =load("C:/Users/nhlp4620/FreeTrial/Pytorch-Vision/references/detection/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", weights_only=True)
        self.load_state_dict(state_dict, strict=False)
        # Load the modified state_dict back into the model
        # Filter state_dict to match modified model keys
        
       
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


#For Applying Quantization aware Training we need some steps:
"""
1- Load the model
2- Module Fusion: combine Conv2d with ReLU and BatchNorm2d ...

     --> "qat_model = load_model(saved_model_dir + float_model_file)
     --> qat_model.fuse_model(is_qat=True)"

3- Set quantization config to define how activations and weights are quantized during training....

     --> qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

4- Apply Quantization....
     --> torch.ao.quantization.prepare_qat(qat_model, inplace=True)
     --> print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)

"""
#mode_fuse(model)
#print("Fused Model", model)

model.eval()

print(f"*"*80)
print(f"Optimized Model Architecture")
print(f"-"*80)
print(model)
print(f"*"*80)

#summary(model, input_size=(1, 3, 1333, 800), depth = 10)