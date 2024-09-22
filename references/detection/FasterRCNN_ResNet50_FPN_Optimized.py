from torchinfo import summary # https://www.geeksforgeeks.org/how-to-print-the-model-summary-in-pytorch/
from torch import nn, load
from torchvision.models.detection import FasterRCNN

from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class DepthWiseSeparable2D(nn.Module):
    def __init__(self, inputChannel, outputChannel,stride, kernel_size = 3, padding = 1, bias = False):
        super(DepthWiseSeparable2D, self).__init__()
        self.DepthWise = nn.Conv2d(inputChannel,inputChannel, stride=stride, kernel_size = kernel_size, padding = padding, groups = inputChannel, bias = bias)
        self.PointWise = nn.Conv2d(inputChannel, outputChannel, stride=1, kernel_size = 1, bias = bias)

    def forward(self, x):
        out = self.DepthWise(x)
        out = self.PointWise(out)

        return out
    

#Here we will replace Relu module with Gelu in "FasterRCNN_Optimized" model..............
def Apply_GELU_(model):

    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            #replace the relu layer with gelu
            setattr(model, child_name, nn.GELU())
        else:
            Apply_GELU_(child)




class FasterRCNN_Optimized(FasterRCNN):
    def __init__(self):
        backbone = resnet_fpn_backbone(
                                      backbone_name = "resnet50",
                                      weights = None, # `None` for NO pre-trained weights loading
                                                      # `ResNet50_Weights.DEFAULT` for specific pre-trained weights loading
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
        if False:
            # Updating backbone using depth-wise separable convolution instead of standard convolution
            #For Layer_1
            backbone.body.layer1[0].conv2=DepthWiseSeparable2D(64, 64, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer1[1].conv2=DepthWiseSeparable2D(64, 64, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer1[2].conv2=DepthWiseSeparable2D(64, 64, kernel_size=(3, 3), stride=(1, 1))
    
            #For Layer_2 
            backbone.body.layer2[0].conv2=DepthWiseSeparable2D(128, 128, kernel_size=(3, 3), stride=(2, 2))
            backbone.body.layer2[1].conv2=DepthWiseSeparable2D(128, 128, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer2[2].conv2=DepthWiseSeparable2D(128, 128, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer2[3].conv2=DepthWiseSeparable2D(128, 128, kernel_size=(3, 3), stride=(1, 1))
    
            #For Layer_3
            backbone.body.layer3[0].conv2=DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(2, 2))
            backbone.body.layer3[1].conv2=DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer3[2].conv2=DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer3[3].conv2=DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer3[4].conv2=DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer3[5].conv2=DepthWiseSeparable2D(256, 256, kernel_size=(3, 3), stride=(1, 1))
    
            #For Layer_4
            backbone.body.layer4[0].conv2=DepthWiseSeparable2D(512, 512, kernel_size=(3, 3), stride=(2, 2))
            backbone.body.layer4[1].conv2=DepthWiseSeparable2D(512, 512, kernel_size=(3, 3), stride=(1, 1))
            backbone.body.layer4[2].conv2=DepthWiseSeparable2D(512, 512, kernel_size=(3, 3), stride=(1, 1))

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


#here we will replace Relu activation function with Gelu....

Apply_GELU_(model)


"""
for name,child in model.backbone.body.named_modules(): 
    #print(child)
    if  hasattr(child,'relu'):
        print("I'm here")
        child._modules['relu'] = nn.GELU()
        #child.relu = nn.GELU()
        print("I'm not here")

"""

# from torch import load
# model.load_state_dict(load("fasterrcnn_resnet50_fpn_coco.pth", weights_only=True))

model.eval()

print(f"*"*80)
print(f"Optimized Model Architecture")
print(f"-"*80)
print(model)
print(f"*"*80)

summary(model, input_size=(1, 3, 1333, 800), depth = 10)