import torch.nn as nn 
import torchvision

class Yolov1(nn.Module):
    #1. Backbone consisting of Resnet32 pretrained on 224x224 images from Imagenet
    #2. 4 Conv,Batchnorm,LeakyReLU Layers for Yolo Detection Head
    #3. Fc layers with final layer having S*S*(5B+C) output dimensions

    def __init__(self, im_size, num_classes, model_config):
        super().__init__(self)
        self.im_size=im_size
        self.backbone_channels = model_config['backbone_channels']
        self.yolo_conv_channels = model_config['yolo_conv_channels']
        self.conv_spatial_size = model_config['conv_spatial_size']
        self.leaky_relu_slope = model_config['leaky_relu_slope']
        self.yolo_fc_hidden_dim = model_config['fc_dim']
        self.yolo_fc_dropout_prob = model_config['fc_dropout']
        self.use_conv = model_config['use_conv']
        self.S = model_config['S'] #S x S is the nb of grids
        self.B = model_config['B'] #nb of boxes predicted for each grid
        self.C = num_classes

        backbone = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )

        # Backbone Layers #
        ###################
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Detection Conv Layers #
        #########################
        self.conv_yolo_layers = nn.Sequential(
            nn.Conv2d(self.backbone_channels,
                      self.yolo_conv_channels,
                      3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels,
                      self.yolo_conv_channels,
                      3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels,
                      self.yolo_conv_channels,
                      3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels,
                      self.yolo_conv_channels,
                      3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope)
            )
        

            # Detection Layers #
            #######################
        if self.use_conv:
            self.fc_yolo_layers = nn.Sequential(
                    nn.Conv2d(self.yolo_conv_channels, 5 * self.B + self.C, 1),
                )
        else:
            self.fc_yolo_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.conv_spatial_size * self.conv_spatial_size *
                            self.yolo_conv_channels,
                            self.yolo_fc_hidden_dim),
                    
                nn.LeakyReLU(self.leaky_relu_slope),
                nn.Dropout(self.yolo_fc_dropout_prob),
                nn.Linear(self.yolo_fc_hidden_dim,
                            self.S * self.S * (5 * self.B + self.C)),
                )
            
            def forward(self, X):
                out = self.features(X)
                out = self.conv_yolo_layers(out)
                out = self.fc_yolo_layers(out)
                if self.use_conv:
                    # Reshape conv output to Batch x S x S x (5B+C)
                    out = out.permute(0, 2, 3, 1)
                return 