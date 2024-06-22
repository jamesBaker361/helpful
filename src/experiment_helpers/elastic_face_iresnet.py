import torch
from torch import nn
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import PILToTensor,Compose,Resize
from facenet_pytorch import MTCNN, InceptionResnetV1,extract_face

#credits to https://github.com/fdbtrs/ElasticFace/tree/main

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100']

def rescale_around_zero(image_tensor:torch.Tensor)->torch.Tensor:
    max_value=torch.max(image_tensor)
    min_value=torch.min(image_tensor)
    if max_value >1 and min_value>=0: #in range 0-255
        image_tensor=(image_tensor -128)/128
    elif min_value>=0 and max_value<=1.0: #in range 0-1
        image_tensor=(image_tensor -0.5)/0.5
    return image_tensor

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1,use_se=False):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride
        self.use_se=use_se
        if (use_se):
         self.se_block=SEModule(planes,16)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if(self.use_se):
            out=self.se_block(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, use_se=False):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.use_se=use_se
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2 ,use_se=self.use_se)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],use_se=self.use_se)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1] ,use_se=self.use_se)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2] ,use_se=self.use_se)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout =nn.Dropout(p=dropout, inplace=True) # 7x7x 512
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,use_se=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation,use_se=use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,use_se=use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)

class SquarePad:
    def __call__(self, image):
        if isinstance(image,Image.Image):
            w, h = image.size
        elif isinstance(image,torch.Tensor):
            c,w,h=image.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')
    
class CustomPILToTensor:
    def __call__(self, image) -> torch.Tensor:
        if isinstance(image,torch.Tensor):
            return image
        elif isinstance(image,Image.Image):
            return PILToTensor()(image)

def get_iresnet_model(
        device,
        num_features=512,
        weight_path="/scratch/jlb638/faces/elastic_face_arc.pth",
        net=iresnet100
)->IResNet:
    state_dict=torch.load(weight_path,map_location=device)
    model=net(num_features=num_features)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_iresnet(
  images:list,
  device:str
)->torch.Tensor:
    return_images=[]
    composition=Compose([
        SquarePad(),
        Resize((112,112)),
        CustomPILToTensor()
    ])
    images=[composition(i).to(torch.float32) for i in images]
    images=[rescale_around_zero(i) for i in images]
    #print('processed min max',min_value,max_value)
    return torch.stack(images).to(device)

def get_face_embedding(images:list,mtcnn:MTCNN, iresnet:IResNet,margin:int)->torch.tensor:
    face_tensors=[]
    for img in images:
        boxes, probs=mtcnn.detect(img)
        if boxes is not None:
            extracted_face_tensor=extract_face(img,boxes[0],112,margin)
        else:
            extracted_face_tensor=torch.ones((3,112,112)) #if we dont find any faces :(
        face_tensors.append(extracted_face_tensor)

    iresnet_input=preprocess_iresnet(face_tensors,mtcnn.device)
    return iresnet(iresnet_input)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path="/scratch/jlb638/faces/elastic_face_arc.pth"
    state_dict=torch.load(weight_path,map_location=device)
    embedding_size=512
    model=iresnet100(num_features=embedding_size)
    model.load_state_dict(state_dict)
    model.eval()
    from PIL import Image
    from torchvision.transforms import PILToTensor,Compose,Resize
    composition=Compose([
        SquarePad(),
        Resize((112,112)),
        PILToTensor()
    ])
    pil_img=Image.open("gnome.jpg")
    '''pt_img=composition(pil_img) #this will be unit8 ~ [0,255] we need to convert it to [-1,1] for iresnet
    pt_img=pt_img.to(torch.float32).unsqueeze(0)
    pt_img= ((pt_img / 255) - 0.5) / 0.5
    embedding=model(pt_img)'''
    images=preprocess_iresnet([pil_img])
    embedding=model(images)
    print(embedding)

    mtcnn = MTCNN()
    embedding=get_face_embedding([pil_img],mtcnn,model,10)
    print(embedding)