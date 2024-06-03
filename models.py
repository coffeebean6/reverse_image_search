import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

class Resnet50:
    def __init__(self) -> None:
        # 从pytorch中获取预训练模型，进行推理
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.__init_transforms()

    def __init_transforms(self):
        # 图像预处理
        self.__image_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # 兼容灰度图
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_last_conv_layer_features(self, image):
        # 给定图片，取fc前最后一层的特征，2048维
        with torch.no_grad():
            x = self.model.conv1(image)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1) # torch.Size([2048])
            size = x.size()
            feat = x.numpy()   # 转为(2048,)
            return feat.reshape(size[1],)
    
    def extract_feature(self, image_path):
        # 给定path，抽取fc前最后一层的特征，2048维
        image = Image.open(image_path)
        image = self.__image_preprocess(image)
        image = image.unsqueeze(0)
        return self.get_last_conv_layer_features(image)
    
    def batch_extract_features(self, image_paths:list) -> list:
        # 批给定path，抽取fc前最后一层的特征，2048维
        features = []
        for image_path in image_paths:
            image = Image.open(image_path)
            image = self.__image_preprocess(image)
            image = image.unsqueeze(0)  # 添加一个批次维度
            feature = self.get_last_conv_layer_features(image) 
            features.append(feature)
        return features

    """遍历父目录下的所有图片，抽取特征"""    
    def batch_extract_features_by_parent_path(self, parent_path:str) -> list:
        extensions = ['.jpg', '.jpeg', '.png'] #, '.gif', '.bmp', '.tiff', '.tif', '.webp'
        image_paths = []
        for root, dirs, files in os.walk(parent_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths, self.batch_extract_features(image_paths)
