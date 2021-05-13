import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.simple_extractor_dataset import SimpleFolderDataset
from utils.transforms import transform_logits

import io

class MyHandler(object):
    dataset_settings = {
        'atr': {
            'input_size': [512, 512],
            'num_classes': 18,
            'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                    'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
        }
    }

    def get_palette(self, num_cls):
        """ Returns the color map for visualizing the segmentation mask.
        Args:
            num_cls: Number of classes
        Returns:
            The color map
        """
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette
        
    def __init__(self):
        self.model = {}
        self.dataset = 'atr'
        self.num_classes = None
        self.input_size = None
        self.transform = None
        self.palette = None

        self.initialized = False
        self.context = None
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def initialize(self, context):
        self.initialized = True
        self.context = context
        self.num_classes = self.dataset_settings[self.dataset]['num_classes']
        self.input_size = self.dataset_settings[self.dataset]['input_size']
        label = self.dataset_settings[self.dataset]['label']
        print("Evaluating total class number {} with {}".format(self.num_classes, label))

        import networks
        self.model = networks.init_model('resnet101', num_classes=self.num_classes, pretrained=None)

        state_dict = torch.load('./checkpoints/lip.pth')['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.cuda()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
        self.palette = self.get_palette(self.num_classes)

    def preprocess(self, data):
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        dataset = SimpleFolderDataset(img=image, input_size=self.input_size, transform=self.transform)
        return DataLoader(dataset)

    def inference(self, dataloader):
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader)):
                image, meta = batch
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]

                output = self.model(image.cuda())
                upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
                parsing_result = np.argmax(logits_result, axis=2)
                output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                output_img.putpalette(self.palette)

                return output_img

    def potprocess(self, img):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='png')
        img_byte_arr = img_byte_arr.getvalue()
        return [img_byte_arr]

