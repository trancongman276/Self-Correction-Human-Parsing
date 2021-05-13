import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

import logging

from torch.autograd import Variable
from torchvision import transforms

logger = logging.getLogger(__name__)


class HUMANPARSING(object):
    """
    HUMANPARSING handler class. This handler takes a greyscale image
    and returns the digit in that image.
    """
    dataset_settings = {
        'lip': {
            'input_size': [473, 473],
            'num_classes': 20,
            'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                    'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                    'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
        },
        'atr': {
            'input_size': [512, 512],
            'num_classes': 18,
            'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                    'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
        },
        'pascal': {
            'input_size': [512, 512],
            'num_classes': 7,
            'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
        }
    }

    checkpoints = {
        'lip': './checkpoints/lip.pth',
        'atr': './checkpoints/atr.pth',
        'pascal': './checkpoints/pascal.pth',
    }

    def get_palette(num_cls):
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
        self.mapping = None
        self.device = None
        os.environ["CUDA_VISIBLE_DEVICES"] = 1

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a PIL image for a MNIST model,
         returns an Numpy array
        """
        image = data[0].get("data")
        dataset = data[0].get("dataset")

        print(f"Image: {image}")
        print(f"Dataset: {dataset}")
        if image is None:
            image = data[0].get("body")

        # mnist_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])
        # image = Image.open(io.BytesIO(image))
        # image = mnist_transform(image)
        # return image
        # import networks
        # for dataset in self.dataset_settings:
        #     num_classes = self.dataset_settings[dataset]['num_classes']
        #     input_size = self.dataset_settings[dataset]['input_size']
        #     label = self.dataset_settings[dataset]['label']
        #     print("Evaluating total class number {} with {}".format(num_classes, label))
        
        #     base_model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
        #     state_dict = torch.load('checkpoints/'+dataset)['state_dict']
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in state_dict.items():
        #         name = k[7:]  # remove `module.`
        #         new_state_dict[name] = v
        #     base_model.load_state_dict(new_state_dict)
        #     base_model.cuda()
        #     base_model.eval()
        #     self.model[dataset] = base_model

        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        # ])

        # palette = self.get_palette(num_classes)
        # # get data
        # from datasets.simple_extractor_dataset import SimpleFolderDataset
        # dataset = SimpleFolderDataset(img=data, input_size=input_size, transform=transform)
        # dataloader = DataLoader(dataset)
        pass

    def inference(self, img, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Convert 2D image to 1D vector
        # with torch.no_grad():
        #     for _, batch in enumerate(tqdm(d)):
        #         image, meta = batch
        #         # img_name = meta['name'][0]
        #         c = meta['center'].numpy()[0]
        #         s = meta['scale'].numpy()[0]
        #         w = meta['width'].numpy()[0]
        #         h = meta['height'].numpy()[0]

        #         output = model(image.cuda())
        #         upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        #         upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        #         upsample_output = upsample_output.squeeze()
        #         upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

        #         logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
        #         parsing_result = np.argmax(logits_result, axis=2)
        #         parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
        #         output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        #         output_img.putpalette(palette)
        #         output_img.save(parsing_result_path)
        #         if args.logits:
        #             logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
        #             np.save(logits_result_path, logits_result)
        # img = np.expand_dims(img, 0)
        # img = torch.from_numpy(img)

        # self.model.eval()
        # inputs = Variable(img).to(self.device)
        # outputs = self.model.forward(inputs)

        # _, y_hat = outputs.max(1)
        # predicted_idx = str(y_hat.item())
        # return [predicted_idx]
        pass

    def postprocess(self, inference_output):
        pass


_service = HUMANPARSING()


def handle(data, context):
    # if not _service.initialized:
    #     _service.initialize(context)
    print(f"DATA: {data}")
    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data