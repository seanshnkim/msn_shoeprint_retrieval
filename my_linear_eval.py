import torch
import logging

import torchvision
import torchvision.transforms as transforms

import os
import sys

from PIL import Image

from src.data_manager import ImageNet

from tqdm import tqdm

from transformers import ViTImageProcessor, ViTForImageClassification  
class LinearClassifier(torch.nn.Module):

    def __init__(self, dim, num_labels=1000, normalize=True):
        super(LinearClassifier, self).__init__()
        self.normalize = normalize
        self.norm = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.norm(x)
        if self.normalize:
            x = torch.nn.functional.normalize(x)
        return self.linear(x)

l2_normalize = True
normalize = l2_normalize
device = 'cuda:0'
num_classes = 1000
num_blocks = 1
model_name = 'deit_large'
emb_dim = 192 if 'tiny' in model_name else 384 if 'small' in model_name else 768 if 'base' in model_name else 1024 if 'large' in model_name else 1280
emb_dim *= num_blocks

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# r_path = 'checkpoint/msn_os_logs/vitl16_600ep.pth.tar'
# r_path = 'vitl16_600ep_fid300SSL.tar'
r_path = 'vitb4_300ep.pth.tar'

encoder = deit.__dict__[model_name]()
emb_dim = 192 if 'tiny' in model_name else 384 if 'small' in model_name else 768 if 'base' in model_name else 1024 if 'large' in model_name else 1280
emb_dim *= num_blocks
encoder.fc = None
encoder.norm = None

encoder.to(device)

checkpoint = torch.load(r_path, map_location='cpu')
pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
for k, v in encoder.state_dict().items():
    if k not in pretrained_dict:
        logger.info(f'key "{k}" could not be found in loaded state dict')
    elif pretrained_dict[k].shape != v.shape:
        logger.info(f'key "{k}" is of different shape in model and loaded state dict')
        pretrained_dict[k] = v
msg = encoder.load_state_dict(pretrained_dict, strict=False)
logger.info(f'loaded pretrained model with msg: {msg}')

# if linear_classifier is not None:
#     pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['classifier'].items()}
#     for k, v in linear_classifier.state_dict().items():
#         if k not in pretrained_dict:
#             logger.info(f'key "{k}" could not be found in loaded state dict')
#         elif pretrained_dict[k].shape != v.shape:
#             logger.info(f'key "{k}" is of different shape in model and loaded state dict')
#             pretrained_dict[k] = v
#     msg = linear_classifier.load_state_dict(pretrained_dict, strict=False)
#     logger.info(f'loaded pretrained model with msg: {msg}')
#     logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
#                 f'path: {r_path}')

encoder.eval()


val_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])

batch_size = 256
image_folder = 'imagenet'
root_path = '.'

root_path = params['root_path']
image_folder = params['image_folder']
BATCH_SIZE = params['batch_size']
NUM_WORKERS = params['num_workers']

val_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=val_transform,
        train=False)

# NOTE: batch_size=64 is the maximum batch size (approximately uses up to 7.9GB) that can fit in my GPU (RTX-2080 8GB)
val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS)


linear_classifier = LinearClassifier(emb_dim, num_classes, normalize).to(device)



model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
for data in val_data_loader:
    for img in os.listdir('imagenet/val/n04067472'):
        img_fpath = os.path.join('imagenet/val/n04067472', img)
        
        image = processor(Image.open(img_fpath).convert('RGB'), return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**image).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        print(model.config.id2label[predicted_label])
        
    # log_str = 'train:' if training else 'test:'
    # log_str = 'test:'
    # logger.info('[%d] %s (val: %.3f%%)'
    #             % (epoch + 1, log_str, val_top1))