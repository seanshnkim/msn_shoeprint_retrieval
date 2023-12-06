import torch
import torchvision.transforms as transforms

import os
import logging
from tqdm import tqdm
import yaml
import argparse
from time import localtime, strftime, time

from src import deit
from src.data_manager import ImageNet

from transformers import ViTMSNModel

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
    
    
parser = argparse.ArgumentParser()

parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--deit_type', type=str,
    help='default deit model type. If using MSN-large, set to "deit_large"',
    default='deit_base')

args = parser.parse_args()

with open(args.fname, 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

pretrained_model = params['model_name']

# load from checkpoint
encoder = deit.__dict__[args.deit_type]()
emb_dim = 192 if 'tiny' in args.deit_type else 384 if 'small' in args.deit_type else 768 if 'base' in args.deit_type else 1024 if 'large' in args.deit_type else 1280
num_blocks = 1
emb_dim *= num_blocks
encoder.fc = None
encoder.norm = None

# checkpoint = torch.load(pretrained_model, map_location='cpu')
checkpoint = ViTMSNModel.from_pretrained(pretrained_model)
# pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
# for k, v in encoder.state_dict().items():
#     if k not in pretrained_dict:
#         print(f'key "{k}" could not be found in loaded state dict')
#     elif pretrained_dict[k].shape != v.shape:
#         print(f'key "{k}" is of different shape in model and loaded state dict')
#         pretrained_dict[k] = v

# encoder.load_state_dict(pretrained_dict, strict=False)

val_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = 1000
normalize = True
linear_classifier = LinearClassifier(emb_dim, num_classes, normalize).to(device)
encoder.to(device)

time_stamp = strftime("%m%d-%H%M", localtime())
model_name = pretrained_model.replace('/', '_')
log_fname = f'eval_{model_name}_{time_stamp}.log'
logging.basicConfig(filename=os.path.join('logs', log_fname), level=logging.INFO,\
                    format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

logger.info(f"Based on Hugging Face Transformer model: {pretrained_model}")
logger.info(f"Dataset location: {os.path.join(root_path, image_folder)}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Number of workers: {NUM_WORKERS}")
logger.info(f"Number of images: {len(val_dataset)}")

start_time = time()
total, correct = 0, 0
for data in tqdm(val_data_loader):
    with torch.cuda.amp.autocast(enabled=True):
        inputs, labels = data[0].to(device), data[1].to(device)
        # outputs.shape = (batch_size, num_labels) = torch.Size([64, 1000])
        outputs = encoder.forward_blocks(inputs, num_blocks)
    outputs = linear_classifier(outputs)
    total += inputs.shape[0]
    correct += outputs.max(dim=1).indices.eq(labels).sum().item()
    
    del inputs, labels, outputs
end_time = time()

logger.info(f"Total time: {end_time - start_time:.2f}s")
logger.info(f"Custom accuracy: {(correct/total) * 100:.2f}%")