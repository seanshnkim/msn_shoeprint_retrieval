import torch
import torchvision.transforms as transforms

import os
import logging
from tqdm import tqdm
import yaml
import argparse
from time import localtime, strftime, time

from transformers import ViTMAEModel, ViTMAEConfig
from src.data_manager import ImageNet


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

args = parser.parse_args()

with open(args.fname, 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

pretrained_model = params['model_name']
# model = ViTMAEModel.from_pretrained(pretrained_model)

configuration = ViTMAEConfig()
model = ViTMAEModel(configuration)

# checkpoint = torch.load(pretrained_model, map_location='cpu')

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
emb_dim = 768
normalize = True
linear_classifier = LinearClassifier(emb_dim, num_classes, normalize).to(device)
model.to(device)

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
        # outputs = model.forward_blocks(inputs, num_blocks)
        outputs = model.forward(inputs)
        # if ViTMAEModel is used, outputs is not a tensor, it is an instance of ViTMAEModelOutput
        # Why? outputs.shape = [32, 50, 768]
    outputs_flattened = outputs.last_hidden_state.mean(dim=1)
    predictions = linear_classifier(outputs_flattened)
    total += inputs.shape[0]
    correct += predictions.max(dim=1).indices.eq(labels).sum().item()
    
    del inputs, labels, outputs
end_time = time()

logger.info(f"Total time: {end_time - start_time:.2f}s")
logger.info(f"Custom accuracy: {(correct/total) * 100:.2f}%")