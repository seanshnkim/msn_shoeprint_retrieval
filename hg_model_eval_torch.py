import torch
import torchvision.transforms as transforms

import os
import logging
from tqdm import tqdm
import yaml
import argparse
from time import localtime, strftime, time

from transformers import ViTForImageClassification, ViTMSNForImageClassification, ViTMSNPreTrainedModel
from transformers import ViTConfig

from src import deit
from src.data_manager import ImageNet

parser = argparse.ArgumentParser()

parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--load_checkpoint',
    help='if specified, load checkpoint',
    action='store_true')
parser.add_argument(
    '--load_from_hub',
    help='if specified, load checkpoint containing config.json and pytorch_model.bin',
    action='store_true')
parser.add_argument(
    '--msn',
    help='if specified, load (facebook) ViT-MSN model. Else, load (google) ViT model',
    action='store_true')

args = parser.parse_args()

with open(args.fname, 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

pretrained_model = params['model_name']
if args.load_checkpoint:
    if args.load_from_hub:
        # model = ViTMSNPreTrainedModel.from_pretrained(pretrained_model)
        model = ViTMSNForImageClassification.from_pretrained(pretrained_model)
    else:
        # FIXME: It depends on the model type and size
        config = ViTConfig.from_pretrained('facebook/vit-msn-large')
        # Instantiate the model architecture
        model = ViTForImageClassification(config)
        # Load your custom model weights
        model_weights = torch.load(pretrained_model, map_location=torch.device('cpu'))
        # Update this line according to how your weights are structured in the checkpoint
        model.load_state_dict(model_weights)
else:
    if args.msn:
        model = ViTMSNForImageClassification.from_pretrained(pretrained_model)
    else:
        # model = ViTForImageClassification.from_pretrained(pretrained_model)
        model = ViTForImageClassification.from_pretrained(pretrained_model, local_files_only=True)

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

# REVIEW: If you do not load model to GPU, you will get 
# "RuntimeError: Input type (c10:Half) and bias type (float) should be the same"
# Refer to this post: https://discuss.huggingface.co/t/is-transformers-using-gpu-by-default/8500/2
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
        outputs = model(inputs).logits

    # predicted_labels.shape = (batch_size,) = torch.Size([64])
    # labels.shape = (batch_size,) = torch.Size([64])
    total += inputs.shape[0]
    correct += outputs.max(dim=1).indices.eq(labels).sum().item()
    
    del inputs, labels, outputs
end_time = time()

logger.info(f"Total time: {end_time - start_time:.2f}s")
logger.info(f"Custom accuracy: {(correct/total) * 100:.2f}%")