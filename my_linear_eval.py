import torch
import logging

import torchvision
import torchvision.transforms as transforms

import os
import sys

from PIL import Image

import src.deit as deit
from src.sgd import SGD
from src.utils import (
    WarmupCosineSchedule
)

from tqdm import tqdm

from transformers import ViTImageProcessor, ViTForImageClassification

class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        transform=None,
        train=True,
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        """

        suffix = 'train/' if train else 'val/'
        data_path = None

        if data_path is None:
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')
        
        
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

start_epoch = 0
num_epochs = 5

def init_data(
    transform,
    batch_size,
    pin_mem=True,
    num_workers=8,
    root_path=None,
    image_folder=None,
    training=True,
    drop_last=True
):

    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training)

    logger.info('ImageNet dataset created')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers)
    logger.info('ImageNet unsupervised data loader created')

    return data_loader


transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])

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
log_freq = 10

data_loader = init_data(
        transform=transform,
        batch_size=batch_size,
        root_path=root_path,
        image_folder=image_folder,
        training=True)


val_data_loader = init_data(
                    transform=val_transform,
                    batch_size=batch_size,
                    image_folder=image_folder,
                    root_path=root_path,
                    training=False,
                    drop_last=False)

criterion = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=True)

linear_classifier = LinearClassifier(emb_dim, num_classes, normalize).to(device)

param_groups = [
        {'params': (p for n, p in linear_classifier.named_parameters()
                    if ('bias' not in n) and ('bn' not in n) and len(p.shape) != 1)},
        {'params': (p for n, p in linear_classifier.named_parameters()
                    if ('bias' in n) or ('bn' in n) or (len(p.shape) == 1)),
         'weight_decay': 0}
    ]

weight_decay = 0.0
# ref_lr = 6.4
ref_lr = 0.001
warmup_epochs = 0
iterations_per_epoch = len(data_loader)

optimizer = SGD(
        param_groups,
        nesterov=True,
        weight_decay=weight_decay,
        momentum=0.9,
        lr=ref_lr)
scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_epochs*iterations_per_epoch,
        start_lr=ref_lr,
        ref_lr=ref_lr,
        T_max=num_epochs*iterations_per_epoch)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
for epoch in tqdm(range(start_epoch, num_epochs)):
    
    # def train_step():
    #     top1_correct, top5_correct, total = 0, 0, 0
    #     for i, data in enumerate(data_loader):
    #         with torch.cuda.amp.autocast(enabled=True):
    #             inputs, labels = data[0].to(device), data[1].to(device)
    #             with torch.no_grad():
    #                 outputs = encoder.forward_blocks(inputs, num_blocks)
    #         outputs = linear_classifier(outputs)
    #         loss = criterion(outputs, labels)
    #         total += inputs.shape[0]
    #         top5_correct += float(outputs.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum())
    #         top1_correct += float(outputs.max(dim=1).indices.eq(labels).sum())
    #         top1_acc = 100. * top1_correct / total
    #         top5_acc = 100. * top5_correct / total
    #         # if training:
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #         scheduler.step()
    #         optimizer.zero_grad()
    #         if i % log_freq == 0:
    #             logger.info('[%d, %5d] %.3f%% %.3f%% (loss: %.3f)'
    #                         % (epoch + 1, i, top1_acc, top5_acc, loss))
    #     return 100. * top1_correct / total
    
    
    # def val_step():
    #     top1_correct, total = 0, 0
    #     for i, data in enumerate(val_data_loader):
    #         with torch.cuda.amp.autocast(enabled=True):
    #             inputs, labels = data[0].to(device), data[1].to(device)
    #             outputs = encoder.forward_blocks(inputs, num_blocks)
    #         outputs = linear_classifier(outputs)
    #         total += inputs.shape[0]
    #         top1_correct += outputs.max(dim=1).indices.eq(labels).sum()
    #         top1_acc = 100. * top1_correct / total
    #         logger.info(f'Epoch: {epoch}, {i}th batch: {top1_acc}')

    #     logger.info('[%d, %5d] %.3f%%' % (epoch + 1, i, top1_acc))
    #     return top1_acc

    # train_top1 = 0.
    # train_top1 = train_step()
    # with torch.no_grad():
    #     val_top1 = val_step()
        
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