import logging
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import yaml
from time import strftime, localtime
import numpy as np
import os
import argparse

start_time_stamp = strftime("%m-%d_%H%M", localtime())
cur_fname = os.path.basename(__file__).rstrip('.py')
log_save_dir = os.path.join('logs', f'{cur_fname}_{start_time_stamp}.log')
logging.basicConfig(filename=log_save_dir, level=logging.INFO)
logger = logging.getLogger()

def main(args):
    with open(args.fname, 'r') as file:
        cfg = yaml.safe_load(file)

    root_path = "FID-300"
    device = args.devices
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model.to(device)
    
    # google_vit-base-patch16-224
    save_dir = f'np_features_{args.model_name}'
    os.makedirs(os.path.join(save_dir, "query"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "ref"), exist_ok=True)
    
    img_list = os.listdir(os.path.join(root_path, cfg['data_path']['query_train']))
    for img in img_list:
        image = Image.open(os.path.join(root_path, cfg['data_path']['query_train'], img)).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs.to(device)
        outputs = model(**inputs)
        
        query_idx_str = img.split('.')[0]
        save_fname = os.path.join(save_dir, "query", f"{query_idx_str}.npy")
        saved_tensor = outputs.logits.cpu().detach().numpy()
        np.save(save_fname, saved_tensor)
        
    img_list = os.listdir(os.path.join(root_path, cfg['data_path']['query_test']))
    for img in img_list:
        image = Image.open(os.path.join(root_path, cfg['data_path']['query_test'], img)).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs.to(device)
        outputs = model(**inputs)
        
        query_idx_str = img.split('.')[0]
        save_fname = os.path.join(save_dir, "query", f"{query_idx_str}.npy")
        saved_tensor = outputs.logits.cpu().detach().numpy()
        np.save(save_fname, saved_tensor)
        
    img_list = os.listdir(os.path.join(root_path, cfg['data_path']['ref_test']))
    for img in img_list:
        image = Image.open(os.path.join(root_path, cfg['data_path']['ref_test'], img)).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs.to(device)
        outputs = model(**inputs)
        
        query_idx_str = img.split('.')[0]
        save_fname = os.path.join(save_dir, "ref", f"{query_idx_str}.npy")
        saved_tensor = outputs.logits.cpu().detach().numpy()
        np.save(save_fname, saved_tensor)


if __name__ == "__main__":
    '''
    args format example:
    
    "args": [
            "--fname",
            "configs/eval/test_custom.yaml",
            "--devices",
            "cuda:0"
            ]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, help="path to config file")
    parser.add_argument("--devices", type=str, help="device to use")
    parser.add_argument("--model_name", type=str, help="model name")
    args = parser.parse_args()
    
    main(args)
