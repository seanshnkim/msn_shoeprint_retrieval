from transformers import ViTMSNModel, ViTMSNConfig, AutoImageProcessor
import torch
from PIL import Image
import requests

# Initializing a ViT MSN vit-msn-base style configuration
configuration = ViTMSNConfig()

# Initializing a model from the vit-msn-base style configuration
model = ViTMSNModel(configuration)

# Accessing the model configuration
model_configuration = model.config

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-large")
model = ViTMSNModel.from_pretrained("facebook/vit-msn-large")
inputs = image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state