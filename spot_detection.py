from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image
import numpy as np
import matplotlib.pyplot as plt
import torch
import models_vit
from PIL import Image

# Load the custom VisionTransformer model
model = models_vit.vit_large_patch16(
    num_classes=2,
    drop_path_rate=0.2,
    global_pool='avg',  # or "max" depending on your preference
)

# Load and preprocess the image
image = Image.open('/Users/alienorvienne/Documents/Medecine/Residency/Studies/Articles/Cochin/Birdshot/code/RETFound_MAE_avj/pic/val/birdshot_spot/cropped_B_s_test9.jpg').convert('RGB')

# Resize the image to match the input size of the model (224x224)
image = image.resize((224, 224))
print(f"Image shape: {image.size}")

# Preprocess the image (should return tensor of shape [1, 3, 224, 224])
input_tensor = preprocess_image(np.array(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print(f"Input tensor shape: {input_tensor.shape}")  # This should print: [1, 3, 224, 224]

# Define the target layers (e.g., the last block of your model)
target_layers = [model.blocks[-1].attn]

print(f"target_layers : {target_layers}")

# Use Grad-CAM with a context manager
with GradCAM(model=model, target_layers=target_layers) as grad_cam:
    # Generate Grad-CAM heatmap
    grayscale_cam = grad_cam(input_tensor=input_tensor, targets=None)

    # Visualize the heatmap overlay on the original image
    heatmap = grayscale_cam[0, :]
    plt.imshow(image, alpha=0.5)  # Show the original image
    plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Overlay the heatmap
    plt.colorbar()
    plt.show()
