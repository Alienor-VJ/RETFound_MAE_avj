import torch
import sys
sys.path.append('/Users/alienorvienne/Documents/Medecine/Residency/Studies/Articles/Cochin/Birdshot/')
from RETFound_MAE_avj.util.notebook_utils import show_image, prepare_model, run_one_image, imagenet_std, imagenet_mean
from torchvision import transforms
from torchvision.utils import make_grid
import models_vit
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import subprocess
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from util.pos_embed import interpolate_pos_embed
from timm.layers import trunc_normal_

def prepare_model_for_embedding(chkpt_dir, arch='vit_large_patch16'):
    # build model
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
    )
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.to('mps')
    return model

from torch.utils.data import DataLoader, Dataset

class MemoryEfficientDataset(Dataset):
    def __init__(self, patches, model_embedding):
        self.patches = patches
        self.model_embedding = model_embedding
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize each image to 224x224
        ])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        batch = self.patches[idx]
        return self.transform(batch)

def compute_embeddings(patches, model_embedding, batch_size=4):

    # Set environment variable for memory management
    torch.mps.empty_cache()
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Convert patches to float32
    patches = patches.float()

    # Create dataset and dataloader
    dataset = MemoryEfficientDataset(patches, model_embedding)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to('mps').float()
            try:
                latent = model_embedding.forward_features(batch.float())  # Process batch
                #print(f"Latent shape before squeeze: {latent.shape}")
                #latent = latent.squeeze(0)
                #print(f"Latent shape after squeeze: {latent.shape}")
                features.append(latent.detach().cpu())
            except RuntimeError as e:
                print(f"Memory error in batch: {e}")
                # Fallback: process even smaller batches or reduce batch size
                return compute_embeddings(patches, model_embedding, batch_size=batch_size//2)

    return torch.cat(features, dim=0)

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img

def reload_sprite_image(sprite_path):
    sprite_image = Image.open(sprite_path).convert("RGB")  # Load image
    transform = transforms.ToTensor()  # Convert to tensor [C, H, W]
    sprite_tensor = transform(sprite_image).unsqueeze(0)  # Add batch dimension [1, C, H, W]
    return sprite_tensor

import torchvision.transforms as transforms

def load_patches(image_path, patch_size):
    img = load_image(image_path)  # Load with PIL or OpenCV

    # resize to a round number of patches
    h, w, _ = np.array(img).shape
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    img = img.resize((new_w, new_h), Image.BILINEAR)
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # Convert to tensor (C, H, W)
    img_patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

    img_patches = img_patches.contiguous().view(3, -1, patch_size, patch_size)
    return np.transpose(img_patches, (1, 0, 2, 3))

# Generate sprite image
def save_sprite_image(patches, path):
    sprite = make_grid(patches, nrow=int(np.sqrt(len(patches))), pad_value=1)
    sprite_image = Image.fromarray((sprite.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    sprite_image.save(path)
    return path

model_embedding = prepare_model_for_embedding('/Users/alienorvienne/Documents/Medecine/Residency/Studies/Articles/Cochin/Birdshot/RETFound_MAE_avj/RETFound_cfp_weights.pth', 'vit_large_patch16')
model_embedding.eval()
print('model loaded')

features = {}
images_for_viz = []  # To store the patches for the sprite image

# Define the directory containing the images
image_dir = "/Users/alienorvienne/Documents/Medecine/Residency/Studies/Articles/Cochin/Birdshot/RETFound_MAE_avj/pic/train/birdshot_nospot/"
uploaded = {os.path.basename(f): os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
print(f"Found {len(uploaded)} images.")

with torch.no_grad():

  for name in tqdm(list(uploaded.keys())):
      full_path = os.path.join(image_dir, name)
      patches = load_patches(full_path, 112)

      images_for_viz += patches.clone().detach().cpu()  # Ensure patches are on CPU

      # Ensure broadcasting works correctly
      imagenet_mean = torch.tensor(imagenet_mean).view(1, 3, 1, 1).to(patches.device)
      imagenet_std = torch.tensor(imagenet_std).view(1, 3, 1, 1).to(patches.device)

      patches = patches - imagenet_mean
      patches = patches / imagenet_std

      latent_feature = compute_embeddings(patches, model_embedding, 4)

      features[name] = latent_feature.detach().cpu().numpy()

      sprite_path = "sprite_image.png"
      sprite_path = save_sprite_image(images_for_viz, sprite_path)

      import pandas as pd
      from torch.utils.tensorboard import SummaryWriter

      # Create a SummaryWriter instance
      logdir = 'tensorboard'
      writer = SummaryWriter(logdir)

      # Convert features into a tensor and metadata for visualization
      embeddings = []
      metadata = []

      for image_name, crops_embeddings in features.items():
          for crop_idx, crop_embedding in enumerate(crops_embeddings):
              embeddings.append(torch.tensor(crop_embedding))
              metadata.append(f"{image_name}_crop_{crop_idx}")

      # Stack embeddings into a single tensor
      embeddings_tensor = torch.stack(embeddings)

      print(f"Embeddings shape: {embeddings_tensor.shape}")
      print(f"Metadata length: {len(metadata)}")

      # Add embeddings to TensorBoard projector
      writer.add_embedding(
          embeddings_tensor,  # Embeddings tensor
          metadata=metadata,  # Metadata (e.g., image names)
          tag="features_projector",  # Tag for the visualization
          label_img=torch.stack([torch.tensor(im) / 255 for im in images_for_viz]),
      )

      # Close the writer
      writer.close()


      def start_tensorboard(log_dir):
          # Ensure the log directory exists
          os.makedirs(log_dir, exist_ok=True)
          # Start TensorBoard
          subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])
          print("TensorBoard is running on http://localhost:6006")

      # Start TensorBoard
      start_tensorboard('/Users/alienorvienne/Documents/Medecine/Residency/Studies/Articles/Cochin/Birdshot/RETFound_MAE_avj/tensorboard/')

      # you need to hit refresh on the tensorboard UI (top right), on the first time