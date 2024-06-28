import torch
import torch.nn as nn
import torchvision.transforms as transforms


def cnn_preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

def vit_preprocess(image, patch_size=16):
    def patchify(img, patch_size):
        p = patch_size
        h, w = img.shape[-2:]
        patches = img.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(3, -1, p, p)
        return patches.permute(1, 0, 2, 3)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(image)
    patches = patchify(img_tensor.unsqueeze(0), patch_size)
    
    patch_embed = nn.Linear(patch_size * patch_size * 3, 768)
    embeddings = patch_embed(patches.flatten(1))
    
    num_patches = embeddings.shape[0]
    pos_encoding = nn.Parameter(torch.zeros(1, num_patches, 768))
    
    embeddings = embeddings + pos_encoding.squeeze(0)
    
    return embeddings

random_image = torch.rand(3, 640, 640)

cnn_processed = cnn_preprocess(random_image)

vit_processed = vit_preprocess(random_image)
