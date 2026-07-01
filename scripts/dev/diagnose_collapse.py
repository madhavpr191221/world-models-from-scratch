"""
Quick diagnostic: small model on real STL-10 data, fast iteration to
debug the L_var collapse issue.
"""
import torch
from jepa_world_models.contrastive_learning import Projector
from jepa_world_models.contrastive_learning.encoders.vit import ViTEncoder
from jepa_world_models.vic_reg_loss.loss import VICRegLoss
from jepa_world_models.data.stl10 import STL10Unlabeled
from jepa_world_models.contrastive_learning.augmentations import vicreg_augmentation
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = ViTEncoder(depth=2, num_heads=2).to(device)  # small for fast iteration
projector = Projector(in_dim=192, proj_dim=128).to(device)
loss_fn = VICRegLoss(lambda_=25.0, mu=25.0, nu=1.0, gamma=1.0, eps=1e-4)
opt = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=1e-4)

transform = vicreg_augmentation(image_size=96)
dataset = STL10Unlabeled(root="data/archive/unlabeled_images", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

encoder.train()
projector.train()

print("Diagnostic run: small model, real STL-10 data\n")
loader_iter = iter(loader)
for step in range(150):
    images_a, images_b = next(loader_iter)
    images_a, images_b = images_a.to(device), images_b.to(device)

    opt.zero_grad()
    z_a = projector(encoder(images_a))
    z_b = projector(encoder(images_b))
    out = loss_fn(z_a, z_b)
    out.total.backward()
    opt.step()

    if step % 10 == 0:
        print(f"step {step:>4} | L_total={out.total.item():>9.4f} | "
              f"L_inv={out.inv.item():>8.4f} | L_var={out.var.item():.4f} | "
              f"L_cov={out.cov.item():.4f}")