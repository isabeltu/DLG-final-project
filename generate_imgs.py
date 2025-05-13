import os
import torch
from torchvision.utils import save_image
from model import Generator


def generate_samples(gen_path, output_dir, num_samples=100, nz=100):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator().to(device)
    state_dict = torch.load(gen_path, map_location=device)
    G.load_state_dict(state_dict)
    G.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, nz, 1, 1, device=device)
        imgs = G(noise)
        imgs = (imgs + 1.0) / 2.0

        for i in range(num_samples):
            filename = f"sample_{i}.png"
            save_image(imgs[i], os.path.join(output_dir, filename))

    print(f"Saved {num_samples} samples to {output_dir}")

generate_samples(
  gen_path='person_pths/netG_epoch_200.pth',
  output_dir='person_samples',
  num_samples=100,
  nz=100
)
