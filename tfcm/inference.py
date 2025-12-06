import os
import torch
import requests
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from .model import ColorizerV14
from .utils import rgb_to_lab, lab_to_rgb, dwt_init, idwt_init

# Update this URL to point to your actual Github Release file
WEIGHTS_URL = "https://github.com/Juste-Leo2/TFCM/releases/download/v1.0/tfcm.pth"
WEIGHTS_NAME = "tfcm.pth"

class Colorizer:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ColorizerV14().to(self.device)
        self.load_weights()
        self.model.eval()
        
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),      
            transforms.CenterCrop(256),  
            transforms.ToTensor()
        ])

    def load_weights(self):
        path = os.path.join(os.path.expanduser("~/.cache/tfcm"), WEIGHTS_NAME)
        if not os.path.exists(path):
            print(f"Weights not found. Downloading to {path}...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            response = requests.get(WEIGHTS_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
                for data in response.iter_content(1024):
                    bar.update(len(data))
                    f.write(data)
        
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        except Exception as e:
            print(f"Error loading weights: {e}")

    def process(self, image_path):
        if isinstance(image_path, str):
            img_pil = Image.open(image_path).convert('RGB')
        else:
            img_pil = image_path.convert('RGB') # Assume PIL Image
            
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            lab = rgb_to_lab(img_tensor)
            L = lab[:, 0:1]
            gray_3ch = torch.cat([L, L, L], dim=1)
            
            # Forward
            feats = self.model.brain(self.norm(gray_3ch))[0]
            if feats.shape[2:] != (L.shape[2]//8, L.shape[3]//8): # Handle size mismatch if any
                 feats = torch.nn.functional.interpolate(feats, size=(L.shape[2]//8, L.shape[3]//8))

            L_dwt = dwt_init(L)
            e1 = self.model.w_enc1(L_dwt)
            e2 = self.model.w_enc2(e1)
            e3 = self.model.w_enc3(e2)
            
            fake_ab_dwt = self.model(feats, e1, e2, e3)
            
            fake_a = idwt_init(fake_ab_dwt[:, 0:4])
            fake_b = idwt_init(fake_ab_dwt[:, 4:8])
            fake_rgb = lab_to_rgb(L, torch.cat([fake_a, fake_b], dim=1))

        return transforms.ToPILImage()(fake_rgb[0].cpu())