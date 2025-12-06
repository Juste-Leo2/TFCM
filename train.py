import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIG V14 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 20 
EPOCHS = 10     
IMG_SIZE = 256 

# Chemins
TRAIN_DIR = "./data/coco/train2017"
VAL_DIR   = "./data/coco/val2017" 

SAVE_PATH = "flash_colorizer_v14_large.pth"
PREVIEW_DIR = "./previews_v14"

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --- UTILS (INCHANGÉS) ---
def rgb_to_lab(image):
    mask = image > 0.04045
    image = torch.where(mask, torch.pow((image+0.055)/1.055, 2.4), image/12.92)
    r,g,b = image[:,0], image[:,1], image[:,2]
    x = 0.412453*r + 0.357580*g + 0.180423*b
    y = 0.212671*r + 0.715160*g + 0.072169*b
    z = 0.019334*r + 0.119193*g + 0.950227*b
    x=x/0.95047; y=y/1.0; z=z/1.08883
    xyz = torch.stack([x,y,z], dim=1)
    mask = xyz > 0.008856
    xyz = torch.where(mask, torch.pow(xyz, 1/3), 7.787*xyz + 16/116)
    L = 116*xyz[:,1]-16; a = 500*(xyz[:,0]-xyz[:,1]); b = 200*(xyz[:,1]-xyz[:,2])
    return torch.stack([L/100.0, a/110.0, b/110.0], dim=1)

def lab_to_rgb(L, ab):
    L = L * 100.0; a = ab[:, 0:1, :, :] * 110.0; b = ab[:, 1:2, :, :] * 110.0
    y = (L + 16) / 116; x = a / 500 + y; z = y - b / 200
    xyz = torch.cat([x, y, z], dim=1)
    mask = xyz > 0.2068966
    xyz = torch.where(mask, torch.pow(xyz, 3), (xyz - 16/116) / 7.787)
    x_c = xyz[:, 0:1, :, :] * 0.95047; y_c = xyz[:, 1:2, :, :] * 1.00000; z_c = xyz[:, 2:3, :, :] * 1.08883
    r = 3.240481 * x_c - 1.537151 * y_c - 0.498536 * z_c
    g = -0.969256 * x_c + 1.875991 * y_c + 0.041556 * z_c
    b = 0.055646 * x_c - 0.204041 * y_c + 1.057311 * z_c
    rgb = torch.cat([r, g, b], dim=1)
    mask = rgb > 0.0031308
    rgb = torch.where(mask, 1.055 * torch.pow(rgb, 1/2.4) - 0.055, 12.92 * rgb)
    return rgb.clamp(0, 1)

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2; x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]; x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]; x4 = x02[:, :, :, 1::2]
    return torch.cat((x1+x2+x3+x4, -x1-x2+x3+x4, -x1+x2-x3+x4, x1-x2-x3+x4), 1)

def idwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_channel = in_channel // 4
    x1 = x[:, 0:out_channel, :, :] / 2; x2 = x[:, out_channel:out_channel*2, :, :] / 2
    x3 = x[:, out_channel*2:out_channel*3, :, :] / 2; x4 = x[:, out_channel*3:out_channel*4, :, :] / 2
    h = torch.zeros([in_batch, out_channel, in_height*r, in_width*r]).to(x.device)
    h[:,:,0::2,0::2]=x1-x2-x3+x4; h[:,:,1::2,0::2]=x1-x2+x3-x4
    h[:,:,0::2,1::2]=x1+x2-x3-x4; h[:,:,1::2,1::2]=x1+x2+x3+x4
    return h

# --- DATASET ---
class HDDataset(Dataset):
    def __init__(self, folder_path, is_train=True):
        print(f"🔍 Scan de {folder_path}...")
        self.files = glob.glob(os.path.join(folder_path, "*.jpg"))
        if len(self.files) == 0: self.files = glob.glob(os.path.join(folder_path, "*.JPG"))
        self.is_train = is_train
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize(IMG_SIZE), 
                transforms.RandomCrop(IMG_SIZE), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
            ])
        print(f"✅ {len(self.files)} images trouvées.")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img)
        except: return torch.zeros(3, IMG_SIZE, IMG_SIZE)

# --- MODELE V14 ---
class ColorizerV14(nn.Module):
    def __init__(self):
        super(ColorizerV14, self).__init__()
        
        # 1. UPGRADE: MobileNetV3 LARGE (Nom corrigé)
        print("🧠 Chargement de MobileNetV3 LARGE...")
        self.brain = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True, out_indices=[2])
        
        # Récupération automatique du nombre de channels
        channels = self.brain.feature_info.channels()[0]
        print(f"ℹ️ MobileNet Features Channels: {channels}")

        # 2. STRATEGIE: Gel partiel (50% des blocs finaux débloqués)
        for p in self.brain.parameters(): p.requires_grad = False
        
        total_blocks = len(self.brain.blocks)
        half_point = total_blocks // 2
        print(f"🔓 Déblocage des blocks {half_point} à {total_blocks}")
        
        for i, block in enumerate(self.brain.blocks):
            if i >= half_point:
                for p in block.parameters(): p.requires_grad = True

        # Encodeur DWT
        self.w_enc1 = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), nn.LeakyReLU(0.2)) 
        self.w_enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2)) 
        self.w_enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2)) 
        
        # Adaptation dynamique
        self.brain_conv = nn.Sequential(nn.Conv2d(channels, 128, 1, 1, 0), nn.Sigmoid())
        
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32, 8, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, img_gray_3ch, L_dwt):
        feats = self.brain(img_gray_3ch)[0] 
        e1 = self.w_enc1(L_dwt)
        e2 = self.w_enc2(e1)
        e3 = self.w_enc3(e2)
        
        if feats.shape[2:] != e3.shape[2:]: 
            feats = torch.nn.functional.interpolate(feats, size=e3.shape[2:])
            
        gate = self.brain_conv(feats)
        fusion = e3 * (1 + gate)
        b = self.bottleneck(fusion)
        d3 = self.dec3(torch.cat([self.up3(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return self.tanh(self.final(d2))

# --- VISUALISATION ---
def visualize(epoch, model, loader):
    model.eval()
    if not os.path.exists(PREVIEW_DIR): os.makedirs(PREVIEW_DIR)
    try: imgs = next(iter(loader))
    except: return
    imgs = imgs[:4].to(DEVICE)
    
    with torch.no_grad():
        lab = rgb_to_lab(imgs)
        L = lab[:, 0:1]
        
        # Sécurité Data Leak
        gray_3ch = torch.cat([L, L, L], dim=1)
        
        fake_ab_dwt = model(NORMALIZE(gray_3ch), dwt_init(L))
        fake_a = idwt_init(fake_ab_dwt[:, 0:4])
        fake_b = idwt_init(fake_ab_dwt[:, 4:8])
        fake_rgb = lab_to_rgb(L, torch.cat([fake_a, fake_b], dim=1))
    
    plt.figure(figsize=(10, 8))
    for i in range(len(imgs)):
        pred = fake_rgb[i].permute(1, 2, 0).cpu().numpy()
        orig = imgs[i].permute(1, 2, 0).cpu().numpy()
        input_gray = L[i].permute(1, 2, 0).cpu().numpy()
        
        plt.subplot(4, 3, i*3 + 1); plt.imshow(input_gray, cmap='gray'); plt.axis('off'); 
        if i==0: plt.title("Input")
        plt.subplot(4, 3, i*3 + 2); plt.imshow(pred); plt.axis('off'); 
        if i==0: plt.title(f"Ep {epoch+1}")
        plt.subplot(4, 3, i*3 + 3); plt.imshow(orig); plt.axis('off'); 
        if i==0: plt.title("Target")
        
    plt.savefig(f"{PREVIEW_DIR}/epoch_{epoch+1}.png")
    plt.close()
    model.train()

# --- MAIN ---
def main():
    print(f"🔥 V14 LARGE (Semi-Frozen + Scheduler) sur {DEVICE}")
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Dossier {TRAIN_DIR} introuvable.")
        return

    train_loader = DataLoader(HDDataset(TRAIN_DIR, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(HDDataset(VAL_DIR, False), batch_size=4, shuffle=True)
    
    model = ColorizerV14().to(DEVICE)
    
    if os.path.exists(SAVE_PATH):
        print("📥 Chargement des poids V14...")
        try:
            model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        except:
            print("⚠️ Poids non compatibles, démarrage à zéro.")

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # 3. UPGRADE: Scheduler (CORRECTIF PYTORCH 2.4+: plus de verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.L1Loss()
    
    print(f"🚀 Début pour {EPOCHS} époques...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for imgs in loop:
            imgs = imgs.to(DEVICE)
            
            lab = rgb_to_lab(imgs)
            L = lab[:, 0:1]
            ab = lab[:, 1:3]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                L_dwt = dwt_init(L)
                real_ab_dwt = torch.cat([dwt_init(ab[:,0:1]), dwt_init(ab[:,1:2])], 1)
                
                # Input gris pour le cerveau
                gray_3ch = torch.cat([L, L, L], dim=1)
                
                fake_ab_dwt = model(NORMALIZE(gray_3ch), L_dwt)
                loss = criterion(fake_ab_dwt, real_ab_dwt)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = running_loss/len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_imgs in val_loader:
                val_imgs = val_imgs.to(DEVICE)
                v_lab = rgb_to_lab(val_imgs)
                v_L = v_lab[:, 0:1]
                v_ab = v_lab[:, 1:3]
                v_gray = torch.cat([v_L, v_L, v_L], dim=1)
                
                v_pred = model(NORMALIZE(v_gray), dwt_init(v_L))
                v_real = torch.cat([dwt_init(v_ab[:,0:1]), dwt_init(v_ab[:,1:2])], 1)
                val_loss += criterion(v_pred, v_real).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Step du scheduler
        scheduler.step(avg_val_loss)
        
        # On affiche le LR manuellement
        current_lr = optimizer.param_groups[0]['lr']
        print(f"✅ Ep {epoch+1} | Loss: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        torch.save(model.state_dict(), SAVE_PATH)
        visualize(epoch, model, val_loader)

if __name__ == '__main__':
    main()