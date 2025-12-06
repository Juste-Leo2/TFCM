import torch

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