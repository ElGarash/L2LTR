import os
import torch
import torch.nn as nn
import numpy as np


from utils.data_utils import get_loader
from torch.utils.data import DataLoader
from tqdm import tqdm
# from apex import amp
import scipy.io as scio
import torch.nn.functional as F
import argparse
import pickle
import torchvision.transforms as transforms
from PIL import Image


from models.model_crossattn import VisionTransformer, CONFIGS
#from utils.data_utils import get_loader




def preprocess(ground_image):
    transform = transforms.Compose(
        [transforms.Resize((args.img_size[0], args.img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

    preprocessed_image = transform(Image.open(ground_image).convert('RGB'))

    return preprocessed_image


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, help="The path of the ground image to calculate the distance for")
parser.add_argument("--img_size", default=(128, 512), type=int, help="Resolution size")
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "R50-ViT-B_32"],
                    default="R50-ViT-B_16",
                    help="Which variant to use.")
parser.add_argument("--model_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()
args.device = device


config = CONFIGS[args.model_type]

model_grd = VisionTransformer(config, args.img_size)

print("loading model form ", os.path.join(args.model_dir,'model_grd_checkpoint.pth'))

state_dict = torch.load(os.path.join(args.model_dir,'model_checkpoint.pth'))
model_grd.load_state_dict(state_dict['model_grd'])

model_grd.to(device)

DESCRIPTORS_DIRECTORY = '/kaggle/working/descriptors/L2LTR'

with open(f"{DESCRIPTORS_DIRECTORY}/satellite_descriptors.pkl", 'rb') as f:
    sat_global_descriptor = pickle.load(f)

grd_global_descriptor = np.zeros([1, 768])
model_grd.eval()


with torch.no_grad():
    x_grd = preprocess(args.image_path)
    x_grd = torch.unsqueeze(x_grd, 0)
    
    x_grd = x_grd.to(args.device)

    grd_global = model_grd(x_grd)

    grd_global_descriptor = grd_global.detach().cpu().numpy()


dist_array = 2.0 - 2.0 * np.matmul(sat_global_descriptor, grd_global_descriptor.T)

print('dist_array shape', dist_array.shape)