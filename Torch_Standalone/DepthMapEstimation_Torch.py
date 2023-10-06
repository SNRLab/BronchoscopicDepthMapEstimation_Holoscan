
import os
import math

import subprocess
import time
import numpy as np
import cupy as cp

#import the depth estimation
'''import importlib.util
import sys
module = importlib.util.spec_from_file_location("module.name", "/path/to/file.py")
foo = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = foo
spec.loader.exec_module(foo)
foo.MyClass()'''

#depth estimation imports
import argparse
import itertools
import datetime
from PIL import Image
import matplotlib.pyplot as plt

import pyigtl

import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

from utils import models
from utils import utils

import torch.nn as nn
import torch.nn.functional as F
import torch
print(torch.__version__)      

#depth estimation intro
parser = argparse.ArgumentParser()
parser.add_argument("-network_name", type=str, default="6Level", help="name of the network")

parser.add_argument("--dataset_name", type=str, default="ex-vivo", help="name of the training dataset")
parser.add_argument("--testing_dataset", type=str, default="AS", help="name of the testing dataset")
parser.add_argument("--lambda_cyc", type=float, default=0.1, help="cycle loss weight")

parser.add_argument("--epoch", type=int, default=50, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=200, help="size of image height")
parser.add_argument("--img_width", type=int, default=200, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--textfile_training_results_interval", type=int, default=50,
                    help="textfile_training_results_interval")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_id", type=float, default=1, help="identity loss weight")
opt = parser.parse_args()
#print(opt)

cuda = torch.cuda.is_available()
input_shape = (opt.channels, opt.img_height, opt.img_width)
# Initialize generator and discriminator
G_AB = models.GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
  G_AB = G_AB.cuda()
  
if opt.epoch != 0:
  # Load pretrained models
  modulePath = os.path.dirname(os.path.abspath(__file__))
  #G_AB.load_state_dict(torch.load("./6Level-ex-vivo-G_AB.pth",map_location=torch.device('cpu')))
  G_AB.load_state_dict(torch.load("./model/6Level-ex-vivo-G_AB.pth"))
  
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_B1_buffer = utils.ReplayBuffer()

#total_params = sum(p.numel() for p in G_AB.parameters())
#print(total_params)

def generateDepth(imageData):
  transforms_testing_non_fliped_ = [transforms.ToTensor()]

  val_dataloader_non_flipped = DataLoader(ImageDataset(imageData, transforms_=transforms_testing_non_fliped_, unaligned=False), batch_size=1, shuffle=False, num_workers=0)
  G_AB.eval()

  for i, batch in enumerate(val_dataloader_non_flipped):
    #start = time.time()
    real_A = Variable(batch["A"].type(Tensor))
    fake_B1 = G_AB(real_A)
    #end = time.time()
    #print("Time elapsed: ", end - start, "seconds")
  outputArray = fake_B1.cpu().detach().numpy()
  twoD_depthArray = (outputArray[0,0,:]*1000).astype(np.uint8)
  arr_min = np.min(outputArray)
  arr_max = np.max(outputArray)
  twoD_depthArray = ((outputArray[0,0,:] - arr_min) * 255 / (arr_max - arr_min)).astype(np.uint8)  
  
  return twoD_depthArray

class ImageDataset(Dataset):
    def __init__(self, imageData, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        #self.imageData = imageData.resize((200, 200))
        self.imageData = imageData

    def __getitem__(self, index):
        item_A = self.transform(self.imageData)
        return {"A": item_A}
  
    def __len__(self):
        return 1

frameCount = 0
lastTime = time.time()
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(frame.shape)

Client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18939)
Client.start()

# Get the FPS of the camera
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
while True:
  # Read a frame from the camera
  ret, frame = cap.read()

  # Check if the frame is valid
  if ret:
    timestamp_start = time.time()

    color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    pil_image = pil_image.resize((200, 200))

    # RGB output
    imageRGB = cp.asnumpy(cp.fliplr(cp.flipud(cp.array(pil_image))))
    imageRGBMessage = pyigtl.ImageMessage(imageRGB, device_name="RGB_Image")
    Client.send_message(imageRGBMessage, wait=False)

    pil_image = pil_image.convert("L")
    
    # Generate depth
    timestamp_in = time.time()
    depth = generateDepth(pil_image)
    currentTimestamp = time.time()
    deltaLatencyTime = math.ceil((currentTimestamp - timestamp_in)*1000)

    # Depth Output
    imageDepth = cp.asnumpy(cp.fliplr(cp.flipud(cp.array(depth))))
    imageDepthMessage = pyigtl.ImageMessage(imageDepth, device_name="Depth_Image")
    Client.send_message(imageDepthMessage, wait=False)

    deltaLatencyTimeTotal = math.ceil((currentTimestamp - timestamp_start)*1000)

    # Framerate calculation
    frameCount += 1
    deltaTime = time.time() - lastTime
    frameRate = np.around(frameCount / deltaTime, 1)
    if frameCount > 5:
      frameCount = 0
      lastTime = time.time()

    print(f'FPS: {frameRate} ; Inference Time: {deltaLatencyTime} ms ; Total Time: {deltaLatencyTimeTotal} ms              ', end='\r')

    if cv2.waitKey(1) == ord('q'):
      break

