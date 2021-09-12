#!/usr/bin/python3

import pyscreenshot as ps
from time import sleep
import numpy as np
import cv2

import torch
import torch.optim as optim

from model import CustomNet

model = CustomNet()
model.to('cuda:0')

optimizer = optim.Adam(model.parameters(), lr=3e-4)

loss_fn = torch.nn.CrossEntropyLoss()

gameover_screen = cv2.imread('gameover.jpg')

# gameover coord
# bbox=(910,920,1000,975)

while True:
    frame = ps.grab(bbox=(30,780,1915,1150))

    is_gameover = (np.array(ps.grab(bbox=(910,920,1000,975))) == gameover_screen)[0][0][0] # TODO: fix

    frame_np = np.array(frame)

    gray_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY) 

    inp = torch.Tensor(gray_frame).to('cuda:0')

    optimizer.zero_grad()

    # TODO: resize and pad

    out = model(inp)

    print(out)

    # loss

    optimizer.step()
    
    sleep(5)







