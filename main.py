#!/usr/bin/python3

import pyscreenshot as ps
from time import sleep
import numpy as np
import cv2

import torch
import torch.optim as optim

from model import CustomNet

import pyautogui

model = CustomNet()
model.to('cuda:0')

optimizer = optim.Adam(model.parameters(), lr=3e-4)

loss_fn = torch.nn.CrossEntropyLoss()

gameover_screen = cv2.imread('gameover.jpg')

# gameover coord
# bbox=(910,920,1000,975)
# frame = ps.grab(bbox=(30,780,1915,1150))

# jump, nothing, duck

def get_labels(out, is_go):
    preds, arg = torch.max(out, 1)

    if arg == 0: # jump
        if is_go:
            return [0, 1, 0]
        else:
            return [1, 0, 0]
    if arg == 1: # nothing
        if is_go:
            return [1, 0, 0]
        else:
            return [0, 1, 0]
    if arg == 2: # duck
        if is_go:
            return [1, 0, 0]
        else:
            return [0, 0, 1]

while True:
    frame = ps.grab(bbox=(30,530,975,1475))

    is_gameover = (np.array(ps.grab(bbox=(910,920,1000,975))) == gameover_screen)[0][0][0] # TODO: fix

    frame_np = np.array(frame)

    old_size = frame_np.shape[:2]

    n_size = 64

    ratio = float(n_size) / max(old_size)

    new_size = tuple([int(x * ratio) for x in old_size])

    frame_res = cv2.resize(frame_np, (new_size[1], new_size[0]))

    # cv2.imshow('wtf2', frame_res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray_frame = cv2.cvtColor(frame_res, cv2.COLOR_BGR2GRAY) 

    inp = torch.Tensor([[gray_frame]]).to('cuda:0')

    optimizer.zero_grad()

    out = model(inp)

    action = torch.argmax(out)

    if is_gameover:
        sleep(1)
        print('gameover')
        pyautogui.press('enter')
        sleep(2.7)
    else:
        if action == 0:
            print('jump')
            pyautogui.press('space')
        if action == 1:
            print('walk')
        if action == 2:
            print('duck')
            pyautogui.press('down')


    out_vals = out[0]

    # print(out)

    labels = torch.Tensor([get_labels(out, is_gameover).index(1)]).long().to('cuda:0')

    # print(labels)

    loss = loss_fn(out, labels)

    loss.backward()
    
    optimizer.step()

    # sleep(5)







