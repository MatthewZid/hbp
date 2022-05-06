import os
import cv2
from utils import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np

IMAGEWIDTH = 640.0
IMAGEHEIGHT = 480.0

def draw_points(states, filename, width, height, videopath='./expert_videos'):
    name = filename.split('.')[0]
    video_out = cv2.VideoWriter(os.path.join(videopath, name+'.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 20, (width, height))

    for i in range(states.shape[0]):
        image = np.zeros((height, width, 3), dtype=np.uint8)

        for j in range(0, states.shape[1], 2):
            c_coords = (states[i, j], states[i, j+1])
            radius = 3
            color = (51, 51, 255)
            thickness = -1
            image = cv2.circle(image, c_coords, radius, color, thickness)

        # connect thumb
        image = cv2.line(image, (states[i,2], states[i,3]), (states[i,4], states[i,5]), (51, 51, 255), 1)
        image = cv2.line(image, (states[i,4], states[i,5]), (states[i,6], states[i,7]), (51, 51, 255), 1)
        image = cv2.line(image, (states[i,6], states[i,7]), (states[i,8], states[i,9]), (51, 51, 255), 1)
        image = cv2.line(image, (states[i,8], states[i,9]), (states[i,10], states[i,11]), (51, 51, 255), 1)

        # connect index
        image = cv2.line(image, (states[i,2], states[i,3]), (states[i,12], states[i,13]), (0, 255, 128), 1)
        image = cv2.line(image, (states[i,12], states[i,13]), (states[i,14], states[i,15]), (0, 255, 128), 1)
        image = cv2.line(image, (states[i,14], states[i,15]), (states[i,16], states[i,17]), (0, 255, 128), 1)
        image = cv2.line(image, (states[i,16], states[i,17]), (states[i,18], states[i,19]), (0, 255, 128), 1)

        # connect middle
        image = cv2.line(image, (states[i,2], states[i,3]), (states[i,20], states[i,21]), (0, 153, 76), 1)
        image = cv2.line(image, (states[i,20], states[i,21]), (states[i,22], states[i,23]), (0, 153, 76), 1)
        image = cv2.line(image, (states[i,22], states[i,23]), (states[i,24], states[i,25]), (0, 153, 76), 1)
        image = cv2.line(image, (states[i,24], states[i,25]), (states[i,26], states[i,27]), (0, 153, 76), 1)

        # connect ring
        image = cv2.line(image, (states[i,2], states[i,3]), (states[i,28], states[i,29]), (255, 0, 0), 1)
        image = cv2.line(image, (states[i,28], states[i,29]), (states[i,30], states[i,31]), (255, 0, 0), 1)
        image = cv2.line(image, (states[i,30], states[i,31]), (states[i,32], states[i,33]), (255, 0, 0), 1)
        image = cv2.line(image, (states[i,32], states[i,33]), (states[i,34], states[i,35]), (255, 0, 0), 1)

        # connect pinky
        image = cv2.line(image, (states[i,2], states[i,3]), (states[i,36], states[i,37]), (255, 0, 255), 1)
        image = cv2.line(image, (states[i,36], states[i,37]), (states[i,38], states[i,39]), (255, 0, 255), 1)
        image = cv2.line(image, (states[i,38], states[i,39]), (states[i,40], states[i,41]), (255, 0, 255), 1)
        image = cv2.line(image, (states[i,40], states[i,41]), (states[i,42], states[i,43]), (255, 0, 255), 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_out.write(image)

    video_out.release()

expert_data = read_expert()
state_feature = []
for key in expert_data.keys():
    coords = get_coords(expert_data[key]).to_numpy()
    state_feature.append(coords)

state_feature = np.concatenate(state_feature, axis=0)

count = 100
for key in expert_data.keys():
    if count == 0: break
    else:
        states = get_coords(expert_data[key]).to_numpy()

        for i in range(1, states.shape[1], 2):
            states[:, i] = ((states[:, i] - state_feature.min()) / (state_feature.max() - state_feature.min())) * IMAGEHEIGHT
        
        for i in range(0, states.shape[1], 2):
            states[:, i] = ((states[:, i] - state_feature.min()) / (state_feature.max() - state_feature.min())) * IMAGEWIDTH
        
        states = np.around(states).astype(int)

        draw_points(states, key, int(IMAGEWIDTH), int(IMAGEHEIGHT))
        count -= 1