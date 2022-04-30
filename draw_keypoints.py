import cv2
from utils import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np

IMAGEWIDTH = 640.0
IMAGEHEIGHT = 480.0

expert_data = read_expert()
states = None

for key in expert_data.keys():
        states = get_coords(expert_data[key]).to_numpy()
        state_normalizer = MinMaxScaler(feature_range=(0,1))
        states = state_normalizer.fit_transform(states)

        for i in range(1, states.shape[1], 2):
            states[:, i] = states[:, i] * IMAGEWIDTH
        
        for i in range(0, states.shape[1], 2):
            states[:, i] = states[:, i] * IMAGEHEIGHT
        
        states = np.around(states).astype(int)
        break

image = np.zeros((480, 640, 3), dtype=np.uint8)

for i in range(0, states.shape[1], 2):
    c_coords = (states[0, i], states[0, i+1])
    radius = 3
    color = (51, 51, 255)
    thickness = -1
    image = cv2.circle(image, c_coords, radius, color, thickness)

# connect thumb
image = cv2.line(image, (states[0,2], states[0,3]), (states[0,4], states[0,5]), (51, 51, 255), 1)
image = cv2.line(image, (states[0,4], states[0,5]), (states[0,6], states[0,7]), (51, 51, 255), 1)
image = cv2.line(image, (states[0,6], states[0,7]), (states[0,8], states[0,9]), (51, 51, 255), 1)
image = cv2.line(image, (states[0,8], states[0,9]), (states[0,10], states[0,11]), (51, 51, 255), 1)

# connect index
image = cv2.line(image, (states[0,2], states[0,3]), (states[0,12], states[0,13]), (0, 255, 128), 1)
image = cv2.line(image, (states[0,12], states[0,13]), (states[0,14], states[0,15]), (0, 255, 128), 1)
image = cv2.line(image, (states[0,14], states[0,15]), (states[0,16], states[0,17]), (0, 255, 128), 1)
image = cv2.line(image, (states[0,16], states[0,17]), (states[0,18], states[0,19]), (0, 255, 128), 1)

# connect middle
image = cv2.line(image, (states[0,2], states[0,3]), (states[0,20], states[0,21]), (0, 153, 76), 1)
image = cv2.line(image, (states[0,20], states[0,21]), (states[0,22], states[0,23]), (0, 153, 76), 1)
image = cv2.line(image, (states[0,22], states[0,23]), (states[0,24], states[0,25]), (0, 153, 76), 1)
image = cv2.line(image, (states[0,24], states[0,25]), (states[0,26], states[0,27]), (0, 153, 76), 1)

# connect ring
image = cv2.line(image, (states[0,2], states[0,3]), (states[0,28], states[0,29]), (255, 0, 0), 1)
image = cv2.line(image, (states[0,28], states[0,29]), (states[0,30], states[0,31]), (255, 0, 0), 1)
image = cv2.line(image, (states[0,30], states[0,31]), (states[0,32], states[0,33]), (255, 0, 0), 1)
image = cv2.line(image, (states[0,32], states[0,33]), (states[0,34], states[0,35]), (255, 0, 0), 1)

# connect pinky
image = cv2.line(image, (states[0,2], states[0,3]), (states[0,36], states[0,37]), (255, 0, 255), 1)
image = cv2.line(image, (states[0,36], states[0,37]), (states[0,38], states[0,39]), (255, 0, 255), 1)
image = cv2.line(image, (states[0,38], states[0,39]), (states[0,40], states[0,41]), (255, 0, 255), 1)
image = cv2.line(image, (states[0,40], states[0,41]), (states[0,42], states[0,43]), (255, 0, 255), 1)

cv2.imwrite('image.jpg', image)