

import cv2
import pandas as pd
from skimage.io import imread, imshow, imsave


img_path = 'image.jpg'
img = cv2.imread(img_path)

# declaring global variables (are used later on)
clicked = False
r = g = b = x_pos = y_pos = 0

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# function to calculate minimum distance from all colors and get the most matching color
def get_color_name(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


# function to get x,y coordinates of mouse double click
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, x_pos, y_pos, clicked
        clicked = True
        x_pos = x
        y_pos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_function)

while True:

    cv2.imshow("image", img)
    if clicked:

        # cv2.rectangle(image, start point, endpoint, color, thickness)-1 fills entire rectangle
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

        # Creating text string to display( Color name and RGB values )
        text = get_color_name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)

        # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # For very light colours we will display text in black colour
        if r + g + b >= 600:
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

    # Break the loop when user hits 'esc' key
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

from skimage.io import imread, imshow, imsave

im = imread('image.jpg')

from matplotlib import pyplot as plt

plt.figure()
imshow(im)
plt.axis('off')
plt.title('Original Image')
plt.show()

red = im[:, :, 0]
green = im[:, :, 1]
blue = im[:, :, 2]

plt.imshow(red, cmap="cool")
plt.title('Cool')
plt.axis('off')
plt.show()

plt.imshow(green, cmap="Paired")
plt.title('Green')
plt.axis('off')
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.io import imread, imshow, imsave

img = Image.open('image.jpg')

imgmatrix = np.array(img)

imggray = imgmatrix.mean(axis=2)

plt.figure()

fig1 = plt.subplot(1,2,1)
fig1.imshow(imgmatrix)

fig1.set_title('Original')
fig1.axis('off')

fig2 = plt.subplot(1,2,2)
fig2.imshow(imggray, cmap='gray')

fig2.set_title('After Processing')
fig2.axis('off')

plt.show()


img.show()

# L = R * 2125/10000 + G * 7154/10000 + B * 0721/10000
img_data = img.getdata()
lst=[]

for i in img_data:
    lst.append(i[0]*0.2125+i[1]*0.7174+i[2]*0.0721)

new_img = Image.new("L", img.size)
new_img.putdata(lst)

new_img.show()

img = img.convert("L")
img.show()