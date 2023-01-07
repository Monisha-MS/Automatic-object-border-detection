#!/usr/bin/env python
# coding: utf-8


# Importing libraries
import cv2
import numpy as np

# Read image
image = cv2.imread(r"input.jpg")

#Resize image
resized_image = cv2.resize(image, (1000, 700))

# Select ROI
r = cv2.selectROI("select the area", resized_image)

# Crop image
cropped_image = resized_image[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
coordinates=(r[0], r[1])

# Display cropped image
#cv2.imshow("Cropped image", cropped_image)
cv2.imwrite("Cropped_image.jpg", cropped_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()

# bg_removal

from rembg import remove
from PIL import Image

input_path = 'Cropped_image.jpg'
output_path = 'bg_removed.png'

input = Image.open(input_path)
output = remove(input)
output.save(output_path)


# contour
im = cv2.imread( 'bg_removed.png',0)
contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
overlay = np.zeros_like(im)
cv2.drawContours(overlay, contours, -1, 255, 3)
cv2.imwrite('overlay.jpg', overlay)

#cv2.imshow('Contour', overlay)
#cv2.waitKey(10000)
#cv2.destroyAllWindows()


# Displaying the coordinates where the overlay image should be placed on the original image
x,y = coordinates
# print(x,y)


#overlaying
Overlay = cv2.imread('overlay.jpg')
h1, w1 = Overlay.shape[:2]
region_of_interest = resized_image[y: y+h1, x:x + w1]
gray = cv2.cvtColor(Overlay, cv2.COLOR_BGRA2GRAY)
#cv2.imshow('Grayscale', gray)
#cv2.waitKey(10000)
#cv2.destroyAllWindows()
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
background_img = cv2.bitwise_and(region_of_interest, region_of_interest, mask = mask)
final_image = cv2.add(background_img, Overlay)
resized_image[y: y+h1, x:x + w1] = final_image


#Saving the border detected output
cv2.imwrite("output.jpg",resized_image)
#Display output
cv2.imshow('Output', resized_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()

