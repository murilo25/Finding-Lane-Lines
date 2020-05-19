
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Find lane lines in an image/video using OpenCV 
# Canny Edge Detector with Hough Transform and region masking

# Read in and grayscale the image
#image = mpimg.imread('exit-ramp.jpg')
#image = mpimg.imread('solidWhiteCurve.jpg')
#image = mpimg.imread('solidWhiteRight.jpg')
#image = mpimg.imread('solidYellowCurve.jpg')
#image = mpimg.imread('solidYellowCurve.jpg')
#image = mpimg.imread('solidYellowCurve2.jpg')
#image = mpimg.imread('solidYellowLeft.jpg')
image = mpimg.imread('whiteCarLaneSwitch.jpg')

cap = cv2.VideoCapture('solidWhiteRight.mp4')

print("W: ")
print(cap.get(3))
print("\nH: ")
print(cap.get(4))
print("\nFPS: ")
print(cap.get(5))
print("\n4cc: ")
print(cap.get(6))
print("\nFORMAT: ")
print(cap.get(8))


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('myVideo.avi',fourcc, 25.0, (960,540))

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
vertices = np.array([[(900,539),(142, 539), (438, 322), (522,322)]], dtype=np.int32)
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 25     # minimum number of votes (intersections in Hough grid cell) was 1
min_line_length = 10 #minimum number of pixels making up a line was 5
max_line_gap = 10  # maximum gap in pixels between connectable line segments was 10
#line_image = np.copy(image)*0 # creating a blank to draw lines on
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',gray)

        line_image = np.copy(frame)*0 # creating a blank to draw lines on

        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        # Next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(edges)   
        ignore_mask_color = 255   
        # This time we are defining a four sided polygon to mask    
        #imshape = image.shape
    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)
        #cv2.imshow('frame',masked_edges)

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
        # Iterate over the output "lines" and draw lines on a blank image
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)

            # Create a "color" binary image to combine with line image
            color_edges = np.dstack((edges, edges, edges)) 

            # Draw the lines on the edge image
            lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 


        final_frame = cv2.bitwise_or(line_image, frame)
        out.write(final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()