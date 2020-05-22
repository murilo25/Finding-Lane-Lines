
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
#cap = cv2.VideoCapture('solidYellowLeft.mp4')
#cap = cv2.VideoCapture('challenge.mp4')
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

fps = cap.get(5)
frame_width = cap.get(3)
frame_height = cap.get(4)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('myVideo.avi',fourcc, fps, (int(frame_width),int(frame_height)))

# filter parameters
alpha = 0.9
frameCounter = 0
filtered_mu_pos = 0
filtered_mu_neg = 0
filtered_b_pos = 0
filtered_b_neg = 0

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
# Define vertices of region of interest polygon
vertices = np.array([[(900,539),(135, 539), (438, 326), (522,326)]], dtype=np.int32) #322
# Define the Hough transform parameters
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 25     # minimum number of votes (intersections in Hough grid cell) was 1
min_line_length = 10 #minimum number of pixels making up a line was 5
max_line_gap = 10  # maximum gap in pixels between connectable line segments was 10

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frameCounter += 1
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
        #plt.imshow(masked_edges)
        #plt.show()

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
        # Iterate over the output "lines" and draw lines on a blank image
        #mu=[]
        #b=[]
        mu_positive = 0
        mu_negative = 0
        b_positive = 0
        b_negative = 0
        positive_count = 0
        negative_count = 0
        for line in lines:
            for x1,y1,x2,y2 in line:
                #cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)

                #compute line eq. (y=mx+b)
                if (x1-x2!=0):
                    slope_m = (y1-y2)/(x1-x2)
                    offset_b = -slope_m*x1+y1

                    # for debug
                    #mu.append(slope_m)
                    #b.append(offset_b)
                    if (slope_m>0):
                        mu_positive = mu_positive + slope_m
                        b_positive = b_positive + offset_b
                        positive_count += 1
                    elif (slope_m<0):
                        mu_negative = mu_negative + slope_m
                        b_negative = b_negative + offset_b
                        negative_count += 1

            # Create a "color" binary image to combine with line image
            color_edges = np.dstack((edges, edges, edges)) 

            # Draw the lines on the edge image
            lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

        # average mu and b
        mu_positive = mu_positive/positive_count
        mu_negative = mu_negative/negative_count
        b_positive = b_positive/positive_count
        b_negative = b_negative/negative_count

        if (frameCounter == 1):
            filtered_mu_pos = mu_positive
            filtered_mu_neg = mu_negative
            filtered_b_pos = b_positive
            filtered_b_neg = b_negative
        else:
            filtered_mu_pos = filtered_mu_pos*alpha + mu_positive*(1 - alpha)
            filtered_mu_neg = filtered_mu_neg*alpha + mu_negative*(1 - alpha)
            filtered_b_pos = filtered_b_pos*alpha + b_positive*(1 - alpha)
            filtered_b_neg = filtered_b_neg*alpha + b_negative*(1 - alpha)

        #filtered_mu_pos = (cumulative_mu_pos + mu_positive)/frameCounter
        #filtered_mu_neg = (cumulative_mu_neg + mu_negative)/frameCounter
        #filtered_b_pos = (cumulative_b_pos + b_positive)/frameCounter
        #filtered_b_neg = (cumulative_mu_neg + b_negative)/frameCounter

        y1p_final = 328
        y2p_final = 540
        x1p_final = int( (y1p_final-filtered_b_pos)/filtered_mu_pos )
        x2p_final = int( (y2p_final-filtered_b_pos)/filtered_mu_pos )
        y1n_final = 328
        y2n_final = 540
        x1n_final = int( (y1n_final-filtered_b_neg)/filtered_mu_neg )
        x2n_final = int( (y2n_final-filtered_b_neg)/filtered_mu_neg )
       
        cv2.line(line_image,(x1p_final,y1p_final),(x2p_final,y2p_final),(0,0,255),10)   #line_image
        cv2.line(line_image,(x1n_final,y1n_final),(x2n_final,y2n_final),(0,0,255),10)   #line_image

        final_frame = cv2.bitwise_or(line_image, frame)
        out.write(final_frame)

        cv2.imshow('frame',final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()

#print(mu)
#print(len(lines))
#print(mu_positive)
#print(mu_negative)