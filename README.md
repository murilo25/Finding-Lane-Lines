# Finding Lane Lines

Find lane lines in a video using OpenCV. The detection is based on the following:
- Canny filter to identify edges (strong color transitions)
- Hough transform to obtain coordinates from line segments
- Region of interest masking

The slope and intersect for each line segment are calculated and separated in two groups: positive and negative slopes, which are used to classify to which lane line it belongs.
Each group of slopes and intersects are averaged to compute a single line segment for each lane line.
A new pair of lines are computed for each frame of the video. The transition between lines from subsequent frames are smoothened using an IIR filter.

FindingLaneLines.py is the main file which takes as input a video and outputs a new video with the lanes highlighted. Example outputs can be found on
The files _leftLane.avi_ and _rightLane.avi_, which are outputs for _solidYellowLeft.mp4_ and _solidWhiteRight.mp4_ files, respectively.

# Bug

Code fails for _challenge.mp4_ input. Reasons:
- Region of interest changes due to different frame dimensions
- Tree shades present in the video causes several "near to horizontal lines" to be detected, averaging slope down and causing failure to detect lane lines. 

Attempts to fix:
- Selecting proper region of interest (1150,650),(200,650),(645,430),(727,430) - improves solution
- Reject slopes lower than a threshold - no improvement
- Tune Hough transform parameters - no improvement

