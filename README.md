# finding-lane-lines

Find lane lines in a video using OpenCV. The detection is based on the following:
- Canny filter to identify edges (strong color transitions)
- Hough transform to obtain coordinates for line segments
- Region of interest masking

The slope and intersect for each line segment are calculated and separated in two groups: positive and negative slopes/intersects, which are used to classify to which lane line it belongs.
Each group of slopes and intersects are averaged to compute a single line segment for each lane line.
A new pair of lines are computed for each frame of the video. The transition between lines from subsequent frames are smoothened using an IIR filter.

FindingLaneLines.py is the main file which takes as input a video and outputs a new video with the lanes highlighted. Example outputs can be found on
_leftLane.avi_ and _rightLane.avi_ files, which are outputs for _solidYellowLeft.mp4_ and _solidWhiteRight.mp4_ files, respectively.
