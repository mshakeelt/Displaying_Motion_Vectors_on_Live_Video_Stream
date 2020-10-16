In this framework we are
* taking the 25 frame from the live video using webcam
* converting RGB frames to YCbCr
* Taking only Y frame
* taking 8*8 blocks
* comparing current Y frame to previous Y frame
* drawing motion vectors

# Usage
python main.py