# Canny_edge_detector

## Overview
This repository is built for educational purposes on edge detection using Canny Algorithm alongside various improvements for accuracy, robustness, and inference speed. The main goal is to illustrate the Canny Algorithm in its entirety, showing the fundamental logic behind the algorithm as well as involved operations.Alongside the edge detector implementation, image and video inference scripts are developed to be used for public usage. However, video inference can not be implemented in real-time to inference speed (1.95s per 640x480 frame). 

## Usage
- For image inference, run the following code snippet.

```python
python image_inference.py --input_path path/to/image_folder --output_path path/to/output_folder
```

- For video inference, run the following code snippet.

```python
python video_inference.py --input_path path/to/video.mp4 --output_path path/to/output.mp4
```

## Visual Results!
- Video
[![Watch the video](https://img.youtube.com/vi/7okfMyesJOs/maxresdefault.jpg)](https://www.youtube.com/watch?v=7okfMyesJOs&ab_channel=CodeKnight)
