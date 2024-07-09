import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from Canny_algorithm import CannyDetector

def apply_count(frame, edges):
    edges = edges.astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contour = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 1)
    return image_with_contour

def main_video(detector: CannyDetector, input_video: str, output_video: str):
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file '{input_video}' does not exist.")
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Error opening video file '{input_video}'.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=True)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count), desc="[EDGE DETECTION]"):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = detector.detect(gray_frame)

        # Convert edges to uint8 before converting to a 3-channel image
        edges_uint8 = edges.astype(np.uint8)
        edges_colored = cv2.cvtColor(edges_uint8, cv2.COLOR_GRAY2BGR)
        out.write(edges_colored)
    
    cap.release()
    out.release()
    print("[INFO] Video Edge Detection complete.")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True, type=str, help="Path to the input video file.")
    parser.add_argument("--output_video", required=True, type=str, help="Path to save the output video file.")
    parser.add_argument("--kernel_size", required=False, type=int, help="Gaussian Kernel Size. Not required.")
    parser.add_argument("--variance", required=False, type=int, help="Variance for Gaussian Kernel. Not required.")
    parser.add_argument("--h_ratio", required=False, type=float, help="Threshold for high/strong edges. Not required.")
    parser.add_argument("--l_ratio", required=False, type=float, help="Threshold for low/weak edges. Not required.")

    args = parser.parse_args()

    detector = CannyDetector(lowThresholdRatio=args.l_ratio if args.l_ratio else 0.05,
                             highThresholdRatio=args.h_ratio if args.h_ratio else 0.15, 
                             kernel_size=args.kernel_size if args.kernel_size else 5, 
                             sigma=args.variance if args.variance else 3)

    main_video(detector=detector, input_video=args.input_video, output_video=args.output_video)
