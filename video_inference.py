import argparse
from Canny_algorithm import * 
import os
from tqdm import tqdm
import cv2

def main_video(detector: CannyDetector, input_video: str, output_video: str):
    cap = cv2.VideoCapture(input_video)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    contour_out = cv2.VideoWriter(f"Contour_{output_video}", fourcc, fps, (width, height), isColor=True)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count), desc="[EDGE DETECTION]"):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = detector.detect(gray_frame)

        image_with_contour = apply_count(frame, edges)
        contour_out.write(image_with_contour)
    
    cap.release()
    contour_out.release()
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
