import argparse
from Canny_algorithm import * 
from glob import glob
import os
from tqdm import tqdm
import cv2 

def main(detector : CannyDetector, input_dir : str, output_dir : str):
    images = glob(os.path.join(input_dir, "*"))
    for img in tqdm(images, desc="[EDGE DETECTION]"):
        image = cv2.imread(img, 0)        
        edges = detector.detect(image)

        output_path = os.path.join(output_dir, os.path.basename(img))
        cv2.imwrite(output_path, edges)
    
    print("[INFO] Batch Edge Detection complete.")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="Folder with input images or single image.")
    parser.add_argument("--output_dir", required=True, type=str, help="Folder to save output files to.")
    parser.add_argument("--kernel_size", required=False, type=int, help="Gaussian Kernel Size. Not required.")
    parser.add_argument("--variance", required=False, type=int, help="Variance for Gaussian Kernel. Not required.")
    parser.add_argument("--h_ratio", required=False, type=float, help="Threshold for high/strong edges. Not required.")
    parser.add_argument("--l_ratio", required=False, type=float, help="Threshold for low/weak edges. Not required.")

    args = parser.parse_args()

    detector = CannyDetector(lowThresholdRatio=args.l_ratio if args.l_ratio else 0.05,
                             highThresholdRatio=args.h_ratio if args.h_ratio else 0.15, 
                             kernel_size=args.kernel_size if args.kernel_size else 5, 
                             sigma=args.variance if args.variance else 3)

    main(detector=detector, input_dir=args.input_dir, output_dir=args.output_dir)