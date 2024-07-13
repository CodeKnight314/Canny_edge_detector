import numpy as np
import cv2
from numpy.fft import rfftn as np_fft
from numpy.fft import irfftn as inv_np_fft

class CannyDetector:
    def __init__(self, lowThresholdRatio=0.05, highThresholdRatio=0.15, kernel_size=5, sigma=5, convolve : bool = True):
        self.lowThresholdRatio = lowThresholdRatio
        self.highThresholdRatio = highThresholdRatio
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        self.fft = convolve

    def gaussian_kernel(self, size : int, sigma : int = 1):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(kernel_1D[i] ** 2) / (2 * sigma ** 2))
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
        kernel_2D *= 1.0 / kernel_2D.max()
        return kernel_2D
    
    def fft_convolve(self, I : np.array, F : np.array):
        """ FFT convolution with NumPy. """
        return inv_np_fft(np_fft(I) * np_fft(F, I.shape))

    def convolve(self, matrix : np.array, kernel : np.array):
        image_height, image_width = matrix.shape
        kernel_height, kernel_width = kernel.shape
        output = np.zeros(matrix.shape)
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        padded_image = np.pad(matrix, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        for x in range(image_width):
            for y in range(image_height):
                output[y, x] = (kernel * padded_image[y: y + kernel_height, x: x + kernel_width]).sum()
        
        return output

    def gaussian_blur(self, matrix : np.array, kernel_size : int = 5, sigma : int = 1):
        kernel = self.gaussian_kernel(kernel_size, sigma)

        if self.fft: 
            return self.fft_convolve(matrix, kernel)
        else: 
            return self.convolve(matrix, kernel)

    def sobel_filters(self, matrix : np.array):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
        if self.fft:
            Ix = self.fft_convolve(matrix, Kx)
            Iy = self.fft_convolve(matrix, Ky)
        else: 
            Ix = self.convolve(matrix, Kx)
            Iy = self.convolve(matrix, Ky)
        
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        
        return G, theta

    def non_maximum_suppression(self, image, D):
        M, N = image.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q = 255
                    r = 255
                    
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = image[i, j+1]
                        r = image[i, j-1]
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = image[i+1, j-1]
                        r = image[i-1, j+1]
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = image[i+1, j]
                        r = image[i-1, j]
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = image[i-1, j-1]
                        r = image[i+1, j+1]
                    
                    if (image[i, j] >= q) and (image[i, j] >= r):
                        Z[i, j] = image[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass
        
        return Z

    def threshold(self, image, lowThresholdRatio=0.05, highThresholdRatio=0.15):
        highThreshold = image.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
        
        M, N = image.shape
        res = np.zeros((M, N), dtype=np.int32)
        
        weak = np.int32(25)
        strong = np.int32(255)
        
        strong_i, strong_j = np.where(image >= highThreshold)
        weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        return res, weak, strong

    def hysteresis(self, image, weak, strong=255):
        M, N = image.shape
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (image[i, j] == weak):
                    try:
                        if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                            or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                            or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                            image[i, j] = strong
                        else:
                            image[i, j] = 0
                    except IndexError as e:
                        pass
        return image

    def detect(self, image):
        blurred_image = self.gaussian_blur(image, self.kernel_size, self.sigma)
        gradient_magnitude, gradient_direction = self.sobel_filters(blurred_image)
        non_max_image = self.non_maximum_suppression(gradient_magnitude, gradient_direction)
        threshold_image, weak, strong = self.threshold(non_max_image, self.lowThresholdRatio, self.highThresholdRatio)
        canny_image = self.hysteresis(threshold_image, weak, strong)
        return canny_image
    
def apply_count(frame: np.array, contour: np.array): 
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
    contour_image = np.zeros_like(frame)
    
    contours, _ = cv2.findContours(contour.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)  # Green contours with thickness 1
    
    image_with_contours = cv2.addWeighted(frame, 0.8, contour_image, 1, 0)
    
    return image_with_contours