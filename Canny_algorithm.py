import numpy as np
import cv2

class CannyDetector():
    def __init__(self, filter_kernel : int, sigma : int): 
        self.filter = self.gaussian_filter(filter_kernel, sigma)
        self.h_ratio = 0.9 
        self.l_ratio = 0.05

    def detect(self, image : np.array):
        smoothed_image = self.convole(image, self.filter)
        G, theta = self.gradient_maps(smoothed_image)
        Z = self.non_maximum_suppression(G, theta)
        Z = self.double_threshold(Z, G, self.h_ratio, self.l_ratio)
        return self.hysteresis(Z)
    
    def gaussian_filter(self, kernel_size : int, sigma : int):
        filter = np.zeros((kernel_size, kernel_size))
        k = (kernel_size - 1)/2

        for i in range(kernel_size):
            for j in range(kernel_size):
                filter[i][j] = np.exp(-(((i+1)-(k+1))**2 + ((j+1)-(k+1))**2)/(2 * sigma ** 2)) / (2.0 * np.pi * sigma ** 2)

        return filter

    def convolve(self, matrix : np.array, kernel : np.array):
        # Padding to ensure same size output
        padding = kernel.shape[0] // 2
        padded_matrix = np.pad(matrix, pad_width=padding)
        result = np.zeros_like(matrix)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                convolve_output = padded_matrix[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel
                result[i][j] = np.sum(convolve_output)

        return result

    def gradient_maps(self, matrix : np.array):
        # Sobel operators
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        Ix = self.convolve(matrix, Kx)
        Iy = self.convolve(matrix, Ky)

        G = np.sqrt(Ix ** 2 + Iy ** 2) * 255 / np.max(np.sqrt(Ix ** 2 + Iy ** 2)) # Gradients in X and Y direction
        theta = np.arctan2(Iy, Ix) * 180 / np.pi # Converting Radians to angles
        theta[theta < 0] += 180
        return G, theta

    def non_maximum_suppression(self, G : np.array, theta : np.array): 
        Z = np.zeros_like(G)

        for i in range(1, G.shape[0]-1): 
            for j in range(1, G.shape[1]-1):
                angle = theta[i,j]
                
                if(0 <= angle <= 22.5 or 157.5 <= angle <= 180):
                    a = G[i+1, j]
                    b = G[i-1, j]
                elif(22.5 <= angle <= 67.5): 
                    a = G[i+1, j+1]
                    b = G[i-1, j-1]
                elif(67.5 <= angle <= 112.5): 
                    a = G[i, j+1]
                    b = G[i, j-1]
                elif(112.5 <= angle <= 157.5): 
                    a = G[i+1, j-1]
                    b = G[i-1, j+1]

                
                if(G[i, j] >= a and G[i, j] >= b): 
                    Z[i, j] = G[i, j]
                else: 
                    Z[i, j] = 0 
        return Z

    def double_threshold(self, Z : np.array, G : np.array, high_threshold_ratio=0.3, low_threshold_ratio=0.1):
        strong_i, strong_j = np.where(G >= high_threshold_ratio)
        weak_i, weak_j = np.where((G <= high_threshold_ratio) & (G >= low_threshold_ratio))
        suppressed_i, suppressed_j = np.where(G < low_threshold_ratio)

        Z[strong_i, strong_j] = 255
        Z[weak_i, weak_j] = 25 
        Z[suppressed_i, suppressed_j] = 0

        return Z

    def hysteresis(self, Z : np.array):
        M, N = Z.shape
        for i in range(1, M-1):
            for j in range(1, N-1):
                if(Z[i, j] == 25):
                    try:
                        if(Z[i-1, j] == 255 or Z[i+1, j] == 255 or Z[i, j-1] == 255 or Z[i, j+1] == 255 or Z[i-1, j-1] == 255 or Z[i-1, j+1] == 255 or Z[i+1, j-1] == 255 or Z[i+1, j+1] == 255):
                            Z[i,j] = 255 
                        else: 
                            Z[i,j] = 0
                    except IndexError:
                        pass
        return Z