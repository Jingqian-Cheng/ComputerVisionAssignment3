import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.segmentation as seg

class Center:
    def __init__(self, L, A, B, x, y):
        self.L = L
        self.A = A
        self.B = B
        self.x = x
        self.y = y
        self.pixels = []

    def update(self, L, A, B, x, y):
        self.L = L
        self.A = A
        self.B = B
        self.x = x
        self.y = y
    
    # def __eq__(self, other):
    #     return self.L == other.L and self.A == other.A and self.B == other.B and self.x == other.x and self.y == other.y

    def __str__(self):
        return "L: " + str(self.L) + " A: " + str(self.A) + " B: " + str(self.B) + " x: " + str(self.x) + " y: " + str(self.y) + " pixels: " + str(len(self.pixels))

class Pixel:
    def __init__(self, L, A, B, x, y):
        self.L = L
        self.A = A
        self.B = B
        self.x = x
        self.y = y

    # def __eq__(self, other):
    #     return self.L == other.L and self.A == other.A and self.B == other.B and self.x == other.x and self.y == other.y

    def __str__(self):
        return "L: " + str(self.L) + " A: " + str(self.A) + " B: " + str(self.B) + " x: " + str(self.x) + " y: " + str(self.y)

class SLICProcessor:
    def __init__(self, image_path, superpixel_num, m):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.superpixel_num = superpixel_num
        self.m = m #The greater the value of m, the more spatial proximity is emphasized and the more compact the cluster
        self.disMap = np.full((self.height, self.width), np.inf)
        self.label = np.full((self.height, self.width), None)
        self.N = self.height * self.width # number of pixels
        self.S = int(np.sqrt(self.N / self.superpixel_num)) # step

    def distance(self, pixel, center):
        d_lab = np.subtract(pixel.L,center.L)**2 + np.subtract(pixel.A,center.A)**2 + np.subtract(pixel.B,center.B)**2
        d_lab = np.sqrt(d_lab)
        d_xy = np.subtract(pixel.x,center.x)**2 + np.subtract(pixel.y,center.y)**2
        d_xy = np.sqrt(d_xy)
        d_lab = round(d_lab, 3)
        d_xy = round(d_xy, 3)
        dis = d_lab + (self.m/self.S)*d_xy
        return round(dis, 3)

    def gradient(self, x, y):
        v1 = np.subtract(self.image[x+1, y, :], self.image[x-1, y, :])
        v2 = np.subtract(self.image[x, y+1, :], self.image[x, y-1, :])
        g1 = np.linalg.norm(v1)
        g2 = np.linalg.norm(v2)
        return g1+g2
    
    # Calculate the threshold for connectivity enforcement
    def threshold(self, percent):
        superpixel_size = int(self.height*self.width*3/self.superpixel_num)
        return int(superpixel_size*percent)
    
    def enforceConnectivity(self, percent=0.1):
        threshold = self.threshold(percent)
        for i in range(self.height):
            for j in range(self.width):
                min_distance = np.inf
                L, A, B = self.image[i, j, :]
                curr_pixel = Pixel(L, A, B, i, j)
                # seach neighbors in 8 directions
                for x in range(i-1,i+1):
                    if x < 0 or x >= self.height: continue
                    for y in range(j-1,j+1):
                        if y < 0 or y >= self.width: continue
                        if x==i and y==j: continue
                        L, A, B = self.image[x, y, :]
                        neighbor_pixel = Pixel(L, A, B, x, y)
                        dis = self.distance(neighbor_pixel, curr_pixel)
                        if dis < threshold and dis < min_distance:
                            # print("threshold: "+str(threshold))
                            # print("dis: "+str(dis))
                            min_distance = dis
                            self.label[i][j] = self.label[x][y]

    def SLIC(self, iteration, threshold):
        centers = []
        cnt = 0
        # Initialize centers
        for i in range(self.S//2, self.height, self.S):
            for j in range(self.S//2, self.width, self.S):
                L, A, B = self.image[i, j, :]
                centers.append(Center(L, A, B, i, j))
                cnt += 1

        # Perturb centers
        for c in centers:
            x = c.x
            y = c.y
            minGradient = np.inf
            for i in range(x-1, x+2):
                if i < 0 or i >= self.height: continue
                for j in range(y-1, y+2):
                    if j < 0 or j >= self.width: continue
                    if i-1 >= 0 and j-1 >= 0 and i+1 < self.height and j+1 < self.width:
                        g = self.gradient(i, j)
                    else:
                        g = np.inf
                    if g < minGradient:
                        minGradient = g
                        L, A, B = self.image[i, j, :]
                        c.update(L, A, B, i, j)

        # iterations
        for n in range(iteration):
            print("Iterating: " + str(n+1) + " / " + str(iteration))
            # Assign pixels to nearest centers
            for c in centers:
                x = c.x
                y = c.y
                c.pixels = []
                
                # search in (2Sx2S) area
                for i in range(x-self.S, x+self.S+1):
                    if i < 0 or i >= self.height: continue
                    for j in range(y-self.S, y+self.S+1):
                        if j < 0 or j >= self.width: continue
                        L, A, B = self.image[i, j, :]
                        pixel = Pixel(L, A, B, i, j)
                        dis = self.distance(pixel, c)
                        if dis < self.disMap[i][j]:
                            self.disMap[i][j] = dis
                            self.label[i][j] = c
            # print(self.label)

            if n >= iteration-1:
                self.enforceConnectivity(threshold) # enforce connectivity, set the threshold to 10% of the average superpixel size

            # add pixels to its label
            for i in range(self.height):
                    if i < 0 or i >= self.height: continue
                    for j in range(self.width):
                        if j < 0 or j >= self.width: continue     
                        L, A, B = self.image[i, j, :]
                        pixel = Pixel(L, A, B, i, j)
                        if self.label[i][j] is not None:
                            self.label[i][j].pixels.append(pixel)      

            if n >= iteration-1: break # last iteration does not recalculate pixels

            # Compute new centers based on its pixels   
            for c in centers: 
                # print(c.pixels)
                x_sum = 0
                y_sum = 0
                num = 0
                for p in c.pixels:
                    x_sum += p.x
                    y_sum += p.y
                    num += 1
                new_x = int(x_sum/num)
                new_y = int(y_sum/num)
                L, A, B = self.image[new_x, new_y, :]
                c.update(L, A, B, new_x, new_y)
                c.pixels = []
                self.disMap = np.full((self.height, self.width), np.inf)
                self.label = np.full((self.height, self.width), None)

        return centers

if __name__ == "__main__":
    superpixel_num = 100
    iteration = 1
    threshold = 0.01
    processor = SLICProcessor('campus.jpeg', superpixel_num, m=10)
    centers = processor.SLIC(iteration, threshold)

    # assign the same color to a cluster
    origin = cv2.imread("campus.jpeg")
    origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
    my_SLIC = np.full((origin.shape[0], origin.shape[1], 3), (0,0,0))

    for idx, c in enumerate(centers):
        # color = colors_list[idx]
        R = 0
        G = 0
        B = 0
        for p in c.pixels:
            i = p.x
            j = p.y
            R += origin[i][j][0]
            G += origin[i][j][1]
            B += origin[i][j][2]
        num = len(c.pixels)
        color = (int(R/num), int(G/num), int(B/num)) # calculate average color of this cluster
        for p in c.pixels:
            i = p.x
            j = p.y
            my_SLIC[i][j] = color

    # result from skimage's SLIC
    skimage_SLIC = seg.slic(origin, n_segments=100, compactness=10)

    # display origin image and new image
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    # fig, ax = plt.subplots(1, 3)
    ax[0].imshow(origin)
    
    ax[0].set_title('Original Image')
    ax[1].imshow(my_SLIC)
    ax[1].set_title('My Segmented Image')
    ax[2].imshow(skimage_SLIC)
    ax[2].set_title('scikit-image\'s Segmented Image')
    plt.savefig('my_plot.png')