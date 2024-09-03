import numpy as np
from PIL import Image
import random


def K_random(image,K):
    indices = random.sample(range(image.shape[0]), K)
    return image[indices]

def get_nearest_centroids(image,centroids):
    m,n = image.shape
    K=centroids.shape[0]
    idx=np.zeros(m,dtype=np.uint8)

    for i in range(m):
        nearest_centroid=-1
        dist = float('inf')
        for k in range(K):
            centroid_dist = 0
            for j in range(n):
                delta=float(image[i][j])-centroids[k][j]
                centroid_dist+=np.square(delta)
            if(nearest_centroid==-1 or dist>centroid_dist):
                nearest_centroid=k
                dist=centroid_dist
        idx[i]=nearest_centroid
    return idx

# centroids are basically the k colors that can efficiently represent the image
def shift_centroids_to_mean(image, K, idx):
    m,n = image.shape       #n=3
    new_centroids = np.zeros((K,n))
    centroid_sum = np.zeros_like(new_centroids)
    centroid_num = np.zeros(K)

    for i in range(m):
        assigned_centroid = int(idx[i])
        for j in range(n):
            centroid_sum[assigned_centroid][j]+=image[i][j]
        centroid_num[assigned_centroid]+=1
    #e.g. centroid_sum[i]=[100,255,0]
    for i in range(K):
        new_centroids[i]=centroid_sum[i]/centroid_num[i]
    
    return new_centroids

def compress_image_mapping(image, K):
    max_iter=10
    num_pixels=image.shape[0]
    k_initial_centroids=K_random(image, K)
    centroids=k_initial_centroids
    idx=np.zeros(num_pixels,dtype = np.uint8)
    for i in range(max_iter):
        print("RUNNING ITERATION NUMBER ",i+1)
        idx=get_nearest_centroids(image,centroids)
        centroids=shift_centroids_to_mean(image,K,idx)
    
    return centroids,idx

#Each pixel in the image is characterized by its RGB values, which typically range from 0 to 255. 
# Our goal is to reduce the number of unique colors in the image while preserving its visual quality.
#we will be using k-means to find the best k colors(centroids) to represent a target image with.
def CompressImage(image_path,k):
    img = Image.open(image_path)
    img = np.array(img)
    l, b, t = img.shape
    img = img.reshape(l*b,3)
    #This produces a 2 dimensional array where each element is a list of length 3 that 
    # represents the RGB values of that pixel
    clustered_colors, img_mapping = compress_image_mapping(img, k)
    np.savez_compressed(image_path + '.npz', centroids=clustered_colors, labels=img_mapping, shape=(l, b))
    
#The image is stored as a three-dimensional matrix, where the first two dimensions represent the 
# pixel coordinates, and the third dimension corresponds to the RGB values. We then reshape the 
# image matrix into a two-dimensional matrix to prepare it for K-means clustering.

def OpenImage(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    l, b, t = img.shape   
    data = np.load(image_path + '.npz')
    centroids = data['centroids']   #new k colors used to represent the compressed image
    labels = data['labels']     # each pixel is assigned(labelled) one of the k color centroids
    decluster = centroids[labels].reshape(l, b, 3) 
    image = Image.fromarray(decluster.astype(np.uint8))
    image = image.convert("RGB")
    image.show()


def run_k_means():
    while True:
        print("\n\t\tOPTIONS\n")
        print("1. COMPRESS THE IMAGE")
        print("2. OPEN COMPRESSED IMAGE")
        print("3. PRESS ANY KEY TO EXIT\n")

        option=int(input("ENTER YOUR CHOICE: "))
        if option==1:
            path=input("ENTER PATH OF THE IMAGE: ")
            k = int(input('ENTER DEGREE OF RESOLUTION (K) (4-8): '))
            print('\n')
            CompressImage(path,k)
            print("IMAGE COMPRESSED SUCCESSFULLY")

        elif option==2:
            path=input("ENTER THE PATH OF THE ORIGINAL IMAGE : ")
            OpenImage(path)
        else:
            print("\nTHANKS FOR USING!")
            break
    
run_k_means()    