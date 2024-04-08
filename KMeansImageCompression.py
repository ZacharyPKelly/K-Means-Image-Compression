#######################################################################################################################################
# This file contains code in order to implement...
#
# Class: AUCSC 460
# Name: Zachary Kelly
# Student ID: 1236421
# Date: March 13th, 2024
#######################################################################################################################################

# IMPORTS #
from PIL import Image
import numpy as np
import sys #sys used to stop numpy from truncating the array when printing for testing purposes
np.set_printoptions(threshold=sys.maxsize)

class kMeansClustering():

    def preprocess_data(self, imagePath):

        #Opening the image and loading it into a variable.
        image = Image.open(imagePath)

        #Load the image. This closes the file after exectution.
        pixels = image.load()

        #Getting the number of pixels in the image
        self.imageWidth, self.imageHeight = image.size

        #Initializing array to hold the RBG values
        self.preprocessedImageArray = []

        #Adding the RBG values to the 2d imageArray representing the photo.
        for x in range(self.imageHeight):

            #Holds all the normalized values for one row of the image at a time
            tempPixelRowArray = [] 

            for y in range(self.imageWidth):

                #converting pixel tuple into array
                tempPixelArray = np.asarray(pixels[x,y])
                #holds the normalized RBG values for each pixel
                tempNormalizationArray = []

                for z in range(len(tempPixelArray)):

                    #Normalizing the pixels to be between 1 and 0
                    normalization = tempPixelArray[z] / 255
                    tempNormalizationArray.append(normalization)

                #Adding the normalized RBG values for the pixel to it's rows array
                tempPixelRowArray.append(tempNormalizationArray)
            
            #Adding the row of normalized pixel RBG values to the 2d array representing the photo
            self.preprocessedImageArray.append(tempPixelRowArray)

    def kMeans_init_centroid(self, numberOfCentroids = 16):

        #Initializing centroid size variable
        self.numberOfCentroids = numberOfCentroids

        #Initializing array to hold centroids
        self.centroidArray = []

        #Initializing array to check if centroids changed after assignement of pixels
        self.centroidCheck = np.zeros([16,3])

        #Setting up the centroids using random numbers between 0 and 1
        for x in range(numberOfCentroids):

            #Each centroid is three values (RBG) 'normalized' between 0 and 1 as a float
            centroid = np.random.uniform(size = 3, low = 0.0, high = 1.0)

            self.centroidArray.append(centroid)

        #Converting centroid arrays to numpy arrays
        self.centroidArray = np.array(self.centroidArray)
        self.centroidCheck = np.array(self.centroidCheck)

    def run_kMeans(self, imagePath):

        #Load in and preprocess the image
        self.preprocess_data(imagePath)

        #Initializing array to hold clustered pixels
        self.clusteredPixels = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

        #Initialize the centroids
        self.kMeans_init_centroid()
        self.find_closest_centroids()
        #iterator
        count = 1
        
        print("------------------------------")
        print("          Iterations          ")
        print("------------------------------\n")

        #while the current centroid array and previous centroid array are not equal keep iterating
        while np.array_equal(self.centroidArray, self.centroidCheck) == False:

            #Copy the current centroid values over to check after they have been changed if they are still the same or not to terminiate while loop
            self.centroidCheck = self.centroidArray.copy()

            #Calculate the centroid each pixel is closest to.
            self.find_closest_centroids()

            #compute the new position for each of the centroids
            for x in range(self.numberOfCentroids):

                self.compute_centroids(self.clusteredPixels[x], x)

            #Clear the cluster array so that we can read in pixels for next iteration
            #If the two are equal after finding closest centroids, dont wipe so we can use the clusters to apply the new colours to the image
            if np.array_equal(self.centroidArray, self.centroidCheck) == False:
            
                self.clear_cluster_array()
            
            #Print the iteration the program is on as I hate blank screens that sit and give no information
            print("Iteration: ", count)
            count = count + 1

        #Compress the image
        self.compress_image(imagePath)

    def find_closest_centroids(self):

        #iterate through all the pixels in the processed pixel array
        for x in range(self.imageHeight):

            for y in range(self.imageWidth):

                #Getting the pixel and its RBG values to be checked for distance to centroids
                pixel = self.preprocessedImageArray[x][y]

                #Will hold the index of the centroid array the closest distance will be added to
                closestCentroidIndex = None

                #Initializing the first closest distance for first check
                #Is 2 as the distance between two points will always be <= ~1.73 for first run initialization: sqrt((1 - 0)^2 + (1 - 0)^2 + (1 - 0)^2) = 1.73205080757
                closestDistance = 2

                #Iterate through and check the given pixels distance for all centroids 
                for z in range(self.numberOfCentroids):

                    #The centroid being checked.
                    centroid = self.centroidArray[z]

                    #Calculate the distance from the pixel to the centroid
                    distance = np.sqrt(np.square(pixel[0] - centroid[0]) + np.square(pixel[1] - centroid[1]) + np.square(pixel[2] - centroid[2]))

                    #If closer to current centroid then previous then save distance (to check next centroid) and centroids index (to store correctly)
                    if distance < closestDistance:

                        closestDistance = distance
                        closestCentroidIndex = z

                self.clusteredPixels[closestCentroidIndex].append([x, y])
            
    def compute_centroids(self, centroidValues, index):

        #Sum totals for the mean of each RBG value as well as the total number to calc mean
        sumR = 0
        sumB = 0
        sumG = 0
        count = 0

        #Continue if at least one pixel was clustered to the given centroid.
        if len(centroidValues) != 0:

            #For each pixel clustered to the centroid.
            for x in range(len(centroidValues)):

                #Get the pixels RBG values from the preprocessed array
                pixelIndices = centroidValues[x]
                pixel = self.preprocessedImageArray[pixelIndices[0]][pixelIndices[1]]

                sumR = sumR + pixel[0]
                sumB = sumB + pixel[1]
                sumG = sumG + pixel[2]
                count = count + 1
            
            #calculate the mean for each centroid RBG Value
            centroidR = sumR / count
            centroidB = sumB / count
            centroidG = sumG / count

            #Replace the previous centroid RBG values with the new centroid RBG values
            self.centroidArray[index] = [centroidR, centroidB, centroidG]

    def compress_image(self, imagePath):

        #Load in the image to be compressed
        compressedImage = Image.open(imagePath)

        print("\n-------------------------------")


        #For each cluster of pixels associated with the centroid
        for x in range(len(self.clusteredPixels)):

            #Convert the centroids RBG values from a range of 0 to 1 back into a range from 0 to 255 and round them
            #NumPy rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to 0.0 etc
            colourFloats = np.round(self.centroidArray[x] * 255)

            #Convert the centroid to integers (needed for putpixel method)
            colourInts = colourFloats.astype(int)

            #Convert the centroid from an array to a tuple (needed for putpixel method)
            colour = tuple(colourInts)

            #Print the colour we are changing the given pixel(s) to
            print("Colour {}: {}".format(x, colour))

            #For each pixel associated with a given centroid
            for y in range(len(self.clusteredPixels[x])):

                #Get the pixels coordinates
                pixel = self.clusteredPixels[x][y]

                #Change the colour at the given coordinate to the colour determined by the centroid
                compressedImage.putpixel((pixel[0], pixel[1]), colour)

        print("-------------------------------")

        #Save the image under a new filename to prevent overwriting
        compressedImage.save("compressedImage.png")

        #Close the file pointer
        compressedImage.close()

    def clear_cluster_array(self):          

        #For each cluster of pixels
        for x in range(len(self.clusteredPixels)):

            #remove all pixels and leave cluster (array) empty
            self.clusteredPixels[x].clear()

# DRIVER CODE #
def main():
    
    kMeansClusteringModel = kMeansClustering()

    kMeansClusteringModel.run_kMeans('testImage.png')

if __name__ == "__main__":

    main()