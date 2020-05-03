# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import sys

#Set up the arguments for giving image location and filtering parameters
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image")
ap.add_argument("-s","--size", type=float, default=1, help="Fraction of original image size outputted, default is 1")
ap.add_argument("--smoothing",nargs=2, type=int, default=[13,50], help="adjust the strength of the smoothing, default is 13 50")
ap.add_argument("-p","--pass", type=int, default=0, help="set the high pass filter threshold")
ap.add_argument("-f","--filter", type=str, default='pyr', help="set the type of filtering before the threshold used")
args = vars(ap.parse_args())
#This adjusts the scaling of the output image, since different images require different scaling
frac=args['size'] 
#This variable is the pass limit for the highpass filter.
highpass = args['pass'] 
#This dictionary is used for determining the type of filtering used: Pyramid mean shifts or the laplacian of the Gaussian
filterdict = {'pyr':0,'pyramid':0, 'pyramidmeanshift':0,'meanshift':0, 'mean':0, 'pyrmean':0, 
              'laplacianofgaussian':1, 'laplace':1, 'lap':1, 'gauss':1}

#Tries the dictionary with the argument given. If it doesn't match it errors
try: 
    filtercheck = filterdict[args['filter']]
#If it errors, it instead goes to the except, keeping the program from halting
except: 
    print('Filter choice failed. Defaulting to Pyramid Mean Shift')
    filtercheck = 0

#Load in the image from the given path
image = cv2.imread(args["image"])
#Store the original image in a place to be output in the final window, resized by given scale factor
row1 = cv2.resize(image, (0, 0), None, frac, frac)


#If a value is given for the pass filter, it then applies the highpass filter
#Otherwise, it just skips over the routine
if(highpass!=0):
    passed = image.copy()
    #Counting variables to give an idea of how much of the image was actaully filtered.
    n=0
    m=0
    #Looks at each pixel, and limits it to just the brightest of the three colors to use as the effective brightness
    max_inten = np.max(image, axis=2)
    #Looks through the whole array, and creates a mask, true for each value with brightness less than the pass threshold
    mask = np.ma.masked_less_equal(max_inten, highpass).mask

    #Loops over every pixel, and if the mask is true, sets the pixel colors to all 0s
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if(mask[x,y]):
                passed[x,y]=[0,0,0]
                n+=1
            else:
                m+=1
    print(f'masked {n} entries and left {m} untouched')
#The next effect uses the name passed, so if the highpass isn't used
else:
    passed = image


#Filtercheck 0 is the value corresponding to the pyramid mean shift
if(filtercheck==0):
    #Apply the pyramid mean shift and then store the shifted image to be output at the end
    shifted = cv2.pyrMeanShiftFiltering(passed, args['smoothing'][0], args['smoothing'][1])
    row1 = np.hstack((row1, cv2.resize(shifted, (0, 0), None, frac, frac)))     
    
#Filtercheck 1 is the value for the laplacian of the Gaussian
#This needs more work, since it doesn't work with the watershed, since it only highlights edges. 
#Given the extra work required, and the fact that the Watershed seemed to be working, this part was
#set aside for now, though we may experiment with it more in the future.
elif(filtercheck==1):
    #Gaussian kernel parameters, kept as variables incase they are to be made an argument
    kwide = 3
    khigh = 3
    #Apply the Gaussian Blur, then take the laplcian of the blurred image
    blur = cv2.GaussianBlur(image,(kwide,khigh), 0)
    laplace = cv2.Laplacian(blur, -1)
    #Store the image to be output at the end
    row1 = np.hstack((row1, cv2.resize(laplace, (0, 0), None, frac, frac)))  

#Failsafe else statement, incase filtercheck is not sucessfully set to a valid value
else:
    print("This message should never be seen... Uh Oh!")
    sys.exit()
    
#Gray scale the image, then apply the Otsu Thresholding method
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#Store the thresholded image for final output
row2 = cv2.resize(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), (0, 0), None, frac, frac)


#Create a euclidiean distance map of the thresholded image. This the value at each 
#non-zero pixel the distance from the nearest 0 valued pixel
D = ndimage.distance_transform_edt(thresh)
#Find all the local maxes, which correspond to the centers of each star
localMax = peak_local_max(D, indices=False, min_distance=5,
	labels=thresh)



#Take each local max and label them in order so they each have an index
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
#Perform a watershed on the negative of the distance function, using each local max
#now local min as a source point for the 'water'
labels = watershed(-D, markers, mask=thresh)
print("{} unique stars found".format(len(np.unique(labels)) - 1))

#Loop over each label found (All grouped pixels)
for label in np.unique(labels):
	#The 0 label is the background, so we ignore it
	if label == 0:
		continue
    
    #Find just the group of pixels
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255
    
	#detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
    
	#draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    #Possibly add numbers if selected
	#cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)

#show the output images 
row2 = np.hstack((row2,cv2.resize(image, (0, 0), None, frac, frac)))
output = np.vstack((row1,row2))    
cv2.imshow("Output", output)
cv2.waitKey(0)