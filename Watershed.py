# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image")
ap.add_argument("-s","--size", type=float, default=.8, help="Fraction of original image size outputted, default is .6")
ap.add_argument("--smoothing",nargs=2, type=int, default=[21,87], help="adjust the strength of the smoothing, default is 21 87")
ap.add_argument("-p","--pass", type=int, default=0, help="set the low pass filter")
ap.add_argument("-f","--filter", type=str, default='pyr', help="set the type of filtering before the threshold used")
args = vars(ap.parse_args())
frac=args['size']
lowpass = args['pass']
filterdict = {'pyr':0,'pyramid':0, 'pyramidmeanshift':0,'meanshift':0, 'mean':0, 'pyrmean':0, 
              'laplacianofgaussian':1, 'laplace':1, 'lap':1, 'gauss':1}
try:
    filtercheck = filterdict[args['filter']]
except:
    print('Filter choice failed. Defaulting to Pyramid Mean Shift')
    filtercheck = 0
# load the image and perform pyramid mean shift filtering
# to aid the thresholding step
image = cv2.imread(args["image"])
row1 = cv2.resize(image, (0, 0), None, frac, frac)

if(filtercheck==0):

    if(lowpass!=0):
        passed = image.copy()
        n=0
        mean_inten = np.max(image, axis=2)
        mask = np.ma.masked_less_equal(mean_inten, lowpass).mask
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if(mask[x,y]):
                    passed[x,y]=[0,0,0]
                    n+=1
        print(f'masked {n} entries')
        #cv2.imshow("lowpass",passed)
    else:
        passed = image
    shifted = cv2.pyrMeanShiftFiltering(passed, args['smoothing'][0], args['smoothing'][1])
    row1 = np.hstack((row1, cv2.resize(shifted, (0, 0), None, frac, frac)))   

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

elif(filtercheck==1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    


else:
    print("this shouldn't happen... FUCKKKKKK")
    








thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
row2 = cv2.resize(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), (0, 0), None, frac, frac)


# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=thresh)


# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique stars found".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue
    
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255
    
	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
    
	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
	#cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
#		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# show the output images 
row2 = np.hstack((row2,cv2.resize(image, (0, 0), None, frac, frac)))
output = np.vstack((row1,row2))

    
    
    
cv2.imshow("Output", output)
cv2.waitKey(0)