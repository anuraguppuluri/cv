'''
when working with hough transforms there are a lot of parameters to tweak to get best result
there is a sample code that helps you put sliders on a test image with some parameter changes connected to that slider
that is the best/fastest way to get right parameters 
u there ? 
The attached code is an easy way to test parameters for functions you want to try and see it live to check what parameters work best 
AREO (10:00 PM):
Change path to image and give a proper image with circles to see it work
'''
import os
import cv2
import numpy as np
#import pinhole-projection as pp

#Function documentation can be found in https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar

'''
    Parameters medianblurr
image	8-bit, single-channel, grayscale input image.
circles	output vector of found circles(cv.CV_32FC3 type). Each vector is encoded as a 3-element floating-point vector (x,y,radius) .
method	detection method(see cv.HoughModes). Currently, the only implemented method is HOUGH_GRADIENT
dp	inverse ratio of the accumulator resolution to the image resolution. For example, if dp = 1 , the accumulator has the same resolution as the input image. If dp = 2 , the accumulator has half as big width and height.
minDist	minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
param1	first method-specific parameter. In case of HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
param2	second method-specific parameter. In case of HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
minRadius	minimum circle radius.
maxRadius	maximum circle radius.
    Parameters BilateralFilter
src	source 8-bit or floating-point, 1-channel or 3-channel image.
dst	output image of the same size and type as src.
d	diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
sigmaColor	filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
sigmaSpace	filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough. When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
'''

def nothing(x):
    pass

def createImg():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img,(100,100),30,(255,255,255),50)
    img = cv2.medianBlur(img,5)
    return img

current_path = os.getcwd()
print(current_path)
sub_folder = "pinhole-projection"
image_name = "test_image_hc.png"
img = cv2.imread(os.path.join(current_path, sub_folder, image_name))


mindist = "mindist"
minrad  = "minrad"
maxrad  = "maxrad"
accum = "accum"
cannyp = "cannyp"
medfiltsize = "medfiltsize"
bilatfiltsize = "bilatfiltsize"
sigmaColor = "sigmaColor"
sigmaSpace = "sigmaSpace" 

window = "Params"
cv2.namedWindow(window)
cv2.createTrackbar(mindist, window, 10, 100, nothing)
cv2.createTrackbar(minrad, window, 10, 150, nothing)
cv2.createTrackbar(maxrad, window, 120, 150, nothing)
cv2.createTrackbar(cannyp, window, 23, 100, nothing)
cv2.createTrackbar(accum, window, 23, 100, nothing)
#cv2.createTrackbar(medfiltsize, window, 5, 100, nothing) 
medfiltsize = 5
#cv2.createTrackbar(bilatfiltsize, window, 12, 100, nothing) 
bilatfiltsize = 12
#cv2.createTrackbar(sigmaColor, window, 132, 200, nothing) 
sigmaColor = 132
#cv2.createTrackbar(sigmaSpace, window, 132, 200, nothing) 
sigmaSpace = 132

while(1):
    temp = img.copy()
    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break
    mindist = cv2.getTrackbarPos('mindist', 'Params')
    minrad  = cv2.getTrackbarPos('minrad', 'Params')
    maxrad = cv2.getTrackbarPos('maxrad', 'Params') 
    accum = cv2.getTrackbarPos('accum','Params')
    cannyp = cv2.getTrackbarPos('cannyp','Params')
    #medfiltsize = cv2.getTrackbarPos('medfiltsize','Params')
    #bilatfiltsize = cv2.getTrackbarPos('bilatfiltsize','Params')
    #sigmaColor  = cv2.getTrackbarPos('sigmaColor', 'Params') 
    #sigmaSpace  = cv2.getTrackbarPos('sigmaSpace', 'Params')

    if (medfiltsize % 2) == 0: 
        medfiltsize = medfiltsize + 1; 
        cv2.setTrackbarPos('medfiltsize','Params', medfiltsize) 

    # test anything other than Hough Circles as well - any function that takes in parameters
    # Noise filters
    # image_median_blurr = cv2.medianBlur(temp, medfiltsize)
    # Part 1 : bilatfiltsize = 12, sigmaColor = 132, sigmaSpace = 132 with 
    image_bilateral_filt = cv2.bilateralFilter(temp, bilatfiltsize, sigmaColor, sigmaSpace)
    gray = cv2.cvtColor(image_bilateral_filt, cv2.COLOR_BGR2GRAY)
    # Part 1: minDist=10, cannyp=50, accum=23, minRadius=13, maxRadius=26        
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=mindist,
                              param1=cannyp,
                              param2=accum,
                              minRadius=minrad,
                              maxRadius=maxrad)

    # draw Hough circles on top the image
    if circles is not None:
        for i in circles[0,:]:
            cv2.circle(image_bilateral_filt,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(image_bilateral_filt,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow("Params", image_bilateral_filt)