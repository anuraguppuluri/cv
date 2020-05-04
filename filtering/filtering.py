import sys
import cv2 as cv
import numpy as np
import os


# all functions defined here can be called in main

def hybrid_img(im1_path, im2_path):
    
    '''
    STEP 1
    '''

    # read in both images
    im1 = cv.imread(im1_path, cv.IMREAD_COLOR)
    im2 = cv.imread(im2_path, cv.IMREAD_COLOR)
    if im1 is None or im2 is None:
        sys.exit("Could not read either image.")
    
    '''
    eventhough bmp potentially supports multiple bit formats i.e. 2^0, 2^1, 2^2, 2^3 bits etc.,
    it is assumed that bmp images are still read in 8 bit int format by cv2.imread
    '''

    # first converting input images to 32 bit float format 
    im1.astype('float')
    im2.astype('float')

    '''
    the values have remained the same
    and so its time to convert them all to the [0,1] floating point range
    '''
    #print(im1)

    im1 = np.divide(im1,255)
    im2 = np.divide(im2,255)

    #print(im1)

    cv.imshow('cat', im1)   
    k = cv.waitKey(0)

    cv.imshow('dog', im2)
    k = cv.waitKey(0)

    '''
    STEP 2
    '''

    kernel_size = 31
    # creating a low pass kernel of size 31x31 with sigma=5
    g = cv.getGaussianKernel(kernel_size, 5)
    g = np.multiply(g, g.transpose())

    # scaling up the kernel matrix just for the sake of displaying
    g_show = np.multiply(g, 255)

    cv.imshow('lp kernel', g_show)
    k = cv.waitKey(0)

    '''
    STEP 3
    '''

    # making an all pass filter where just the central cell is a 1 and all the others are 0
    a = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center_cell = int(kernel_size/2) + 1
    a[center_cell, center_cell] = 1

    # making the high pass filter
    h = a - g

    cv.imshow('hp kernel', h)
    k = cv.waitKey(0)

    '''
    STEP 4
    '''
    
    '''
    ddepth: The depth of the Mat object (image) returned by cv2.filter2D.
    A negative value (such as −1) indicates that the depth is the same as the source.
    '''
    ddepth = -1

    # filtering im1 with the high-pass kernel
    im1_hp = cv.filter2D(im1, ddepth, h)

    cv.imshow('high im1', im1_hp)
    k = cv.waitKey(0)

    # filtering im2 with the low-pass kernel
    im2_lp = cv.filter2D(im2, ddepth, g)

    cv.imshow('low im2', im2_lp)
    k = cv.waitKey(0)

    '''
    STEP 5 and BONUS
    '''

    '''
    just remember that the images have to be the same shape otherwise  
    Exception has occurred: ValueError
    operands could not be broadcast together with shapes (1000,1000,3) (1500,1500,3) 
    ''' 
    # the hybrid image
    img_hy = im1_hp + im2_lp

    cv.imshow('hybrid', img_hy)
    k = cv.waitKey(0)



def main():
    print(len(sys.argv))

    if len(sys.argv) < 3:
        print("Not enough arguments: usage as below")
        print("python file_name.py 'path1' 'path2'")
        return

    # check if both the image paths are correct
    print(sys.argv[1])
    print(sys.argv[2])
    
    print(os.path.isfile(sys.argv[1]))
    print(os.path.isfile(sys.argv[2]))

    # the logic of the code will begin when len(sys.argv) is what we need it to be
    print("Path1 image will be high passed, path2 image will be low passed.")
    hybrid_img(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()