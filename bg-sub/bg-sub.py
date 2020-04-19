import os
import cv2 as cv
import sys
import numpy as np

# PATHS

# get current dir so it can work on any os
# asuming that frames dir is present in the same dir as this code 
path = os.getcwd()
print("Current Directory:", path, "\n")

# no need to get parent dir 
parent = os.path.dirname(path)
print("Parent directory:", parent, "\n")

'''
zero-in on the correct file(s)
'''
# for part 1
path_to_frames = os.path.join(path, "frames")
part1_path = os.path.join(path_to_frames, "000000.jpg")

# for part 2
path_to_frames = os.path.join(path, "frames")
part2_path = os.path.join(path_to_frames, "%06d.jpg")

# check if the final paths are correct
print(os.path.isfile(part1_path))
print(os.path.isfile(part2_path))




def part1():
    # read the image as a cv::Mat object and show it
    img = cv.imread(part1_path)
    if img is None:
        sys.exit("Could not read the image.")
    cv.imshow("part1 first frame color", img)

    # wait forever until user presses any key; return value is the key pressed
    k = cv.waitKey(0)

    # print the shape of the cv::Mat object (a numpy array) containing the image
    print(img.shape)
    print("rows: " + str(img.shape[0]))
    print("cols: " + str(img.shape[1]))
    print("channels: " + str(img.shape[2]))

    # double check rows: 
    print("rows: ") 
    print(len(img[:,0,0]))
    print(len(img[0,:,0]))
    print(len(img[0,0,:]))

    # printing the image itself
    print(img)

    # convert the image to grayscale, show it and wait
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('part1 first frame gray', gray)
    k = cv.waitKey(0)

    # save gray to a png
    cv.imwrite("part1_gray.png", gray)




def part2():
    # list of grayscale frames to be averaged
    gray_list = []

    # set up video capture from frames dir
    cap = cv.VideoCapture(part2_path)

    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # operations on the frame
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_list.append(gray)

        # display the original frame
        cv.imshow('part2 original video', frame)
        # apparently cv.waitkey(1) = 25 milliseconds 
        if cv.waitKey(100) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows() 

    # perform moving-object subtraction 
    avg_img = np.mean(gray_list, axis=0)
    avg_img = avg_img.astype('uint8')

    cv.imshow('part2 averaged image', avg_img)
    k = cv.waitKey(0)

    cv.imwrite("part2_gray.png", avg_img)




def part3():
    im1 = cv.imread("part1_gray.png", cv.IMREAD_GRAYSCALE)
    if im1 is None:
        sys.exit("Could not read the image.")

    im2 = cv.imread("part2_gray.png", cv.IMREAD_GRAYSCALE)
    if im2 is None:
        sys.exit("Could not read the image.")

    # the absolute difference between the grayscale background image and the grayscale first frame 
    abs_diff = cv.absdiff(im2, im1)
    cv.imshow('part3 absolute difference', abs_diff)
    k = cv.waitKey(0)

    # apply binary thresholding to the previous absolute difference image
    # this simple thresholding returns two outputs: the threshold that was used and the thresholded image
    ret1, thresh1 = cv.threshold(abs_diff, 40, 255, cv.THRESH_BINARY)
    cv.imshow('part3 simple thresh', thresh1)
    k = cv.waitKey(0)

    # apply Otsu's thresholding to the absolute difference image
    # Otsu's thresholding returns two outputs: the optimal threshold value and the thresholded image
    ret2, thresh2 = cv.threshold(abs_diff, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('part3 Otsu\'s thresh', thresh2)
    k = cv.waitKey(0)




def bonus():
    # get the background image for doing absolute difference with each frame
    im2 = cv.imread("part2_gray.png", cv.IMREAD_GRAYSCALE)
    if im2 is None:
        sys.exit("Could not read the image.")

    # set up video capture from frames dir
    cap = cv.VideoCapture(part2_path)

    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # turn the frame into grayscale
        im1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # absolute difference
        abs_diff = cv.absdiff(im2, im1)

        # thresholding
        ret, thresh = cv.threshold(abs_diff, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        #ret, thresh = cv.threshold(abs_diff, 40, 255, cv.THRESH_BINARY)
        #ret, thresh = cv.threshold(abs_diff, 40, 255, 0)

        # find the contours in the thresholded frame
        #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # apparently, we don't have to draw the contours back onto the thresholded frame
        #ct_frame = cv.drawContours(thresh, contours, -1, (255,255,255), 3)
        
        # but instead we are supposed to draw the bounding boxes onto the thresholded frame
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            ct_frame = cv.rectangle(thresh,(x,y),(x+w,y+h),(255,255,255),1)
        
        # display the bounding-boxed, thresholded frame
        cv.imshow('bonus boxed thresh frame', ct_frame)
        if cv.waitKey(100) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows() 



part1()
part2()
part3()
bonus()
