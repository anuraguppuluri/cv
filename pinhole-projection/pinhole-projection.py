import sys
import cv2 as cv
import numpy as np
import os



# all functions defined here can be called in main

# function to convert 3d/world points to 2d/image points
def world_to_image(cam_f, pri_p, world_p):
    # unpack principle point tuple
    (cx,cy) = pri_p

    # create intrinsics matrix
    intrinsics = np.array([[cam_f, 0, cx],[0, cam_f, cy],[0, 0, 1]])

    # get the homogeneous version of the required 2d/image point
    homo_2d_p = np.matmul(intrinsics, world_p)

    # convert to inhomogeneous by dividing all dimensions with the last one and ignoring it 
    homo_2d_p = np.divide(homo_2d_p, homo_2d_p[2])
    return (homo_2d_p[0], homo_2d_p[1])




# function to draw the projection of a line with thickness of 1 px between given 3D/world points on a given image
def draw_line(img, cam_f, pri_p, world_p1, world_p2):
    # unpack principle point tuple
    (cx,cy) = pri_p

    # convert both 3d points to 2d
    image_p1 = world_to_image(cam_f, (cx,cy), world_p1)
    image_p2 = world_to_image(cam_f, (cx,cy), world_p2)

    # integerize 2D points to draw line
    p1 = (int(image_p1[0]), int(image_p1[1]))
    p2 = (int(image_p2[0]), int(image_p2[1]))

    img = cv.line(img,p1,p2,(255,0,0),1)
    return img



def hough_cir(video_path):
    print("***********Hough Circles*************")
    print()
    
    # font for writing on the image
    font = cv.FONT_HERSHEY_SIMPLEX

    # actual world coordinates radius of the ball R in cm
    world_r = 3

    # camera intrinsic 1: focal lenght f
    cam_f = 485.82423388827533
    # camera intrinsic 2: principle point (cx,cy)
    cx = 134.875
    cy = 239.875

    # set up video capture from video_path
    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # turn the frame into grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # filtered the frame with a gaussian kernel of given size and sigma to remove the noise
        sigma = 2            
        gray = cv.GaussianBlur(gray,(9,9),sigma,sigma)

        # Hough circle transform 
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=75)
    
    
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(frame, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(frame, center, radius, (255, 0, 255), 3)

                print("Projected position of the ball in image coordinates: (" + str(i[0]) + "," + str(i[1]) + ")")

                # compute the projected position of the ball in camera coordinates i.e., sans (cx,cy) and f
                cam_x = (i[0] - cx)/cam_f
                cam_y = (i[1] - cy)/cam_f

                print("Projected position of the ball in camera coordinates: (" + str(cam_x) + "," + str(cam_y) + ")")
                
                # estimate the world coordinates depth Z in cm from the image coordinates of the ball
                world_z = (world_r/radius) * cam_f 

                # write the world coordinates depth of the ball on the frame
                cv.putText(frame,str(int(world_z))+" cm",center, font, 0.8,(255,255,255),2,cv.LINE_AA)

                # estimate the complete world coordinates position of the ball (X,Y,Z)
                world_x = cam_x * world_z
                world_y = cam_y * world_z

                print("Estimated position of the ball in world coordinates: (" + str(world_x) + "," + str(world_y) + "," + str(world_z) + ")")

                # draw a bounding cube of side lenght 2*projected-ball-radius around the projected ball in the frame 

                # connecting wireframes
                frame = draw_line(frame, cam_f, (cx,cy), (world_x-world_r, world_y-world_r, world_z-world_r), (world_x-world_r, world_y-world_r, world_z+world_r))
                frame = draw_line(frame, cam_f, (cx,cy), (world_x+world_r, world_y-world_r, world_z-world_r), (world_x+world_r, world_y-world_r, world_z+world_r))
                frame = draw_line(frame, cam_f, (cx,cy), (world_x-world_r, world_y+world_r, world_z-world_r), (world_x-world_r, world_y+world_r, world_z+world_r))
                frame = draw_line(frame, cam_f, (cx,cy), (world_x+world_r, world_y+world_r, world_z-world_r), (world_x+world_r, world_y+world_r, world_z+world_r))

                # front wireframe
                frame = draw_line(frame, cam_f, (cx,cy), (world_x-world_r, world_y-world_r, world_z-world_r), (world_x+world_r, world_y-world_r, world_z-world_r))
                frame = draw_line(frame, cam_f, (cx,cy), (world_x+world_r, world_y-world_r, world_z-world_r), (world_x+world_r, world_y+world_r, world_z-world_r))
                frame = draw_line(frame, cam_f, (cx,cy), (world_x+world_r, world_y+world_r, world_z-world_r), (world_x-world_r, world_y+world_r, world_z-world_r))
                frame = draw_line(frame, cam_f, (cx,cy), (world_x-world_r, world_y+world_r, world_z-world_r), (world_x-world_r, world_y-world_r, world_z-world_r))

                # back wireframe
                frame = draw_line(frame, cam_f, (cx,cy), (world_x-world_r, world_y-world_r, world_z+world_r), (world_x+world_r, world_y-world_r, world_z+world_r))
                frame = draw_line(frame, cam_f, (cx,cy), (world_x+world_r, world_y-world_r, world_z+world_r), (world_x+world_r, world_y+world_r, world_z+world_r))
                frame = draw_line(frame, cam_f, (cx,cy), (world_x+world_r, world_y+world_r, world_z+world_r), (world_x-world_r, world_y+world_r, world_z+world_r))
                frame = draw_line(frame, cam_f, (cx,cy), (world_x-world_r, world_y+world_r, world_z+world_r), (world_x-world_r, world_y-world_r, world_z+world_r))

        # display detected circles, estimated depth and drawn lines in the frame
        cv.imshow('Hough circle, depth and wireframe cube', frame)
        if cv.waitKey(30) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows() 

    



def main():
    print(len(sys.argv))

    if len(sys.argv) < 2:
        print("Not enough arguments: usage as below")
        print("python file_name.py 'path'")
        return

    print(sys.argv[1])
    # check if the video path is correct
    print(os.path.isfile(sys.argv[1]))

    # the logic of the code will begin when len(sys.argv) is what we need it to be
    hough_cir(sys.argv[1])

if __name__ == "__main__":
    main()