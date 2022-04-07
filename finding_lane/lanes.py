import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height), (1100,height), (550,250)]]) # array of polygons
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def canny(lane_image):
    # convert image to grayscale
    gray = cv2.cvtColor(lane_image,
                        cv2.COLOR_RGB2GRAY)  # we work with gray image as processing is faster that 3 channel image
    # when you apply canny() function, this step will become optional as it automatically apply 5*5 gaussian kernel to our image
    # reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # convolve the gray image by 5*5 gaussian kernel with deviation of zero
    # Edge detection
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def video_lane(video):
    while video.isOpened():
        _, frame = video.read()
        image_lane(frame, 1)
        if cv2.waitKey(1) == ord('q'):
            break
        """try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('q'):  # if key 'q' is pressed
                print('You Pressed A Key!')
                break  # finishing the loop
        except:
            break"""

def image_lane(image, t):
    # copy image
    copy_image = np.copy(image)  # to avoid changes be reflected on original image
    canny_image = canny(copy_image)
    masked_canny = region_of_interest(canny_image)
    """Hough transform
        cv2.HoughLinesP(image that you want to detect lines,
                        ro = distance resolution of Hough accumilated array in pixels (grid),
                        teta = angle resolution of accumalator in radians,
                        threshold = minimum number of votes to consider it as line  describing our points,
                        placeholder array = here an empty array,
                        minimum length of a line in pixel that we accept,
                        maximum distance betwenn segmented lines in pixel that we allow to be connected to 
                        make a single line instead of beeing broken,
    )"""
    lines = cv2.HoughLinesP(masked_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = display_lines(copy_image, lines)
    blended_image = cv2.addWeighted(copy_image, 0.8, line_image, 1, 1)
    # show image
    cv2.imshow('show', blended_image)
    if t == 0:
        cv2.waitKey(t)

def main():
    # load image
    image = cv2.imread('test_image.jpg')
    #image_lane(image, 0)
    video = cv2.VideoCapture('./test.mp4')
    video_lane(video)
    # plt.imshow(canny)
    # plt.show()
    
if __name__ == "__main__":
    main()