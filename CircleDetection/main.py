
import cv2
import math
from my_hough import Hough_transform
from  my_canny import Canny

Path = "./picture_source/picture.jpg"
Save_Path = "./picture_result/"
Reduced_ratio = 2
Guassian_kernal_size = 3
HT_high_threshold = 45
HT_low_threshold = 25
Hough_transform_step = 6
Hough_transform_threshold = 110

if __name__ == '__main__':
    img_gray = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
    img_RGB = cv2.imread(Path)
    y, x = img_gray.shape[0:2]
    img_gray = cv2.resize(img_gray, (int(x / Reduced_ratio), int(y / Reduced_ratio)))
    img_RGB = cv2.resize(img_RGB, (int(x / Reduced_ratio), int(y / Reduced_ratio)))
    # canny takes about 40 seconds
    print ('Canny ...')
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    CV_canny = cv2.Canny(img_gray, 25, 45)
    cv2.imwrite(Save_Path + "CV:canny_result.jpg", CV_canny)



    canny = Canny(Guassian_kernal_size, img_gray, HT_high_threshold, HT_low_threshold)
    canny.canny_algorithm_s()
    cv2.imwrite(Save_Path + "canny_result_s.jpg", canny.img)
    print("canny_result_s")
    canny = Canny(Guassian_kernal_size, img_gray, HT_high_threshold, HT_low_threshold)
    canny.canny_algorithm()
    cv2.imwrite(Save_Path + "canny_result.jpg", canny.img)
    print("canny_result")
    # hough takes about 30 seconds
    print ('Hough ...')

    cv_circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT ,1, 5,
                         param1=100)

    for circle in cv_circles[0,:]:
        cv2.circle(img_RGB, (int(circle[0]), int(circle[1])), int(circle[2]), (0,0,155))
    cv2.imwrite(Save_Path + "CV：hough_result.jpg", img_RGB)

    Hough = Hough_transform(canny.img, canny.angle, Hough_transform_step, Hough_transform_threshold)
    circles = Hough.Calculate()
    for circle in circles:
        cv2.circle(img_RGB, (math.ceil(circle[0]), math.ceil(circle[1])), math.ceil(circle[2]), (0,0,255))
    cv2.imwrite(Save_Path + "hough_result.jpg", img_RGB)
    print ('Finished!')
