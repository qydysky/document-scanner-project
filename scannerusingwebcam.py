import cv2
import numpy as np
import utilis

webcam= True
path="testimage.jpg"
cap=cv2.VideoCapture(0)
cap.set(10,160)
imgheight=500
imgwidth=480

utilis.initializeTrackbars()
count=0

while True:


    if webcam:success, img=cap.read()
    else: img=cv2.imread(path)
    img=cv2.resize(img,(imgwidth,imgheight))
    imgblank = np.zeros((imgheight, imgwidth, 3), np.uint8)
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    thres=utilis.valTrackbars()
    imgthres=cv2.Canny(imgBlur,thres[0],thres[1])
    kernel=np.ones((5,5))
    imgdial=cv2.dilate(imgthres,kernel,iterations=2)
    imgthres=cv2.erode(imgdial,kernel,iterations=2)

    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    biggest, maxArea = utilis.biggestContour(contours)
    if biggest.size != 0:
        biggest = utilis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utilis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [imgwidth, 0], [0, imgheight], [imgwidth, imgheight]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (imgwidth, imgheight))

        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (imgwidth, imgheight))

        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        imageArray = ([img, imgGray, imgthres, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])


    else:
        imageArray = ([img, imgGray, imgthres, imgContours],
                  [imgblank, imgblank, imgblank, imgblank])


    lables = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utilis.stackImages(imageArray, 0.75)
    cv2.imshow("Result", stackedImage)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1



