# python 3.7
# opencv 3.4.2.16 사용

import numpy as np
import cv2
import math

drawing = False  # 모델생성여부
top_left, bottom_right = (), ()  # 모델 roi 생성시 사용하는 좌표
roi = []
kp1, des1 = None, None  # roi의 SIFT 검출 결과로 나오는 kp와 des
roi_w, roi_h = 0, 0  # roi 너비, 높이


################################################################
# 테스트 동영상은 ppt 애니메이션으로 단순하게 만들었습니다.

# 현재 코드 한계점 정리
# 1. 다수의 모델이 있는 영상중 드래그하여 하나의 모델만 사용 가능
# 2. 선택된 모델이 검색영상에 여러 개 있을 경우 매칭 x
# 3. 연산이 많아서 속도가 느림

# 단순한 테스트만 했기에 이 외에도 오류가 더 있을것으로 생각됩니다.
##############################################################

# 모델영상에서 드래그한 영역으로 roi를 만들어냄
def makingRoi():
    global roi, kp1, des1, roi_w, roi_h
    roi_h, roi_w = roi.shape[:2]

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(roi, None)


# roi(모델영상)을 검색영상과 매칭
def matching():
    global kp1, des1, target
    target_g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(target_g, None)

    # BF Matcher를 이용해 매칭
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []  # 거리가 가까운 (==좋은) 매칭점들을 저장하는 배열
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    # 좋은 매칭점의 개수가 기준점 이하면 매칭 실패로 간주
    MIN = 10
    if len(good) < MIN:
        print("Match Miss\n")
        return 0, 0

    # 검색영상의 매칭 중심을 찾기 https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    pts = np.float32([[0, 0], [0, roi_h - 1], [roi_w - 1, roi_h - 1], [roi_w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 타겟 이미지의 중심좌표를 저장, 출력
    dx = (dst[0][0][0] + dst[1][0][0] + dst[2][0][0] + dst[3][0][0]) / 4
    dy = (dst[0][0][1] + dst[1][0][1] + dst[2][0][1] + dst[3][0][1]) / 4

    return dx, dy


# 모델영상에서 드래그하여 roi 설정
def mouse_drawing(event, x, y, flags, params):
    global top_left, bottom_right, drawing, model, roi

    # 마우스 드래그 시작
    if event == cv2.EVENT_LBUTTONDOWN:
        top_left = x, y
        drawing = True

    # 드래그 중
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img2 = model.copy()
            cv2.rectangle(img2, top_left, (x, y), (0, 0, 255), 1)
            cv2.imshow('Model', img2)

    # 드래그 끝, roi 생성
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img2 = model.copy()
        cv2.rectangle(img2, top_left, (x, y), (0, 0, 255), 1)
        cv2.imshow('Model', img2)
        bottom_right = x, y
        roi = model[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        cv2.imshow("Model_1", roi)
        makingRoi()


# 모델영상
model = cv2.imread("models.png")
model = cv2.blur(model, (3, 3))

sift = cv2.xfeatures2d.SIFT_create()

cv2.namedWindow("Model")
cv2.setMouseCallback("Model", mouse_drawing)

cv2.imshow("Model", model)
print("모델 영역을 드래그하세요\n")

while True:
    if cv2.waitKey(1) == 32:
        break

##################################################################

# 검색영상
path = "test2.mp4"
file = cv2.VideoCapture(path)
dx, dy = 0, 0

while True:
    ret, target = file.read()
    if not ret:
        break
    target = cv2.resize(target, (720, 480))

    if matching() != (0, 0):
        dx, dy = matching()

    # 매칭된 이미지를 원으로 표시함
    target = cv2.circle(target, (math.floor(dx), math.floor(dy)), 50, (0, 255, 0), 2)
    cv2.imshow("frame", target)

    if cv2.waitKey(1)==32:
        break
