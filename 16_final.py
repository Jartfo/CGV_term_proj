# python 3.7
# opencv 3.4.2.16


import numpy as np
import cv2
import math


class ROI:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.top_left = (0, 0)
        self.bottom_right = (1, 1)
        self.roiImg = []
        self.gray_img = []
        self.kp, self.des = [], []

    def img_to_gray(self):
        self.gray_img = cv2.cvtColor(self.roiImg, cv2.COLOR_BGR2GRAY)

    def making_keyP_desc(self):
        self.kp, self.des = cv2.xfeatures2d.SIFT_create().detectAndCompute(self.gray_img, None)


roiList = []  # 검색할 모델들의 배열
roiCount = 0  # 모델 갯수
drawing = False  # 모델생성여부
color = [(0, 0, 255), (0, 255, 0), [255, 255, 255]]




def mouse_drawing(event, x, y, flags, params):
    global drawing, model, roiList, roiCount


    # 마우스 드래그 시작, roiList 에 새로운 roi 추가
    if event == cv2.EVENT_LBUTTONDOWN:
        roiList.append(ROI())
        roiList[roiCount].top_left = (x, y)
        drawing = True

    # 드래그 중
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copied = model.copy()
            cv2.rectangle(img_copied, roiList[roiCount].top_left, (x, y), color[roiCount], 1)
            cv2.imshow('Model', img_copied)

    # 드래그 끝, roi 설정
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = roiList[roiCount]
        cv2.rectangle(model, roi.top_left, (x, y), color[roiCount], 1)
        cv2.imshow('Model', model)

        # ROI 값들 설정
        roi.bottom_right = (x, y)
        roi.roiImg = model[roi.top_left[1]:roi.bottom_right[1], roi.top_left[0]:roi.bottom_right[0]]
        roi.height, roi.width = roi.roiImg.shape[:2]
        roi.img_to_gray()
        roi.making_keyP_desc()


        roiCount = roiCount + 1
        print("모델 영역을 추가하기 위해서는 드래그를,\n매칭단계 진행을 위해서는 스페이스바를 눌러주세요")


sift = cv2.xfeatures2d.SIFT_create()


# roi(모델영상)을 검색영상과 매칭
def matching(roi, tar_kp, tar_des):
    # BF Matcher 를 이용해 매칭
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(roi.des, tar_des, k=2)

    good = []  # 거리가 가까운 (==좋은) 매칭점들을 저장하는 배열
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 좋은 매칭점의 개수가 기준점 이하면 매칭 실패로 간주
    MIN = 10
    if len(good) < MIN:
        print("Match Miss")
        return 0, 0

    # 검색영상의 매칭 중심을 찾기
    src_pts = np.float32([roi.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([tar_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    pts = np.float32([[0, 0], [0, roi.height - 1], [roi.width - 1, roi.height - 1], [roi.width - 1, 0]]).reshape(-1, 1,
                                                                                                                 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 타겟 이미지의 중심좌표를 저장, 출력
    dx = math.floor((dst[0][0][0] + dst[1][0][0] + dst[2][0][0] + dst[3][0][0]) / 4)
    dy = math.floor((dst[0][0][1] + dst[1][0][1] + dst[2][0][1] + dst[3][0][1]) / 4)

    return dx, dy


# 모델영상
model = cv2.imread("models_with_black.png")
model = cv2.blur(model, (3, 3))

sift = cv2.xfeatures2d.SIFT_create()

cv2.namedWindow("Model")
# color = [(0, 0, 255), (0, 255, 0), [255, 255, 255]]
cv2.putText(model, "first : red", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
            (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(model, "second : green", (230, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
            (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(model, "third : white", (530, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
            (255, 255, 255), 2, cv2.LINE_AA)

cv2.setMouseCallback("Model", mouse_drawing)

cv2.imshow("Model", model)
print("모델 영역을 드래그하세요\n")

while True:
    if cv2.waitKey(1) == 32:
        break

# 검색영상
path = "video.mp4"
file = cv2.VideoCapture(path)

dx, dy = -50, -50  # 매칭점의 x,y 위치
dx_b, dy_b = [], []  # 각 모델의 dx, dy 백업값 (매칭 실패시 사용)
center = []  # 각 모델을 표시할 원의 중심, (dx,dy)

# 선택된 모델 갯수만큼 배열을 늘림
for i in range(roiCount):
    center.append((0, 0))
    dx_b.append(-50)
    dy_b.append(-50)

frame = 0
matchFailCount = 0



first_select_count = 0
second_select_count = 0
third_select_count = 0
while True:
    ret, target = file.read()
    if not ret:
        break
    #color = [(0, 0, 255), (0, 255, 0), [255, 255, 255]]

    cv2.putText(target, "first select: {}".format(first_select_count), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(target, "second select: {}".format(second_select_count), (100, 200), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(target, "third select: {}".format(third_select_count), (100, 300), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

    # 연산량을 줄이기 위해 크기조절
    target = cv2.resize(target, (600, 450))
    print("target shape", target.shape)

    # 동영상의 keypoint, descriptor 획득
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    tar_kp, tar_des = sift.detectAndCompute(target_gray, None)

    # 연산량을 줄이기 위해 3프레임마다 연산
    if frame % 3 == 0:
        # 각각의 모델들과 grayscale 된 동영상을 매칭
        for i in range(roiCount):

            center[i] = (0, 0)
            dx, dy = matching(roiList[i], tar_kp, tar_des)

            # 매칭실패시 백업값으로 복구
            if dx == 0 and dy == 0:
                dx = dx_b[i]
                dy = dy_b[i]
                center[i] = dx, dy
                matchFailCount = matchFailCount + 1

                if i == 0:
                    first_select_count = 0
                if i == 1:
                    second_select_count = 0
                if i == 2:
                    third_select_count = 0

                # 5번 연속 매칭실패했다면 사라진것으로 간주, dx dy 와 백업값 초기화
                if matchFailCount > 4:
                    dx = dy = dx_b[i] = dy_b[i] = -50

            # 매칭 성공시 백업값 갱신
            else:
                dx_b[i] = dx
                dy_b[i] = dy
                center[i] = dx, dy
                matchFailCount = 0
                # 다른 모델과 겹치지 않도록 gray 이미지에서 매치된 부분의 화소를 비움
                cv2.circle(target_gray, (math.floor(dx), math.floor(dy)), 25, (0, 0, 0), cv2.FILLED)
                if i == 0:
                    first_select_count = 1
                if i == 1:
                    second_select_count = 1
                if i == 2:
                    third_select_count = 1


    # 매칭된 이미지를 원으로 표시함
    for i in range(roiCount):
        target = cv2.circle(target, (center[i][0], center[i][1]), 25, color[i], 2)

    cv2.imshow("frame", target)
    frame = frame + 1

    if cv2.waitKey(1) == 32:
        break
