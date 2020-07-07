
import numpy as np
import cv2

########################################################################


class ORB__:
    def __init__(self, file_name):
        self.file_name = file_name
        self.img = self.read_image_file(self.file_name)
        self.gray_img = self.img_to_gray()
        self.orb = cv2.ORB_create()
        self.KeyPoints, self.Descriptor = self.making_keyP_desc()
        self.top_left = (0, 0)
        self.bottom_right = (1 , 1)
        self.drawing = False
        self.roi = []

        cv2.namedWindow("Query image")

    def read_image_file(self, file_name):
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.blur(img, (3,3))
        return img

    def img_to_gray(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return img

    def making_keyP_desc(self):
        KeyPoints, Descriptor = self.orb.detectAndCompute(self.gray_img, None)
        return KeyPoints, Descriptor

    def mouse_drawing(self, event, x, y, flag, params):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.top_left = x, y
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copied = self.img.copy()
                cv2.rectangle(img_copied, self.top_left, (x, y), (0, 0, 255), 1)
                cv2.imshow('Model', img_copied)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            img_copied = self.img.copy()
            cv2.rectangle(img_copied, self.top_left, (x, y), (0, 0, 255), 1)
            cv2.imshow('Model', img_copied)
            self.bottom_right = x, y
            self.get_ROI(img_copied, self.top_left, self.bottom_right)

    def callback(self):
        while True:
            cv2.setMouseCallback("Query image", self.mouse_drawing)
            if cv2.waitKey(1) == 32:
                break

    def get_ROI(self, Query_img, top_left, bottom_right):
        self.roi = Query_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        cv2.imshow("ROI in Query image", self.roi)




def matching(query_descriptor, train_descriptor):

    # matcher = cv2.BFMatcher()
    # matches = matcher.knnMatch(query_descriptor, train_descriptor, k=2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(query_descriptor, train_descriptor)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def draw_query_boundary_in_train(query_img, query_keypoint, train_img, train_key_point, matches):

    # select top 10 matches
    good_matches = matches[:10]

    src_pts = np.float32([query_keypoint[m.queryIdx].pt for m in good_matches])
    src_pts = src_pts.reshape(-1, 1, 2)
    dst_pts = np.float32([train_key_point[m.queryIdx].pt for m in good_matches])
    dst_pts = dst_pts.reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = query_img.shape[: 2]

    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
    pts = pts.reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M)
    dst += (w, 0)

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)

    img3 = cv2.drawMatches(query_img, query_keypoint, train_img, train_key_point, good_matches, None, **draw_params)

    # Draw bounding box in Red
    img3 = cv2.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("result", img3)
    cv2.waitKey()
    # or another option for display output
    # plt.imshow(img3, 'result'), plt.show()













def main():

    query_orb = ORB__("zombie_only.png")
    query_orb.img_to_gray()
    query_orb.callback()
    query_orb.making_keyP_desc()

    train_orb = ORB__("zombie_mush.png")
    train_orb.img_to_gray()
    train_orb.img_to_gray()
    train_orb.making_keyP_desc()

    matches = matching(query_orb.Descriptor, train_orb.Descriptor)
    draw_query_boundary_in_train(query_orb.img, query_orb.KeyPoints, train_orb.img, train_orb.KeyPoints, matches)











if __name__ == "__main__":
    main()






