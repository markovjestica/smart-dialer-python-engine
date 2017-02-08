import cv2
import numpy as np

#function for showing contours
def show_image(img, name, contours):
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow(name, img)

#images for knn training, 20 numbers for all
images = ["zero_training.jpeg","one_training.jpeg","two_training.jpeg","three_training.jpeg","four_training.jpeg","five_training.jpeg","six_training.jpeg","seven_training.jpeg","eight_training.jpeg","nine_training.jpeg",
          "minus_training.jpeg", "plus_training.jpeg", "slash_training.jpeg"]
train_array = np.zeros((260, 1125), np.float32)
i = 0
for img in images:
    one = cv2.imread("images/" + img)
    gray_img = 255 - cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)
    gray_original = gray_img
    ret, thresh_img = cv2.threshold(gray_img, 127, 255, 0)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh_img, kernel, iterations=1)

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #show_image(one, img, contours)

    for cont in contours:
        num = cont
        cx, cy, cw, ch = cv2.boundingRect(num)
        forResize = gray_original[cy:cy + ch, cx:cx + cw]
        #to avoid spots
        if cw > 8 and ch > 8:
            resized = cv2.resize(forResize, (25, 45), interpolation=cv2.INTER_CUBIC)

            # cv2.imshow("resized"+str(i), resized)

            train_array[i] = resized.reshape(-1, 1125).astype(np.float32)

            i = i + 1

print train_array.shape

#labels for knn training
num_labels = np.arange(13)          #10='-'  11='+'   12='/'
train_labels = np.repeat(num_labels, 20)[:, np.newaxis]
print train_labels.shape

#knn training
knn = cv2.ml.KNearest_create()
knn.train(train_array, cv2.ml.ROW_SAMPLE, train_labels)

#test 1
test_img1 = cv2.imread("images/test5.jpeg")
gray_test = 255 - cv2.cvtColor(test_img1, cv2.COLOR_BGR2GRAY)
gray_test_original = gray_test
retVal, thresh_test = cv2.threshold(gray_test, 127, 255, 0)
kernel = np.ones((5, 5), np.uint8)
thresh_test_img = cv2.dilate(thresh_test, kernel, iterations=1)

_, contours_test, _ = cv2.findContours(thresh_test_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
show_image(test_img1, "test1", contours_test)

final_result = {}
k = 0
for cont_test in contours_test:
    num_test = cont_test
    cx, cy, cw, ch = cv2.boundingRect(num_test)
    forResize_test = gray_test_original[cy:cy + ch, cx:cx + cw]
    if cw > 8 and ch > 8:
        resized_test = cv2.resize(forResize_test, (25, 45), interpolation=cv2.INTER_CUBIC)
        #cv2.imshow("resized"+str(k), resized_test)
        resized_for_knn = resized_test.reshape(-1, 1125).astype(np.float32)

        #returnVal, result,neighbours,dist = knn.findNearest(resized_for_knn,5)
        returnVal, result, neighbours, dist = knn.findNearest(resized_for_knn, 4)
        final_result[cx] = result[0][0]

        k = k + 1

#print final result
keys = final_result.keys()
keys = np.sort(keys)
for key in keys:
    if final_result[key] == 10:
        print '-'
    elif final_result[key] == 11:
        print '+'
    elif final_result[key] == 12:
        print '/'
    else:
        print final_result[key]

cv2.waitKey(0)
