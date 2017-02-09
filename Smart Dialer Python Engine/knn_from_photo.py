import cv2
import numpy as np

#function for showing contours
def show_image(img, name, contours):
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow(name, img)

#image resize function
def img_resize(obj_img):
    (h, w) = obj_img.shape
    hw = max(w, h)
    res2 = np.zeros((hw, hw), np.uint8)
    res2[(hw-h)/2:(hw+h)/2, (hw-w)/2:(hw+w)/2] = obj_img
    return cv2.resize(res2, (img_x, img_y), interpolation=cv2.INTER_CUBIC)

#size of characters
img_x = 30
img_y = 45

#images for knn training, 40 numbers for all
images = ["zero_training.jpeg", "zero_training2.jpeg", "one_training.jpeg", "one_training2.jpeg", "two_training.jpeg",
          "two_training2.jpeg", "three_training.jpeg", "three_training2.jpeg", "four_training.jpeg",
          "four_training2.jpg",
          "five_training.jpeg", "five_training2.jpg", "six_training.jpeg", "six_training2.jpeg", "seven_training.jpeg",
          "seven_training2.jpg", "eight_training.jpeg", "eight_training2.jpg", "nine_training.jpeg",
          "nine_training2.jpeg",
          "plus_training.jpeg", "plus_training2.jpeg", "slash_training.jpeg", "slash_training2.jpeg"]
train_array = np.zeros((len(images) * 20, img_x*img_y), np.float32)
i = 0
for img in images:
    one = cv2.imread("images/" + img)
    print(img)
    gray_img = 255 - cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)
    gray_original = gray_img.copy()
    ret, thresh_img = cv2.threshold(gray_img, 127, 255, 0)
    kernel = np.ones((10, 10), np.uint8)
    thresh = cv2.dilate(thresh_img, kernel, iterations=1)

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #show_image(one, img, contours)

    for cont in contours:
        num = cont.copy()
        cx, cy, cw, ch = cv2.boundingRect(num)
        forResize = gray_original[cy:cy + ch, cx:cx + cw]
        #to avoid spots
        if cw > 8 and ch > 8:
            #resized = cv2.resize(forResize, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
            resized = img_resize(forResize)

            #cv2.imshow("resized"+str(i), resized)

            train_array[i] = resized.reshape(-1, img_x*img_y).astype(np.float32)

            i = i + 1

print train_array.shape

#labels for knn training
num_labels = np.arange(12)          #10='+'  11='/'   12='-'
train_labels = np.repeat(num_labels, 40)[:, np.newaxis]
print train_labels.shape

#knn training
knn = cv2.ml.KNearest_create()
knn.train(train_array, cv2.ml.ROW_SAMPLE, train_labels)

#test 1
test = cv2.imread("images/test7.jpg")
test = cv2.resize(test, (900, 400), interpolation=cv2.INTER_CUBIC)
gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)   #POKUSATI SA 255 - cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)
gray_original = gray_test.copy()
 #one_threshold = cv2.adaptiveThreshold(gray_one, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
test_threshold = cv2.adaptiveThreshold(gray_test, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,8)
_, test_threshold = cv2.threshold(test_threshold, 127, 255, 0)
test_threshold_original = test_threshold.copy()
kernel = np.ones((3, 3), np.uint8)
test_threshold = cv2.dilate(255-test_threshold, kernel, iterations=1)
cv2.imshow("aaa", test_threshold)
_, contours, _ = cv2.findContours(test_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
show_image(test,"testtt", contours)

final_result = {}
k = 0
for cont_test in contours:
    num_test = cont_test.copy()
    cx, cy, cw, ch = cv2.boundingRect(num_test)
    forResize_test = 255 - test_threshold_original[cy:cy + ch, cx:cx + cw]
    if (cw > 10 and ch > 10) or (cw/ch > 3):
        if cw/ch > 1.2:
            final_result[cx] = 12

        else:
            #resized_test = cv2.resize(forResize_test, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
            resized_test = img_resize(forResize_test)
            cv2.imshow("resized"+str(k), resized_test)
            resized_for_knn = resized_test.reshape(-1, img_x*img_y).astype(np.float32)

            #returnVal, result,neighbours,dist = knn.findNearest(resized_for_knn,5)
            returnVal, result, neighbours, dist = knn.findNearest(resized_for_knn, 3)
            final_result[cx] = result[0][0]

            k = k + 1

#print final result
keys = final_result.keys()
keys = np.sort(keys)
for key in keys:
    if final_result[key] == 10:
        print '+'
    elif final_result[key] == 11:
        print '/'
    elif final_result[key] == 12:
        print '-'
    else:
        print final_result[key]

cv2.waitKey(0)
