import cv2
import easyocr
import matplotlib.pyplot as plt
import os


print(os.getcwd())

image_path = "D:/Courses/Computer_vision_engineer/Text_detection/data/test2.png"

# read image
img = cv2.imread(image_path)

# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
text_ = reader.readtext(img)

# ideally the threshold should be over 0.5
# seting a threshold makes sure that there is no bbox drawn for other objects in the background
threshold = 0.25

# draw bbox on text
for txt in text_:
    print(txt)
    bbox, text, score = txt
    
    if score > threshold:
        # bbox[0] = top left corner; bbox[2] = bottom right corner
        # (0, 255, 0) = green color; 2 = thickness
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

# covert image from BGR to RGB
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

