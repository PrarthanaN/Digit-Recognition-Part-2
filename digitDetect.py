import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps

X, y  = fetch_openml("mnist_784", version = 1, return_X_y = True)
X = np.array(X)
print(pd.Series(y).value_counts())
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nClasses = len(classes)

XTrain, XTest, YTrain, YTest = train_test_split(X, y, random_state = 0, train_size = 7500, test_size = 2500)
XTrainScale = XTrain / 255.0
XTestScale = XTest / 255.0

lr = LogisticRegression(solver = "saga", multi_class = "multinomial")
lr.fit(XTrainScale, YTrain)

yPredict = lr.predict(XTestScale)
accuracy = accuracy_score(YTest, yPredict)
print(accuracy)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperLeft = (int(width / 2 - 56), int(height / 2 - 56))
        bottomRight = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upperLeft, bottomRight, (255, 0, 0), 2)
        # Region of interest (Focal Point)
        roi = gray[upperLeft[1]: bottomRight[1], upperLeft[0]: bottomRight[0]]
        #Converting the CV2 image to PIL format
        PILFormat = Image.fromarray(roi)
        ImageBW = PILFormat.convert("L")
        ImageResized = ImageBW.resize((28, 28), Image.ANTIALIAS)
        pixelFilter = 20
        ImageResizeInverted = PIL.ImageOps.invert(ImageResized)
        minPixel = np.percentile(ImageResizeInverted, pixelFilter)
        ImageScaled = np.click(ImageResizeInverted-minPixel, 0, 255)
        maxPixel = np.max(ImageResizeInverted)
        ImageScaled = np.asarray(ImageScaled) / maxPixel
        testSample = np.array(ImageScaled).reshape(1, 784)

        testPredict = lr.predict(testSample)
        print("The predicted class is: ", testPredict)
        cv2.imshow("frame", gray)
        if cv2.waitKey(1):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
