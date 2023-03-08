import cv2 as cv
import torch
import numpy as np

from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms


# class FaceDetector(object):
# object declares the class as a new-style class, which inherit from a built-in base class
# https://www.w3docs.com/snippets/python/what-is-the-difference-between-old-style-and-new-style-classes-in-python.html#:~:text=In%20Python%2C%20classes%20can%20be,old%2Dstyle%20classes%20do%20not.
# All classes are new-style classes by default in 3.x versions, so its not required
class FaceDetector:
    def __init__(self, mtcnn, classifier):
        self.mtcnn = mtcnn
        self.classifier = classifier

    def _draw(self, frame, boxes, probs, landmarks):
        for box, prob, ldm in zip(boxes, probs, landmarks):
            # need to set everything as int
            box = box.astype("int")
            ldm = ldm.astype("int")

            cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            cv.putText(
                frame,
                str(prob),
                (box[2], box[3]),
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 0, 255),
                1,
                cv.LINE_AA,
                False,
            )  # placed in the bottom left?

            # Draw landmarks
            cv.circle(frame, tuple(ldm[0]), 5, (0, 0, 255), -1)
            cv.circle(frame, tuple(ldm[1]), 5, (0, 0, 255), -1)
            cv.circle(frame, tuple(ldm[2]), 5, (0, 0, 255), -1)
            cv.circle(frame, tuple(ldm[3]), 5, (0, 0, 255), -1)
            cv.circle(frame, tuple(ldm[4]), 5, (0, 0, 255), -1)
        return frame

    def detect_ROI(self, boxes):
        ROIs = []  # regions of interest
        # box -> (X1, X2, Y1, Y2)
        for box in boxes:
            ROI = [
                int(box[1]),
                int(box[3]),
                int(box[0]),
                int(box[2]),
            ]  # first 2 elements are top left, last 2 bottom right
            ROIs.append(ROI)
        return ROIs

    def _blur_face(self, img, factor=3.0):
        # size of blurring kernel based on input image(?)
        (h, w) = img.shape[:2]
        kW = int(w / factor)
        kH = int(h / factor)

        # Why does it need to be odd?
        # if kW % 2 == 0:
        #     kW -= 1

        # if kH % 2 == 0:
        #     kH -= 1
        return cv.GaussianBlur(img, (kW, kH), 0)

    def _is_it_me(self, face):
        # running classifier on face

        rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        PIL_img = Image.fromarray(rgb.astype("uint8"), "RGB")

        # need to augment it such that
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # in normalize fn, 1st array is the mean, 2nd is the std
            ]
        )
        # Using mean and std of imagenet is a common practice that helps with fine-tuning the performance of the pretrained model

        preprocessed_img = preprocess(PIL_img)

        # torch.unsqueeze creates a tensor from the processed image with a dimension size of one insertted at index 0
        batch_t = torch.unsqueeze(preprocessed_img, 0)
        with torch.no_grad():  # disable gradient calculation - reduce memory usage and speeds up computations
            out = self.classifier(batch_t)
            _, pred = torch.max(out, 1)

        prediction = np.array(pred[0])
        if prediction == 0:
            return True
        else:
            return False

    def run(self, blur=True):
        capture = cv.VideoCapture(0)  # use main camera as capture
        while True:
            ret, frame = capture.read()
            try:
                # https://www.kaggle.com/code/timesler/guide-to-mtcnn-in-facenet-pytorch#Bounding-boxes-and-facial-landmarks
                # For our use case, we look at facial landmarks, look into other use cases later

                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                self._draw(frame, boxes, probs, landmarks)
                # draw it. Leverage a trained mtcnn to automatically detect them in a frame, send it to the draw() fn which will highlight it for us

                if blur:
                    # get ROI
                    ROIs = self.detectROI(boxes)
                    for ROI in ROIs:
                        (startY, endY, startX, endX) = ROI
                        face = frame[startY:endY, startX:endX]

                        # run classifier on ROI
                        pred = self._is_it_me(face)
                        if pred:
                            blurred_face = self._blur_face(face)
                            frame[startY:endY, startX:endX] = blurred_face

            except Exception as e:
                print(str(e))
                pass

            cv.imshow("camera", frame)
            if cv.waitKey(20) & 0xFF == ord(
                "d"
            ):  # if 20 seconds, or 0xFF = order d -> if letter d is pressed, break out of loop
                break

        capture.release()
        cv.destroyAllWindows()


# mtcnn = MTCNN()
# fcd = FaceDetector(mtcnn)
# fcd.run()
