
import torch
import cv2

trained_model = r'C:\Users\Murat\Desktop\yolo_flask\best.pt'

#my_model2 = Licence_Plate(trained_model)


class Licence_Plate:

      def __init__(self,trained_model):

            self.model = torch.hub.load('ultralytics/yolov5', 'custom', trained_model)

      def detect(self,img):

            self.result_model = self.model(img)
            self.img = img
            img = self.plot()

            return img

      def plot(self):

            det = self.result_model.xyxy[0]

            for i in range(len(det)):

                pt1 = (int(det[i][0]), int(det[i][1]))
                pt2 = (int(det[i][2]), int(det[i][3]))
                img = cv2.rectangle(self.img,pt1,pt2,color=(0,255,0),thickness=2)

            return img

my_model = Licence_Plate(trained_model)