import cv2
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error

def totalError(test_path):
    print("Testing the accuracy of the trained model using mse as metric............")
    with open(test_path + '/labels.txt', "r") as f_o:
        lines = f_o.readlines()
        center_pred = []
        center_actual = []
        for line in lines:
            vals = line.split(' ')
            center_actual.append([float(vals[1]), float(vals[2])])

            center_x_pred, center_y_pred = (testYOLOv3(test_path + '/' + vals[0]))
            center_pred.append([center_x_pred, center_y_pred])

    # mae = mean_absolute_error(center_pred, center_actual)
    mse = mean_squared_error(center_pred, center_actual)
    return mse       

def testYOLOv3(img_path):
    cwd = os.getcwd()

    # Load Yolo
    weight_path = cwd + '/testing_cfg_weights/yolov3_training_last.weights'
    test_cfg_path = cwd + '/testing_cfg_weights/yolov3_testing.cfg'
    net = cv2.dnn.readNet(weight_path, test_cfg_path)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Loading image
    img = cv2.imread(img_path)
    height, width, channel = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = (255,0,0)

            label_center_x = str(round(center_x/width, 2))
            label_center_y = str(round(center_y/height, 2))
            
            cv2.circle(img, (center_x, center_y), 2, (0,0,255), -1)

            # Draw center and show center points
            cv2.putText(img, '(', (center_x-10, center_y + 30), font, 1, color, 1)
            cv2.putText(img, label_center_x, (center_x, center_y + 30), font, 1, color, 1)
            cv2.putText(img, ',', (center_x+35, center_y + 30), font, 1, color, 1)
            cv2.putText(img, label_center_y, (center_x+40, center_y + 30), font, 1, color, 1)
            cv2.putText(img, ')', (center_x+80, center_y + 30), font, 1, color, 1)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1000)

    cv2.destroyAllWindows()
    return (round(center_x/width, 4), round(center_y/height, 4))

def main():
    centre_x, centre_y = testYOLOv3(sys.argv[1])
    print(centre_x, centre_y)
    mse = totalError(sys.argv[2])
    print("The mean squared error over test images is ", mse)

if __name__ == "__main__":
    main()