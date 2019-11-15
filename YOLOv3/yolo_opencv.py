"""

Topic:      Object detection plus localisation.

Authors:    Akhil Tarikere, 
            Rishikesh Vanarse, 
            Pranav Guruprasad and
            Sri Shreyas Sundaresan only.

Date:       (October, 2019)

"""


# Imports.
import cv2
import argparse
import numpy as np


# Defining the constants.
thresholdDistance = 500
confidenceThreshold = 0.4


# Taking two images and the other inputs.
ap = argparse.ArgumentParser()
ap.add_argument('-i1', '--image1', required=True,
                help = 'path to input image1')
ap.add_argument('-i2', '--image2', required=True,
                help = 'path to input image2')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



# Getting the images and storing them in the array.
image1 = cv2.imread(args.image1)
image2 = cv2.imread(args.image2)
image = [image1, image2]

listClasses = [[],[]]


for i in range(len(image)):
    # Finding the width and height for the images.
    Width = image[i].shape[1]
    Height = image[i].shape[0]
    # print(Width, Height)
    scale = 0.00392

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image[i], scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidenceThreshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                print(center_x, center_y, class_id)
                listClasses[i].append((center_x, center_y, class_id))


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for j in indices:
        j = j[0]
        box = boxes[j]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image[i], class_ids[j], confidences[j], round(x), round(y), round(x+w), round(y+h))

    # cv2.imshow("object detection", image[i]) # To show the image with the classes.
    cv2.waitKey()
        
    cv2.imwrite("object-detection.jpg", image[i])
    cv2.destroyAllWindows()
    print('..')


print(listClasses)


# To compare between the various classes in the two photos.
# listClasses[0] = Photo-1, listClasses[1] = Photo-2.
for sets1 in listClasses[0]:
    for sets2 in listClasses[1]:
        if sets1[2] == sets2[2]:
            dist = np.sqrt((sets1[0]-sets2[0])**2 + (sets1[1]-sets2[1])**2)
            if dist <= thresholdDistance:
                # Get kinect z-axis to find the vector difference.
                # Append the co-ordinates to a new list.
                # Find the new position with respect to that object.
                # Find the average position with respect all the objects.
                print("Yaay2!", sets1, sets2)
