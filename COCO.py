import numpy as np
import tensorflow as tf
import cv2 as cv
from scipy.spatial import distance

from nms import non_max_suppression

nms= "False"
COCO_CLASSES_LIST = [
    'unlabeled',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

def edit_frame(img,x_center, y_center,x,y,right,bottom,COCO_CLASSES_LIST,clr,box_clr):
    cv.circle(img, (x_center, y_center), 1, (0, 255, 0), 1)
    if box_clr=="green":
        cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
    else:
        cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

    cv.putText(img, str(COCO_CLASSES_LIST + " " + clr), (int(x), int(y)), cv.FONT_HERSHEY_PLAIN, 1.0,
               (125, 255, 51), 1)
    return img

def find_clr(frame, x,y):
    b, g, r = (frame[x, y])
    if b > 240 and g > 240 and r > 240:
        clr = "white"
    elif (b < 65 and g < 65 and r < 65):
        clr = "black"
    else:
        clr = " "

    return clr


def process_frame(frame,car_count,detected_points,frame_cnt):
    # Read and preprocess an image.
    img = frame.copy()
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    if nms == "False":
        boxes = out[2][0]
        scores = out[1][0]
        classes = out[3][0]
    else:
        boxes, scores, classes = non_max_suppression(out[2][0], out[1][0], out[3][0], cols, rows)

    # Visualize detected bounding boxes.
    num_detections = len(classes)

    for i in range(num_detections):
        classId = int(classes[i])
        if classId is not None:
            score = float(scores[i])
            bbox = [float(v) for v in boxes[i]]
            if score > 0.4:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows

                x_center = int((x + right)/2)
                y_center = int((y + bottom)/2)

                # straight line
                cv.line(img,(100,250),(640,250),(0,0,255),1)
                clr = find_clr(frame,y_center, x_center)

                if y_center <= 250:
                    img = edit_frame(img,x_center, y_center,x,y,right,bottom,COCO_CLASSES_LIST[classId],clr,box_clr="green")
                else:
                    if y_center == 251:
                        if len(detected_points) != 0:

                            d = distance.euclidean((x_center,y_center), detected_points[0])

                            if d > 10 and (right < 630 and bottom < 350)\
                                    and frame_cnt - detected_points[1] > 10:
                                detected_points = []
                                detected_points.append((x_center, y_center))
                                detected_points.append(frame_cnt)
                                car_count += 1
                        else:
                            detected_points.append((x_center, y_center))
                            detected_points.append(frame_cnt)
                            car_count += 1


                    img = edit_frame(img,x_center, y_center,x,y,right,bottom,COCO_CLASSES_LIST[classId],clr,box_clr="red")

                cv.putText(img, str("car crossing red line: " + str(car_count)), (300, 25), cv.FONT_HERSHEY_PLAIN, 1.0,
                           (0, 0, 255), 1)

                cv.imwrite("out_img.jpg", img)

    return img,car_count, detected_points

def process_video():
    cap = cv.VideoCapture('slow.mp4')
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    ret, frame = cap.read()
    height, width, layers = frame.shape
    out = cv.VideoWriter("output.mp4", fourcc, 40, (width, height))
    car_count = 0
    frame_cnt = 1
    detected_points = []
    while (1):
        ret, frame = cap.read()
        print("frame#:", frame_cnt)

        if ret == True:
            # print(frame.shape)
            out_frame,car_count,detected_points = process_frame(frame,car_count,detected_points,frame_cnt)
            out.write(out_frame)
        else:
            break
        frame_cnt += 1
    print("done")
    out.release()
    cap.release()

# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    process_video()


    # cv.imwrite("out.jpg",img)
# process_video()