import numpy as np
import tensorflow as tf
import cv2 as cv2
from PIL.ImageFont import ImageFont
from object_detection.utils import label_map_util
import PIL.ImageFont as ImageFont
from PIL import ImageDraw

from object_detection.utils import visualization_utils as vis_util


def create_category_index(label_path='labelmap.txt'):
    f = open(label_path)
    category_index = {}
    for i, val in enumerate(f):
        if i != 0:
            val = val[:-1]
            if val != '???':
                category_index.update({(i - 1): {'id': (i - 1), 'name': val}})

    f.close()
    return category_index

def draw_bounding_box_on_image(image,ymin,xmin,ymax,xmax ):
  """Adds a bounding box to an image."""
  print(image.shape)
  im_width, im_height,im_depth = image.shape
  left, right, top, bottom = int(xmin * im_width), \
                             int(xmax * im_height),\
                             int(ymin * im_width), \
                             int(ymax * im_height)

  print("========>",left, right, top, bottom)
  image =cv2.rectangle(image, (left, top), (right, bottom), (255,0,0), 2)

  return image

def process_live_video(interpreter):
    classes_found = []
    category_index = create_category_index()
    print(category_index)
    cap = cv2.VideoCapture(0)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    threshold = 0.6
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = cv2.rectangle(frame,(100,0),(500,0),(0,0,255),5)
        if ret:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (300, 300))
            img_rgb = img_rgb.reshape([ 1, 300, 300,3])
            interpreter.set_tensor(input_details[0]['index'], img_rgb)
            interpreter.invoke()


            output_dict = {
                'detection_boxes': interpreter.get_tensor(output_details[0]['index'])[0],
                'detection_classes': interpreter.get_tensor(output_details[1]['index'])[0],
                'detection_scores': interpreter.get_tensor(output_details[2]['index'])[0],
                'num_detections': interpreter.get_tensor(output_details[3]['index'])[0]
            }

            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                use_normalized_coordinates=False,
                min_score_thresh=threshold,
                line_thickness=10,
                agnostic_mode=False)

            boxes = output_dict['detection_boxes']
            # get all boxes from an array
            max_boxes_to_draw = boxes.shape[0]
            # get scores to get a threshold
            scores = output_dict['detection_scores']
            # this is set as a default but feel free to adjust it to your needs
            min_score_thresh = threshold
            # iterate over all objects found
            for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                if scores is None or scores[i] > min_score_thresh:
                    # boxes[i] is the box which will be drawn
                    index= output_dict['detection_classes'][i]
                    class_name = category_index[output_dict['detection_classes'][i]]['name']
                    print("This box is gonna get used", boxes[i], scores[i] *100, class_name)
                    classes_found.append(class_name)
                    frame = draw_bounding_box_on_image(frame,boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3])


            cv2.imshow('object_detection', cv2.resize(frame, (320, 320)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

interpreter = tf.lite.Interpreter(model_path="detect.tflite")

process_live_video(interpreter)