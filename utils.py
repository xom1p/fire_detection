import openvino as ov
import cv2
import numpy as np
import glob
from ultralytics.utils.plotting import colors

model_path = './models/best.xml' #/mount/src/ai_fire_safety_project

core = ov.Core()

model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

label_map = ['fire', 'smoke']

def prepare_data(image, input_layer):

    input_w, input_h = input_layer.shape[2], input_layer.shape[3]
    input_image = cv2.resize(image, (input_w,input_h))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image/255

    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, 0)

    return input_image

infer_times_OV = []

for i, image_path in enumerate(glob.glob(f'Pothole-detection-using-YOLOv5-1/valid/images/*.jpg')):

    image = cv2.imread(image_path)
    input_image =prepare_data(image, input_layer)

    #---OpenVino---
    infer_start = time.time()
    output = compiled_model([input_image])[output_layer]
    inference_time = time.time() - infer_start
    infer_times_OV.append(inference_time)
    #------
    display('Image: ' +  str(i))


def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] == scores.shape[0]

    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    areas = (ys2 - ys1) * (xs2 - xs1)

    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []

    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)

        if not len(scores_indexes):
            break

        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])

        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)

def compute_iou(box, boxes, box_area, boxes_area):

    assert boxes.shape[0] == boxes_area.shape[0]

    ys1 = np.maximum(box[0], boxes[:, 0])
    xs1 = np.maximum(box[1], boxes[:, 1])
    ys2 = np.minimum(box[2], boxes[:, 2])
    xs2 = np.minimum(box[3], boxes[:, 3])

    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    unions = box_area + boxes_area - intersections
    ious = intersections / unions
    return ious

def evaluate(predictions, label_map, conf = .3):

    boxes = []
    scores = []
    labels = []

    for i, preds in enumerate(predictions[4:]):


        detected_objects = np.argwhere(preds>conf)


        if len(detected_objects):

            for index in detected_objects:

                    score = predictions[4][index][0]

                    xcen = predictions[0][index][0]
                    ycen = predictions[1][index][0]
                    w = predictions[2][index][0]
                    h = predictions[3][index][0]


                    xmin = xcen - (w/2)
                    xmax = xcen + (w/2)
                    ymin = ycen - (h/2)
                    ymax = ycen + (h/2)
                    box = (xmin, ymin, xmax, ymax)

                    boxes.append(box)
                    scores.append(score)
                    labels.append(i)

    return np.array(boxes), np.array(scores), np.array(labels)

def visualize(nms_output, boxes, orig_image, label_names,scores, input_layer ):
    orig_h, orig_w, c = orig_image.shape
    color = (0,0,0)
    for i in nms_output:
        xmin, ymin, xmax, ymax = boxes[i]

        xmin = int(xmin*orig_w/input_layer.shape[2])
        ymin = int(ymin*orig_h/input_layer.shape[3])
        xmax = int(xmax*orig_w/input_layer.shape[2])
        ymax = int(ymax*orig_h/input_layer.shape[3])

        color = colors(label_names[i])
        cv2.rectangle(orig_image, (xmin,ymin), (xmax,ymax), color, 4)

        text = str(int(np.rint(scores[i]*100))) + "% " + label_map[label_names[i]]
        cv2.putText(orig_image, text, (xmin+2,ymin-5), cv2.FONT_HERSHEY_SIMPLEX,
                   .75, color, 2, cv2.LINE_AA)

    return orig_image

def predict_image(img, conf_threshold):

    # ----- OpenVino ----- #
    OV_image = img.copy()
    input_image =prepare_data(OV_image, input_layer)
    output = compiled_model([input_image])[output_layer]
    boxes, scores, label_names = evaluate(output[0],label_map, conf_threshold)

    if len(boxes):
        nms_output = non_max_suppression(boxes, scores, conf_threshold)
        visualize(nms_output, boxes, OV_image, label_names, scores, input_layer)

    return OV_image
