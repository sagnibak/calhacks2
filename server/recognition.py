import cv2
import numpy as np
import tensorflow as tf

global_graph = tf.Graph()

def load_graph(frozen_model):
    """Loads a frozen TensorFlow graph for inference."""
    with tf.gfile.GFile(frozen_model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with global_graph.as_default() as graph:
        tf.import_graph_def(graph_def, name='baby')
    
    return graph

print(global_graph is load_graph('ssd_mobilenet/frozen_inference_graph.pb'))  # True

global_session = tf.Session(graph=global_graph)

# ops = list(global_graph.get_operations())
# # ops.reverse()
# for idx, op in enumerate(ops):
#     if idx < 10:
#         print(op.name)

input_img = global_graph.get_tensor_by_name('baby/image_tensor:0')
detection_classes = global_graph.get_tensor_by_name('baby/detection_classes:0')
detection_boxes = global_graph.get_tensor_by_name('baby/detection_boxes:0')
detection_scores = global_graph.get_tensor_by_name('baby/detection_scores:0')
num_detections = global_graph.get_tensor_by_name('baby/num_detections:0')
print('input_img:', input_img)
# print('y:', y)

def predict(img):
    img = cv2.imread(img, 1)  # convert image to numpy array
    # img = cv2.resize(img, (300, 300))  # resize to 300 x 300
    img = np.expand_dims(img, axis=0)
    output = {}
    output['classes'] = global_session.run(detection_classes, feed_dict={input_img: img})
    output['boxes'] = global_session.run(detection_boxes, feed_dict={input_img: img})
    output['scores'] = global_session.run(detection_scores, feed_dict={input_img: img})
    output['num_detections'] = global_session.run(num_detections, feed_dict={input_img: img})

    return output

def categories(idx):  # 80 classes
    try:
        result= ['background',  # class zero
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'][idx]
    except IndexError as e:
        print('Failed to look up index', idx)
        result = None
    
    return result
