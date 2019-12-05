import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util
# this is the script that created by TensorFlow authors
from utils import visualization_utils as vis_util


def stream_detection(model_path="ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb",
                     label_path="ssd_mobilenet_v1_coco_2018_01_28/mscoco_label_map.pbtxt"):
    """
    this is the function that will detect the object from real time video
    :param model_path: the path to the model
    :param label_path: the path to the label file path
    :return: no return
    """
    capture = cv2.VideoCapture(0)
    detection = tf.Graph()
    # create TensorFlow instance
    with detection.as_default():
        graph_def_1 = tf.compat.v1.GraphDef()
        # load the model
        with tf.compat.v2.io.gfile.GFile(model_path, 'rb') as fid:
            graph_def_1.ParseFromString(fid.read())
            tf.import_graph_def(graph_def_1, name='')
    # create the categories index from the label text
    label_map = label_map_util.load_label(label_path)
    cat = label_map_util.convert_to_categories(label_map)
    categories_index = label_map_util.create_index(cat)
    # detection part
    with detection.as_default():
        # create a TensorFlow session
        with tf.compat.v1.Session(graph=detection) as sess:
            image_tensor = detection.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection.get_tensor_by_name('detection_scores:0')
            detection_classes = detection.get_tensor_by_name('detection_classes:0')
            num_detections = detection.get_tensor_by_name('num_detections:0')
            # read in the video from the web cam
            while capture.isOpened():
                # get frame from the video
                ret, image_np = capture.read()
                # origin image matrix
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                # expend to image tensor
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # detect the tensor once a time
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # use the result to draw box om the frame image
                vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                                   categories_index, use_normalized_coordinates=True,
                                                                   line_thickness=2)
                # show the video stream
                cv2.imshow("capture", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_detection()