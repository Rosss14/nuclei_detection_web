import tensorflow as tf

import numpy as np

from PIL import Image

from six import BytesIO

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

model = tf.saved_model.load('saved_model')
category_index = label_map_util.create_category_index_from_labelmap('label_map.pbtxt', use_display_name=True)

def load_image_into_numpy_array(path):
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  #if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
  #  detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
  #            output_dict['detection_masks'], output_dict['detection_boxes'],
  #             image.shape[0], image.shape[1])      
  #  detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
  #                                     tf.uint8)
  #  output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def detect_and_save(model, image, category_index):
  image_np = load_image_into_numpy_array(image)

  output_dict = run_inference_for_single_image(model, image_np)

  image_np_det = np.copy(image_np)

  vis_util.visualize_boxes_and_labels_on_image_array(image_np_det,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    min_score_thresh=0.36,
    line_thickness=1)
  
  image_det_path = image[:-4] + '_detection.png'

  image_det = Image.fromarray(image_np_det)
  image_det.save(image_det_path)

  return image_det_path