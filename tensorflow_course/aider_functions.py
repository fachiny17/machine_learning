### Aider functions for TensorFlow operations. these comprises of helpful functions and storing them for easy accessibility

import tensorflow as tf

# Create a function to import an image  and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into tensor and reshapes into (224, 224, 3)

    Args:
        filename (_type_): string filename of target image
        img_shape (int, optional): size to resize target image to. Defaults to 224.
        scale (bool, optional): whether to scale pixel values to range(0, 1). Defaults to True.
    """
    
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img/255.
    else:
        return img
    

# Create function to unzip a zipfile into current working directory 
# (since we're going to be downloading and unzipping a few files)
import zipfile

def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

    

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
    
import datetime
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


# aider function to crop faces from an image
def detect_and_crop_faces(image, detector):
    """
    Takes an image and YOLO detector.
    Returns a list of cropped face images.
    """
    results = detector(image)
    crops = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        face = image[y1:y2, x1:x2]
        
        # Skip empty crops
        if face.size > 0:
            crops.append(face)

    return crops

# Create output directory/folder if it doesn't exist
def create_folder(path):
    """Creates a folder if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

# Main preprocessing function
def process_dataset(input_dir, output_dir, detector):
    """
    Loops through each student's folder,
    detects faces in each image,
    and saves cropped faces.
    """
    for student_name in os.listdir(input_dir):
        student_input_folder = os.path.join(input_dir, student_name)
        student_output_folder = os.path.join(output_dir, student_name)

        create_folder(student_output_folder)

        print(f"[INFO] Processing student: {student_name}...")

        for img_file in os.listdir(student_input_folder):
            img_path = os.path.join(student_input_folder, img_file)
            image = load_image(img_path)

            if image is None:
                continue

            face_crops = detect_and_crop_faces(image, detector)

            # Save all detected face crops
            for i, face in enumerate(face_crops):
                save_path = os.path.join(
                    student_output_folder,
                    f"{os.path.splitext(img_file)[0]}_crop{i}.jpg"
                )
                cv2.imwrite(save_path, face)

        print(f"[DONE] Finished {student_name}.\n")


import cv2
# Function to load image for the face detection and cropping
def load_image(img_path):
    """Reads an image from disk and returns it in BGR format."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not load image at {img_path}")
    return img
