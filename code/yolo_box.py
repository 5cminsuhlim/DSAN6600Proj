import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import logging
from multiprocessing import Pool, set_start_method, cpu_count
import torch

# load yolo
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

cwd = os.getcwd()
parent = os.path.dirname(cwd)
print(parent)
print(cwd)

#set parent as current working directory
cwd = parent
print(cwd)

project_path = cwd
raw_imgs_path = os.path.join(project_path, 'data/raw')
test_imgs_path = os.path.join(project_path, 'data/test-images')
all_imgs = glob.glob(os.path.join(raw_imgs_path, '*.jpg'))
test_imgs = glob.glob(os.path.join(test_imgs_path, '*.jpg'))

### TESTING ON SUBSET ###
all_imgs = all_imgs[:200]
batch_size = int(len(all_imgs) * 0.07)

print(raw_imgs_path)
print(all_imgs[:3])

def process_images(model, img_paths):
    imgs = [Image.open(img_path).convert('RGB') for img_path in img_paths]
    results = model(imgs)
    dets = results.pandas().xyxy
    people_det = [det[(det['class'] == 0) & (det['confidence'] >= 0.7)] for det in dets]
    return people_det
def worker(img_paths):
    return process_images(model, img_paths)

print(os.path.join(project_path, 'data/boundingbox'))

def crop_and_save(img_paths, detections_list, verbose=False):
  out_path = os.path.join(project_path, 'data/boundingbox')
  os.makedirs(out_path, exist_ok=True)

  for img_path, detections in zip(img_paths, detections_list):
    if len(detections)<2:
      continue
    # Load img
    img = Image.open(img_path)
    # find the bounding box that contains all objects
    xmin = int(detections['xmin'].min())
    ymin = int(detections['ymin'].min())
    xmax = int(detections['xmax'].max())
    ymax = int(detections['ymax'].max())    
    
    #find new area and see if it is sufficiently smaller than original area
    size=img.size[0]*img.size[1]
    cropped_size=(xmax-xmin)*(ymax-ymin)

    #find new area and see if it is sufficiently smaller than original area
    size=img.size[0]*img.size[1]
    cropped_size=(xmax-xmin)*(ymax-ymin)

    if cropped_size < .75*size:
      #crop image around bounding box
      cropped_img = img.crop((xmin, ymin, xmax, ymax))

      # get image name; remove extension
      img_name = os.path.basename(img_path)
      img_name_no_ext = os.path.splitext(img_name)[0]

      # create new image name w/ bounding box coords
      new_img_name = f"{img_name_no_ext}_subject_box_{xmin}_{ymin}_{xmax}_{ymax}.jpg"

      # save cropped img
      save_path = os.path.join(out_path, new_img_name)
      cropped_img.save(save_path)

      if verbose:
        print(f"Saved: {save_path}")

def main():
    # Ensure the start method is 'spawn'
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    torch.cuda.empty_cache()

    # Logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s - %(levelname)s')

    # Assuming test_imgs is defined and worker, crop_and_save functions are implemented properly
    batch_size = 50
    batches = [all_imgs[i:i + batch_size] for i in range(0, len(all_imgs), batch_size)]
    num_processes = 2

    with Pool(processes=num_processes) as pool:
        for result, batch_paths in zip(pool.imap_unordered(worker, batches), batches):
            if not result:
                logging.warning("No detections in this batch: %s", batch_paths)
                print("No detections in this batch:", batch_paths)
                continue
            crop_and_save(batch_paths, result, verbose=False)
            logging.info("Processed batch: %s", batch_paths)

if __name__ == "__main__":
   main()