import os
import csv
import shutil

csv_file = "clothing-dataset/images.csv"

with open(csv_file, 'r') as f:
  reader = csv.reader(f)
  for row in reader:
    if row[0] == "image":
      continue
    if row[2].lower() == "not sure" or row[3].lower() == "true":
      continue
    img_path = "clothing-dataset/images/" + row[0] + '.jpg'
    class_name = row[2]
    cp_path = '../data/clothing/image_src/' + class_name
    if not os.path.exists(cp_path):
      os.makedirs(cp_path)
    if os.path.exists(img_path):
      shutil.copy(img_path,cp_path)
