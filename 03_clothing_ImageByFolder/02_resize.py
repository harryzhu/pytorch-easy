import os
import glob
import shutil
import random

from PIL import Image

# import torch
# import torchvision
# from torchvision import transforms

image_src = '../data/clothing/image_src'
image_dst = '../data/clothing/image_dst'
image_small = '../data/clothing/images_small'

image_train = '../data/clothing/images/train_set'
image_test = '../data/clothing/images/test_set'

def resize_image():
  all_src_images = glob.glob(image_src + "/**/*.jpg")

  for img in all_src_images:
    save_path = img.replace(image_src,image_dst)
    small_path = img.replace(image_src,image_dst)

    if not os.path.exists(os.path.dirname(save_path)):
      os.makedirs(os.path.dirname(save_path))


    img_data = Image.open(img)
    img_width , img_height = img_data.size
    if img_width < 350 or img_height < 450:
      if not os.path.exists(os.path.dirname(small_path)):
        os.makedirs(os.path.dirname(small_path))
        shutil.copy(img,small_path)
    else:
      img_data = img_data.resize((400,500))
      img_data.save(save_path)



def split_train_test():
  all_dirs = glob.glob(image_dst + "/*")
  for folder in all_dirs:
    imgs = glob.glob(folder + "/*.jpg")
    imgs_count = len(imgs)
    if imgs_count < 50:
      #print('ignore:',folder)
      continue
    imgs_test_p10 = int(imgs_count / 10) + 1
    #print(imgs_test_p10)
    random.shuffle(imgs)


    for idx, img in enumerate(imgs):
      test_img = img.replace(image_dst, image_test)
      train_img = img.replace(image_dst, image_train)

      if not os.path.exists(os.path.dirname(test_img)):
          os.makedirs(os.path.dirname(test_img))

      if not os.path.exists(os.path.dirname(train_img)):
          os.makedirs(os.path.dirname(train_img))

      if idx <= imgs_test_p10:
          shutil.move(img, test_img)
          print(test_img)
      else:
          shutil.move(img, train_img)
          print(train_img)

    



resize_image()

split_train_test()
