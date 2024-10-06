
train_set_dir = "../data/clothing/images/train_set"
test_set_dir = "../data/clothing/images/test_set"

#train_set_dir = "../data/mini_imagenet100/images/train_set"
#test_set_dir = "../data/mini_imagenet100/images/test_set"


image_extension = ".jpg"

image_classes_file = "output/image_classes.txt"
num_classes = 14


image_resize_width = 224
image_resize_height = 224

batch_size = 64

transform_normalization_file = "output/image_transform_normalization.txt"
transform_normalization_mean = [0.5894, 1.3996, 1.7191]
transform_normalization_std = [0.2528, 0.2523, 0.2520]



