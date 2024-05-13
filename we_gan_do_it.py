import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

# image_06734 to image_06773
# class_00001


num_imgs = 39
imgs_dir = "C:/Users/farrj/Documents/Scripts/COMP4531/gan_stuff/stack_gan/data/sample_jpg/"
captions_dir = "C:/Users/farrj/Documents/Scripts/COMP4531/gan_stuff/stack_gan/data/text_c10/text_c10/class_00001/image_06734.txt"

images = []
img_names_list = []
# get image names
for file in os.listdir(imgs_dir):
    img_names_list.append(file)
    #print(img_names_list)
#print(img_names_list)

for name in img_names_list:
    #print(os.path.join(imgs_dir, name))
    img_raw = cv2.imread(os.path.join(imgs_dir, name))
    # cv2.imshow('image', img_raw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #images.append(img_raw)
    img = cv2.resize(img_raw, (256, 256))
    #img = img.astype(np.float32)       # when resizing with cv2.resize, if using this to change type, it messes up the image

    ## STILL NEED TO CONVER TO FLOAT
    images.append(img)



# for name in img_names_list:
#     max_dim = 256
#     img = tf.io.read_file(os.path.join(imgs_dir, name))
#     img = tf.image.decode_image(img, channels= 3)
#     img = tf.image.convert_image_dtype(img, tf.float32)

#     shape = tf.cast(tf.shape(img)[:-1], tf.float32)
#     long_dim = max(shape)
#     scale = max_dim / long_dim

#     new_shape = tf.cast(shape * scale, tf.int32)

#     img = tf.image.resize(img, new_shape)
#     img = img[tf.newaxis, :]



# used this section to check images before reshaping
#print(images[0])
#print(images[0].shape)
#cv2.imshow('image_06734', images[0])

#used this section to check images after reshaping
print(images[0].shape)
cv2.imshow('image_06734', images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

print(type(images))
# list right now
img_data = tf.convert_to_tensor(images, dtype= float)
print("after", type(img_data))



### load in text data
captions = tf.data.TextLineDataset(captions_dir)
for text in captions.take(5):
    print("Sentence: ", text.numpy())
## can load labels in at this step as well
## will need to step through all the directories, currently only using one text file description (for first image in class 00001)


# create vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens= 5000,         # since output_mode="int", effective max_tokens=5000-2
    standardize= 'lower_and_strip_punctuation',
    output_mode= 'int',
    output_sequence_length= 20
)
# create vocabulary
vectorize_layer.adapt(captions)
print(vectorize_layer.get_vocabulary())




### set up generator
                            # not need???
### set up discriminator

### set up model
model = tf.keras.models.Sequential([
    # feed in images and text
    tf.keras.layers.Dense(20, activation='relu'),       # need to change num #
    tf.keras.layers.Dense(10)
])

#inputs = tf.keras.layers.Input(shape- (1, ), dtype= tf.string, name= "text")

dataset = dataset.map(lambda x, y: (preprocessing_layer(x), y))
dataset = dataset(prefetch(tf.data.AUTOTUNE))
model.fit(dataset, ...)