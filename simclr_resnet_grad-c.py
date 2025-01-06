import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from sklearn.decomposition import NMF

"""
https://github.com/fawazsammani/explain-cl/blob/main/methods.py
"""

class GradCAM:
    def __init__(self, ssl_model):
        self.ssl_model = ssl_model
        self.gradients = {}
        self.features = {}


    def save_grads(self, tape, img_index, target_tensor):
        self.gradients[img_index] = tape.gradient(target_tensor, self.features[img_index])

    def save_features(self, img_index, feats):
        self.features[img_index] = feats

    def __call__(self, img1, img2):
        with tf.GradientTape() as tape:
            tape.watch(img1)
            tape.watch(img2)

            features1 = self.ssl_model.encoder.net(img1, training=False)
            features2 = self.ssl_model.encoder.net(img2, training=False)

            self.save_features('1', features1)
            self.save_features('2', features2)

            out1 = tf.reduce_mean(features1, axis=[1, 2])
            out2 = tf.reduce_mean(features2, axis=[1, 2])
            out1 = self.ssl_model.contrastive_head(out1)
            out2 = self.ssl_model.contrastive_head(out2)

            score = tf.reduce_mean(tf.keras.losses.cosine_similarity(out1, out2))

        self.save_grads(tape, '1', features1)
        self.save_grads(tape, '2', features2)

        return score


def weight_activation(feats, grads):
    relu_grads = tf.nn.relu(grads)
    cam = feats * relu_grads
    cam = tf.reduce_sum(cam, axis=-1).numpy()
    return cam


def get_gradcam(ssl_model, img1, img2):
    grad_cam = GradCAM(ssl_model)
    score = grad_cam(img1, img2)

    cam1 = weight_activation(grad_cam.features['1'], grad_cam.gradients['1'])
    cam2 = weight_activation(grad_cam.features['2'], grad_cam.gradients['2'])
    return cam1, cam2


def get_interactioncam(ssl_model, img1, img2, reduction, grad_interact=False):
    grad_cam = GradCAM(ssl_model)
    score = grad_cam(img1, img2)

    G1 = grad_cam.gradients['1']
    G2 = grad_cam.gradients['2']

    if grad_interact:
        B, H, W, D = G1.shape
        G1_ = tf.reshape(G1, [B, H * W, D])
        G2_ = tf.reshape(G2, [B, H * W, D])

        G_ = tf.matmul(tf.transpose(G1_, perm=[0, 2, 1]), G2_)
        G1 = tf.reduce_max(G_, axis=-1, keepdims=True)
        G2 = tf.reduce_max(G_, axis=1, keepdims=True)

    if reduction == 'mean':
        joint_weight = tf.reduce_mean(grad_cam.features['1'], axis=[1, 2]) * tf.reduce_mean(grad_cam.features['2'], axis=[1, 2])
    elif reduction == 'max':
        max_pooled1 = tf.reduce_max(grad_cam.features['1'], axis=[1, 2])
        max_pooled2 = tf.reduce_max(grad_cam.features['2'], axis=[1, 2])
        joint_weight = max_pooled1 * max_pooled2
    else:
        reshaped1 = tf.reshape(grad_cam.features['1'], [B, H * W, D])
        reshaped2 = tf.reshape(grad_cam.features['2'], [B, H * W, D])

        features1_query = tf.expand_dims(tf.reduce_mean(reshaped1, axis=1), axis=1)
        features2_query = tf.expand_dims(tf.reduce_mean(reshaped2, axis=1), axis=1)

        attn1 = tf.nn.softmax(tf.matmul(features1_query, tf.transpose(reshaped1, perm=[0, 2, 1])), axis=-1)
        attn2 = tf.nn.softmax(tf.matmul(features2_query, tf.transpose(reshaped2, perm=[0, 2, 1])), axis=-1)

        att_reduced1 = tf.squeeze(tf.matmul(attn1, reshaped1), axis=1)
        att_reduced2 = tf.squeeze(tf.matmul(attn2, reshaped2), axis=1)

        joint_weight = att_reduced1 * att_reduced2

    joint_weight = tf.expand_dims(tf.expand_dims(joint_weight, axis=1), axis=1)
    feats1 = grad_cam.features['1'] * joint_weight
    feats2 = grad_cam.features['2'] * joint_weight

    cam1 = weight_activation(feats1, G1)
    cam2 = weight_activation(feats2, G2)

    return cam1, cam2


import tensorflow as tf
import numpy as np
from PIL import Image

# Load and preprocess the image
img_path = 'images/dog.jpeg'
img = Image.open(img_path).convert('RGB')

# Define transformations
def pure_transform(image):
    # Normalize the image to [0, 1]
    return image / 255.0

def aug_transform(image):
    # Apply augmentation: random flip and random brightness
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)  # Adjust brightness randomly
    return image / 255.0

# Process the image based on the flag
augment_first_img = False

# Convert PIL image to a TensorFlow tensor
img_tensor = tf.convert_to_tensor(np.array(img), dtype=tf.float32)

# Apply transformations
if augment_first_img:
    img1 = tf.expand_dims(aug_transform(img_tensor), axis=0)
else:
    img1 = tf.expand_dims(pure_transform(img_tensor), axis=0)

img2 = tf.expand_dims(aug_transform(img_tensor), axis=0)

# Example function for adding normalization
def modify_transforms(normal_transforms, no_shift_transforms, ig_transforms):
    def add_normalization_to_transform(transform):
        def normalized_transform(image):
            image = transform(image)
            return tf.image.per_image_standardization(image)  # Normalize to mean 0, stddev 1
        return normalized_transform

    normal_transforms['pure'] = add_normalization_to_transform(normal_transforms['pure'])
    normal_transforms['aug'] = add_normalization_to_transform(normal_transforms['aug'])
    no_shift_transforms = {k: add_normalization_to_transform(v) for k, v in no_shift_transforms.items()}
    ig_transforms = {k: add_normalization_to_transform(v) for k, v in ig_transforms.items()}

    return normal_transforms, no_shift_transforms, ig_transforms

# Define the normal transforms dictionary
normal_transforms = {'pure': pure_transform, 'aug': aug_transform}
