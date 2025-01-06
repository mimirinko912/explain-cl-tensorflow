
import os

from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS


flags.DEFINE_float(
    'learning_rate', 0.3,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_enum(
    'learning_rate_scaling', 'linear', ['linear', 'sqrt'],
    'How to scale the learning rate as a function of batch size.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'train_batch_size', 512,
    'Batch size for training.')

flags.DEFINE_string(
    'train_split', 'train',
    'Split for training.')

flags.DEFINE_integer(
    'train_epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'train_steps', 0,
    'Number of steps to train for. If provided, overrides train_epochs.')

flags.DEFINE_integer(
    'eval_steps', 0,
    'Number of steps to eval for. If not provided, evals over entire dataset.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 0,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_string(
    'eval_split', 'validation',
    'Split for evaluation.')

flags.DEFINE_string(
    'dataset', 'imagenet2012',
    'Name of a dataset.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_bool('lineareval_while_pretraining', False,
                  'Whether to finetune supervised head while pretraining.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')

flags.DEFINE_string(
    'master', None,
    'Address/name of the TensorFlow master to use. By default, use an '
    'in-process master.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

flags.DEFINE_bool(
    'use_tpu', True,
    'Whether to run on TPU.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_enum(
    'optimizer', 'lars', ['momentum', 'adam', 'lars'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_string(
    'eval_name', None,
    'Name for eval.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 3,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', 0,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')

flags.DEFINE_boolean(
    'global_bn', True,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

flags.DEFINE_integer(
    'resnet_depth', 50,
    'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float(
    'se_ratio', 0.,
    'If it is bigger than 0, it will enable SE.')

flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')

flags.DEFINE_boolean(
    'use_blur', True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.')

flags.DEFINE_bool(
  'generate_gradcam', False, 'Whether to generate Grad-CAM heatmaps or not')



import tensorflow as tf
import numpy as np
import model
import numpy as np
from PIL import Image
import random
import cv2
import io

# Define transformations
def pure_transform(image):
    # Normalize the image to [0, 1]
    return image / 255.0

def aug_transform(image):
    # Apply augmentation: random flip and random brightness
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)  # Adjust brightness randomly
    return image / 255.0


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



def overlay_heatmap(img, heatmap, denormalize = False):
    loaded_img = img.squeeze(0).cpu().numpy().transpose((1, 2, 0))

    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        loaded_img = std * loaded_img + mean

    loaded_img = (loaded_img.clip(0, 1) * 255).astype(np.uint8)
    cam = heatmap / heatmap.max()
    cam = cv2.resize(cam, (224, 224))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)   # jet: blue --> red
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    added_image = cv2.addWeighted(cam, 0.5, loaded_img, 0.5, 0)
    return added_image


def show_image(x, squeeze = True, denormalize = False):

    if squeeze:
        x = x.squeeze(0)

    x = x.cpu().numpy().transpose((1, 2, 0))

    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = std * x + mean

    return x.clip(0, 1)


class GradCAM(tf.keras.models.Model):

    def __init__(self, md):
        super(GradCAM, self).__init__()

        self.gradients = {}
        self.features = {}

        self.feature_extractor = md.resnet_model  # Use resnet model as feature extractor
        self.contrastive_head = md._projection_head  # Use projection head for contrastive learning
        self.measure = tf.keras.losses.CosineSimilarity()

    def save_grads(self, img_index):
        def hook(grad):
            self.gradients[img_index] = grad
        return hook

    def save_features(self, img_index, feats):
        self.features[img_index] = feats

    def call(self, img1, img2):


        # Extract features using the resnet model
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)

        print(f"Features shape - features1: {features1.shape}, features2: {features2.shape}")
        print("************************************")
        self.save_features('1', features1)
        self.save_features('2', features2)

        # Check feature dimensions for debugging
        print(f"Shape of features1: {features1.shape}")
        print(f"Shape of features2: {features2.shape}")

        # Handle feature dimensions
        if len(features1.shape) == 4:  # [batch_size, height, width, channels]
            # Calculate the mean over spatial dimensions (height and width)
            out1 = tf.reduce_mean(features1, axis=[1, 2])
            out2 = tf.reduce_mean(features2, axis=[1, 2])
        elif len(features1.shape) == 2:  # [batch_size, features]
            # Already flattened, use as-is
            out1, out2 = features1, features2
        else:
            raise ValueError(f"Unexpected feature shape: {features1.shape}")

        # Pass through the projection head (contrastive head)
        out1, out2 = self.contrastive_head(out1), self.contrastive_head(out2)
        """
        WTF 這她媽怎麼可能
        Traceback (most recent call last):
        File "Grad_c.py", line 348, in call
            score = self.measure(out1, out1)
        File "<string>", line 3, in raise_from
        tensorflow.python.framework.errors_impl.InvalidArgumentError: Shapes of all inputs must match: values[0].shape = [1,128] != values[1].shape = [1,2048] [Op:Pack] name: x
        """
        print(f"Projected shape - out1: {out1}, out2: {out2}")
        print("************************************")

        # Calculate the similarity score
        score = self.measure(out1, out2)

        return score


def weight_activation(feats, grads):

    # Apply weight activation using the gradient
    cam = feats * tf.nn.relu(grads)
    cam = tf.reduce_sum(cam, axis=1).numpy()  # Assuming batch size 1 for simplicity
    return cam

def get_gradcam(md, img1, img2):
    grad_cam = GradCAM(md)
    score = grad_cam(img1, img2)
    grad_cam.stop_training = True  # stop tracking gradients after forward pass

    # Extract features and gradients
    cam1 = weight_activation(grad_cam.features['1'], grad_cam.gradients['1'])
    cam2 = weight_activation(grad_cam.features['2'], grad_cam.gradients['2'])

    return cam1, cam2

def get_interactioncam(md, img1, img2, reduction, grad_interact=False):
    grad_cam = GradCAM(md)
    score = grad_cam(img1, img2)
    grad_cam.stop_training = True  # stop tracking gradients

    G1 = grad_cam.gradients['1']
    G2 = grad_cam.gradients['2']

    if grad_interact:
        G1 = tf.reshape(G1, [G1.shape[0], -1, G1.shape[-1]])  # Reshape for batch matrix multiplication
        G2 = tf.reshape(G2, [G2.shape[0], -1, G2.shape[-1]])
        G_ = tf.matmul(G1, G2, transpose_b=True)  # (B, D, D)
        G1 = tf.reduce_max(G_, axis=-1)  # (B, D)
        G2 = tf.reduce_max(G_, axis=1)  # (B, D)
        G1 = tf.expand_dims(G1, axis=-1)
        G2 = tf.expand_dims(G2, axis=-1)

    if reduction == 'mean':
        joint_weight = tf.reduce_mean(grad_cam.features['1'], axis=[1, 2]) * tf.reduce_mean(grad_cam.features['2'], axis=[1, 2])
    elif reduction == 'max':
        joint_weight = tf.reduce_max(grad_cam.features['1'], axis=[1, 2]) * tf.reduce_max(grad_cam.features['2'], axis=[1, 2])
    else:
        B, D, H, W = grad_cam.features['1'].shape
        reshaped1 = tf.reshape(grad_cam.features['1'], [B, H * W, D])
        reshaped2 = tf.reshape(grad_cam.features['2'], [B, H * W, D])
        features1_query = tf.reduce_mean(reshaped1, axis=1, keepdims=True)
        features2_query = tf.reduce_mean(reshaped2, axis=1, keepdims=True)

        attn1 = tf.nn.softmax(tf.matmul(features1_query, reshaped1, transpose_b=True), axis=-1)
        attn2 = tf.nn.softmax(tf.matmul(features2_query, reshaped2, transpose_b=True), axis=-1)

        att_reduced1 = tf.reduce_sum(attn1 * reshaped1, axis=1)
        att_reduced2 = tf.reduce_sum(attn2 * reshaped2, axis=1)
        joint_weight = att_reduced1 * att_reduced2

    joint_weight = tf.expand_dims(joint_weight, axis=-1)

    feats1 = grad_cam.features['1'] * joint_weight
    feats2 = grad_cam.features['2'] * joint_weight

    cam1 = weight_activation(feats1, G1)
    cam2 = weight_activation(feats2, G2)

    return cam1, cam2

def main(argv,augment_first_img = False):
    # Load the .npy file
    npy_path = 'data/水稻_01插秧期_002_95204077_O230227d_27_hr4_moa/水稻_01插秧期_002_95204077_O230227d_27_hr4_moa_0.npy'
    img_array = np.load(npy_path)  # Load the numpy array
    img_array = img_array[:, :, :3]

    # Verify the shape and ensure it is a 3D array (H, W, C)
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        raise ValueError(f"Expected a 3D array with shape (H, W, 3), got {img_array.shape}")

    # Convert to TensorFlow tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    # Apply transformations
    if augment_first_img:
        img1 = tf.expand_dims(aug_transform(img_tensor), axis=0)
    else:
        img1 = tf.expand_dims(pure_transform(img_tensor), axis=0)

    img2 = tf.expand_dims(aug_transform(img_tensor), axis=0)

    #stub example
    img2 = img1
    md = model.Model(4)

    gradcam1, gradcam2 = get_gradcam(md, img1, img2)
    intcam1_mean, intcam2_mean = get_interactioncam(md, img1, img2, reduction = 'mean')
    intcam1_maxmax, intcam2_maxmax = get_interactioncam(md, img1, img2, reduction = 'max', grad_interact = True)
    intcam1_attnmax, intcam2_attnmax = get_interactioncam(md, img1, img2, reduction = 'attn', grad_interact = True)

    fig, axs = plt.subplots(2, 5, figsize=(20,8))
    np.vectorize(lambda ax:ax.axis('off'))(axs)

    denorm = True

    axs[0,0].imshow(show_image(img1[0], squeeze = False, denormalize = denorm))
    axs[0,1].imshow(overlay_heatmap(img1, gradcam1, denormalize = denorm))
    axs[0,1].set_title("Grad-CAM")
    axs[0,2].imshow(overlay_heatmap(img1, intcam1_mean, denormalize = denorm))
    axs[0,2].set_title("IntCAM Mean")
    axs[0,3].imshow(overlay_heatmap(img1, intcam1_maxmax, denormalize = denorm))
    axs[0,3].set_title("IntCAM Max + IntGradMax")
    axs[0,4].imshow(overlay_heatmap(img1, intcam1_attnmax, denormalize = denorm))
    axs[0,4].set_title("IntCAM Attn + IntGradMax")

    axs[1,0].imshow(show_image(img2[0], squeeze = False, denormalize = denorm))
    axs[1,1].imshow(overlay_heatmap(img2, gradcam2, denormalize = denorm))
    axs[1,2].imshow(overlay_heatmap(img2, intcam2_mean, denormalize = denorm))
    axs[1,3].imshow(overlay_heatmap(img2, intcam2_maxmax, denormalize = denorm))
    axs[1,4].imshow(overlay_heatmap(img2, intcam2_attnmax, denormalize = denorm))

    plt.subplots_adjust(wspace=0.01, hspace = 0.01)

if __name__ == '__main__':
    app.run(main)
