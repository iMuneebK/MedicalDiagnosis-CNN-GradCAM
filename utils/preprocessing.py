import tensorflow as tf

def preprocess_image(image_path, img_height=224, img_width=224):
    """Load and preprocess an image for the CNN."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0  # Normalize to [0,1]
    return img
