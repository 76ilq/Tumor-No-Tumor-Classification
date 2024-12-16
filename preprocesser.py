import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random

class ImagePreprocessor:
    def __init__(self, image_size=(224, 224)):
        """
        Initializes the ImagePreprocessor with the desired image size.
        :param image_size: Tuple specifying (width, height) for resizing.
        """
        self.image_size = image_size

    def load_image(self, image_path):
        """
        Loads an image from the specified path.
        :param image_path: Path to the image file.
        :return: PIL Image object if successful, None otherwise.
        """
        try:
            img = Image.open(image_path).convert("RGB")  # Convert to RGB (if grayscale, convert to RGB)
            return img
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            return None

    def resize_image(self, img):
        """
        Resizes the image to the specified size.
        :param img: PIL Image object.
        :return: Resized PIL Image object.
        """
        return img.resize(self.image_size)

    def normalize_image(self, img):
        """
        Normalize the image pixel values to the range [0, 1].
        :param img: PIL Image object.
        :return: Numpy array with normalized pixel values.
        """
        return np.array(img) / 255.0

    def apply_gaussian_blur(self, img, radius=1):
        """
        Apply Gaussian blur to the image.
        :param img: PIL Image object.
        :param radius: Radius for the Gaussian blur.
        :return: Blurred PIL Image object.
        """
        return img.filter(ImageFilter.GaussianBlur(radius))

    def augment_image(self, img):
        """
        Apply random augmentations to the image:
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (0, 90, 180, 270 degrees)
        - Random crop
        :param img: PIL Image object.
        :return: Augmented PIL Image object.
        """
        # Random horizontal flip
        if random.choice([True, False]):
            img = ImageOps.mirror(img)
        # Random vertical flip
        if random.choice([True, False]):
            img = ImageOps.flip(img)

        # Random rotation (0, 90, 180, 270 degrees)
        angle = random.choice([0, 90, 180, 270])
        img = img.rotate(angle)

        # Random cropping
        crop_margin = 5  # Crop 5 pixels from each side
        width, height = img.size
        img = img.crop((crop_margin, crop_margin, width - crop_margin, height - crop_margin))
        img = img.resize(self.image_size)  # Resize back to original size
        return img

    def preprocess(self, image_path, augment=True):
        """
        Full preprocessing pipeline: load, resize, normalize, apply blur, augment.
        :param image_path: Path to the image file.
        :param augment: Boolean indicating whether to apply random augmentations.
        :return: Preprocessed numpy array.
        """
        img = self.load_image(image_path)
        if img is None:
            return None
        
        img = self.resize_image(img)
        img = self.apply_gaussian_blur(img)
        
        # Apply augmentation if specified
        if augment:
            img = self.augment_image(img)

        # Normalize image (convert to numpy array with pixel values between 0 and 1)
        return self.normalize_image(img)
