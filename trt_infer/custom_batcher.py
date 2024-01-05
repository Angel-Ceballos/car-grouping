import torchvision.transforms as transform
import os
import sys
import cv2

class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, input, shape, dtype):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param preprocessor: Set the preprocessor to use, V1 or V2, depending on which network is being used.
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []
        self.center = None
        self.transform = transform.Compose(
            [
                transform.ToTensor(),
                transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

    def preprocess_image(self, img_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes cropping,
        resizing, normalization, data type casting, and transposing.
        This Image Batcher implements two algorithms:
        * V2: The algorithm for EfficientNet V2, as defined in automl/efficientnetv2/preprocessing.py.
        * V1: The algorithm for EfficientNet V1, aka "Legacy", as defined in automl/efficientnetv2/preprocess_legacy.py.
        :param image_path: The path to the image on disk to load.
        :return: A numpy array holding the image sample, ready to be contacatenated into the rest of the batch.
        """

        def pad_crop(image, c_pos, m=64):
            """
            A subroutine to implement padded cropping. This will create a center crop of the image, padded by 32 pixels.
            :param image: The PIL image object
            :return: The PIL image object already padded and cropped.
            """
            crop_area = image[c_pos[1]-m: c_pos[1]+m, c_pos[0]-m:c_pos[0]+m]
            # return cv2.cvtColor(crop_area, cv2.COLOR_BGR2RGB)
            return crop_area

        src_img = cv2.imread(img_path)
        crop_image = pad_crop(src_img, c_pos=self.center)
        crop_image = self.transform(crop_image)
        crop_image = crop_image[None,:,:,:]

        return src_img ,crop_image

    # @profile
    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding two items per iteration: a numpy array holding a batch of images, and the list of
        paths to the images loaded within this batch.
        """
        for image_path in self.images:
            src_image, data = self.preprocess_image(image_path)
            yield src_image, data