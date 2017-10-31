import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils import pix_to_grid

class YOLOTrainer:

    _dim = 208
    _anchor_num = 7
    _class_num = 2

    def __init__(self, pickled_images=None, pickled_labels=None):
        """

        Args
        ----
        image_dir (str) : path to director that contains the images
        """

        if pickled_images is not None and pickled_labels is not None:
            print("loading from file")
            self.images = np.load(pickled_images)
            self.labels = np.load(pickled_labels)
            print("...done")
            self.process_images_and_labels()


    def process_images_and_labels(self):
        """Take the data loaded from the binary file, or loaded in by
        clean_and_pickle, and format it for the network
        """
        print("loading data onto reel")
        self._load_reel()
        print("...done")

        print("splitting to train and test sets")
        self._split_to_train_and_test()
        print("...done")


    def clean_and_pickle(self, image_dir, labels_dir, show=False, save=True):
        """

        Args
        ----
        image_dir (str)
        labels_dir (str)
        show (bool)
        """

        images = []
        labels = []
        image_paths = glob.glob(image_dir+"/**/*.JPEG", recursive=True)

        for img_path in image_paths:

            raw_img = Image.open(img_path)

            # Don't bother reading in the image if it's a different type
            # I don't want to screw around with them, there's only a few
            if raw_img.mode != "RGB":
                continue

            # Crop the image to a square, based on the smallest dimension
            org_shape = np.array(raw_img.size)
            min_dim = org_shape.min()
            center = org_shape/2
            left = center[0] - min_dim/2
            right = center[0] + min_dim/2
            bottom = center[1] + min_dim/2
            top = center[1] - min_dim/2
            im = raw_img.crop((left, top, right, bottom))

            # resize to the input shape expected by YOLO
            im = im.resize((self._dim, self._dim))

            # construct the path where the image label is held
            _img_file_name = os.path.splitext(
                    os.path.basename(img_path)
                    )[0] + ".txt"
            _img_sub_dir = os.path.basename(os.path.dirname(img_path))
            label_path = os.path.join(labels_dir, _img_sub_dir, _img_file_name)

            # Read in the label and adjust for the reshaped image 
            with open(label_path) as f:
                image_labels = []

                for line in f:

                    new_center = np.array(im.size)/2
                    c, _rx, _ry, _rw, _rh = np.asarray(line.split(), np.float32)

                    # the x and y scaling factor after the image resizing
                    alpha_x = org_shape[0] / min_dim
                    alpha_y = org_shape[1] / min_dim

                    # Find the new center in the adjusted image (as % of img dim)
                    rx = alpha_x * (_rx - .5) + .5
                    ry = alpha_y * (_ry - .5) + .5

                    # Force the point to be inside of the image
                    rx = np.minimum(rx, .99) 
                    rx = np.maximum(rx, 0)
                    ry = np.minimum(ry,  .99)
                    ry = np.maximum(ry, 0)

                    # Find the new width and height for the adjusted image
                    rw = _rw * alpha_x
                    rh = _rh * alpha_y

                    # save the new image label
                    image_labels.append([c, rx, ry, rw, rh])

                # Convince yourself the conversion was correct
                if show and input("draw? ") == "y":
                    raw_img.show()
                    x = rx * self._dim
                    y = ry * self._dim
                    w = rw * self._dim
                    h = rh * self._dim
                    r=20
                    draw = ImageDraw.Draw(im)
                    draw.rectangle(
                            (x-w/2, y-h/2, x+w/2, y+h/2), 
                            fill=None, 
                            outline=0)
                    im.show()
            
            # Flatten and normalize images
            im = np.array(im)/255
            
            # append to the data set
            images.append(im)
            labels.append(image_labels)

        self.images = np.asarray(images)
        self.labels = np.asarray(labels)

        # save the data to a binary file for future use
        if save:
            print("saving data set")
            np.save("./data/images", self.images)
            np.save("./data/labels", self.labels)

        # load the newly imported in the yolo format
        self.process_images_and_labels()


    def _load_reel(self):
        """Convert labels into tensor form

        Using the given labels of the form:
            <class> <x coor> <y coor> <w coor> <h coor>
            
        Generate a tensor that matches the output style of yolo
        such that a 7 x 7 matrix, corresponding to the 
        """

        #images = np.reshape(self.images, (-1, self._dim, self._dim, 3))

        indicator_obj = []
        xx = []
        yy = []
        ww = []
        hh = []
        cc = []

        for label_set in self.labels:
            _indicator_obj = np.zeros(
                    shape=((self._anchor_num, self._anchor_num, 1))
                    )
            _xx = np.zeros_like(_indicator_obj)
            _yy = np.zeros_like(_indicator_obj)
            _ww = np.zeros_like(_indicator_obj)
            _hh = np.zeros_like(_indicator_obj)
            _cc = np.zeros(
                    shape=((self._anchor_num, self._anchor_num, self._class_num))
                    )

            for label in label_set:
                c, rx, ry, rw, rh = label
                _x = rx * self._dim
                _y = ry * self._dim
                i, j = pix_to_grid(_x, _y)

                _indicator_obj[i,j,...] = 1
                _xx[i,j,...] = _x
                _yy[i,j,...] = _y
                _ww[i,j,...] = rw * self._dim
                _hh[i,j,...] = rh * self._dim
                _cc[i,j,int(c)] = 1


            indicator_obj.append(_indicator_obj)
            xx.append(_xx)
            yy.append(_yy)
            ww.append(_ww)
            hh.append(_hh)
            cc.append(_cc)

        self.gndTru = np.concatenate((cc, xx, yy, ww, hh, indicator_obj), axis=3)

    def _split_to_train_and_test(self):
        """Generate train an test sets from the batch 

        If this operation has already been done, don't to it again, otherwise
        you risk mixing test data into training set

        Attributes
        ----------
        imgTrain
        imgTest
        gndTruTrain
        gndTruTest

        """
        #images = self.images.reshape(-1, 208, 208, 3)

        try:
            self.imgTrain
            self.imgTest
            self.gndTruTrain
            self.gndTruTest
        except:
            self.imgTrain, self.imgTest, self.gndTruTrain, self.gndTruTest = train_test_split(
                    self.images, 
                    self.gndTru, 
                    test_size=0.33, 
                    random_state=42)

            self.imgTrain, self.imgTest, self.gndTruTrain, self.gndTruTest
        else:
            print("Data already split into train and test sets. Doing nothing...")


    def get_batches(self, batch_sz):
        """split the data up into batches

        TODO
        ----
        something about the yield keyword
        """

        _iter = []
        images, labels = shuffle(self.imgTrain, self.gndTruTrain)
        batch_grp = np.asarray(range(batch_sz))
        while True:
            batch_grp += batch_sz
            try:
                _iter.append(
                    (images[batch_grp], labels[batch_grp])
                )
            except:
                break

        return _iter

    def test_self(self):
        """Run through a series of tests to make sure the data is okay

        Tests
        -----
        Make sure there are no NAN values
        """

        print("running tests on the input data")
        assert not np.any(np.isnan(self.images))
        print("All values are real (no NAN)")
        print("image shapes: ", self.images.shape)
        print("label shapes: ", self.gndTru.shape)
        print("...done")

def convert_raw_images(yolo_trainer):
    image_dir = "./data/images"
    labels_dir = "./data/labels"
    yolo_trainer.clean_and_pickle(image_dir, labels_dir)

if __name__ == "__main__":
    yolo_trainer = YOLOTrainer("./data/images.npy", "./data/labels.npy")
    #convert_raw_images(yolo_trainer)
    yolo_trainer.test_self()
    batches = yolo_trainer.get_batches(64)



