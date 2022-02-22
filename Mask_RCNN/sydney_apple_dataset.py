import os, sys
import json
import numpy as np
import skimage.draw
from mrcnn import utils

############################################################
#  Dataset
# For the Sydney Orchard Dataset
############################################################

class SydneyAppleDataset(utils.Dataset):

    def load_annotations(self, mypath):

        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        dataMap = {}
        for file in onlyfiles:
            img_id = file[:-4]
            dataList = []
            with open(join(mypath,file), 'r') as inFile:
                strBuffer = inFile.read()
                rawStrList = strBuffer.splitlines()
                print(len(rawStrList))
                if len(rawStrList) >= 2:
                    rawStrList = rawStrList[1:]
                    for data in rawStrList:
                        dataList.append(data.split(','))

            dataMap[img_id] = dataList

        return dataMap

    def load_object(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("apple", 1, "apple")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations_dir = os.path.join(dataset_dir, 'annotations')
        # Load annotations
        annotations = self.load_annotations(annotations_dir)

        # Add images
        for id, value in annotations.items():

            image_path = os.path.join(dataset_dir, id+'.png')
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "apple",
                image_id=id,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=value)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "apple":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"]+50, info["width"]+50, len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.circle(float(p[2]), float(p[1]),float(p[3]))

            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "apple":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)




