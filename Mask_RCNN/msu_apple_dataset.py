import os, sys
import json
import numpy as np
import skimage.draw
from mrcnn import utils
import mask_generation
############################################################
#  Dataset
############################################################


class MSUAppleDataset(utils.Dataset):

    def load_annotations(self, file_name):
        dataMap = {}
        file = file_name
        dataList = []
        with open(file, 'r') as inFile:
            strBuffer = inFile.read()
            rawStrList = json.loads(strBuffer)
            rawStrList = rawStrList['_via_img_metadata']

            for key, value in rawStrList.items():
                dataMap[value['filename']] = value['regions']

        return dataMap

    def load_object(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("apple", 1, "apple")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations_file = os.path.join(dataset_dir, 'annotations.json')
        # Load annotations
        annotations = self.load_annotations(annotations_file)

        # Add images
        for id, value in annotations.items():

            image_path = os.path.join(dataset_dir, id)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # remove small annotations smaller than 30x30 pixels in 1280x720 resolution
            new_values = [i for i in value if i['shape_attributes']['width'] >= 30 and i['shape_attributes']['height'] >= 30]

            self.add_image(
                "apple",
                image_id=id,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=new_values)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "apple":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # image = skimage.io.imread(info['path'])
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, item in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # for the circle annotation dataset
            # rr, cc = skimage.draw.circle(float(p[2]), float(p[1]),float(p[3]))

            # Get indexes of pixels inside the polygon and set them to 1
            anno = item['shape_attributes']
            if anno['name'] == 'rect':
                start = (anno['y'], anno['x'])
                height_extent = info['height'] - anno['y'] if anno['height'] + anno['y'] > info["height"] else anno['height']
                width_extent = info['width'] - anno['x'] if anno['width'] + anno['x'] > info["width"] else anno['width']
                extent = (height_extent, width_extent)

                rr, cc = skimage.draw.rectangle(start, extent=extent)

                # mask[rr, cc, i] = mask_generation.kmeans_mask_produce(image[rr, cc, :])


            elif anno['name'] == 'polygon':
                x = np.asarray(anno['all_points_x'])
                y= np.asarray(anno['all_points_y'])
                rr, cc = skimage.draw.polygon(y, x)

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



