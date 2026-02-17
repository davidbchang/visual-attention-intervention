import json

import cv2
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class VSRDataset(Dataset):
    def __init__(self, img_feature_path, json_path, coco_annotations_dir=None): 
        if coco_annotations_dir:
            coco_data = COCO(f"{coco_annotations_dir}/instances_train2017.json")
            categories = coco_data.loadCats(coco_data.getCatIds())
            self.category_names = [category["name"] for category in categories]
            self.category_names = sorted(self.category_names, key=lambda x: -len(x))

        self.data_json = []
        self.imgs = {}

        with open(json_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                image_filename = j_line["image"]
                image = cv2.imread(f"{img_feature_path}/{image_filename}")
                self.imgs[image_filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if "subj" in j_line and "obj" in j_line:
                    entities = [j_line["subj"], j_line["obj"]]
                else:
                    entities = None
                    if coco_annotations_dir:
                        entities = []
                        for category_name in self.category_names:
                            if category_name in j_line["caption"]:
                                entities.append(category_name)

                                if len(entities) == 2:
                                    break

                        assert len(entities) == 2, \
                            f"Num of entities != 2 for image: {j_line["image"]}, caption: {j_line["caption"]}. entities: {entities}"

                j_line["entities"] = entities

                self.data_json.append(j_line)
            
    def __getitem__(self, idx):
        data_point = self.data_json[idx]

        return {
            "image": self.imgs[data_point["image"]],
            "image_name": data_point["image"],
            "image_link": data_point["image_link"],
            "caption": data_point["caption"],
            "entities": data_point["entities"],
            "label": data_point["label"]
        }

    def __len__(self):
        return len(self.data_json)
