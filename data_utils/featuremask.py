import json
import numpy as np
import os
from labelme.utils.shape import shapes_to_label
import matplotlib.pyplot as plt

jsondir = "highest3_experiment"

folder = os.listdir("G:/.shortcut-targets-by-id/1j3VRcg8GosML98qgJcN3di-lgTFtcd0n/data/thumbnails/thumbnails_feature_masks/" + jsondir)


for file in folder:
    if file[-4:] == "json":
        with open("G:/.shortcut-targets-by-id/1j3VRcg8GosML98qgJcN3di-lgTFtcd0n/data/thumbnails/thumbnails_feature_masks/" + jsondir + "/" + file,
                  "r", encoding="utf-8") as f:
            dj = json.load(f)

        # make it to mask
        label_name_to_value = {dj["shapes"][i]["label"]: (i+1) for i in range(len(dj["shapes"]))}
        mask = shapes_to_label((dj['imageHeight'],dj['imageWidth']),shapes=dj["shapes"], label_name_to_value=label_name_to_value)



        mask_img = mask[0].astype("int")#boolean to 0,Convert to 1
        mask_img = mask_img + 1
        file = file[:-5]
        file = file[2:]
        np.save("G:/.shortcut-targets-by-id/1j3VRcg8GosML98qgJcN3di-lgTFtcd0n/data/thumbnails/thumbnails_feature_masks/" + jsondir + "_numpy/" + file + ".npy", mask_img)

