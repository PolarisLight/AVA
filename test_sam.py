from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import json


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


img_name = "53.jpg"
img = cv2.imread("dataset/images/" + img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=50)
tic = time.time()
masks = mask_generator.generate(img)

print(f'everything prompt time: {time.time() - tic}')

for mask in masks:
    mask["segmentation"] = mask["segmentation"].tolist()
with open("output/" + img_name + ".json", "w") as f:
    json.dump(masks, f)
# plt.figure(figsize=(20, 20))
# plt.imshow(img)
# show_anns(masks)
# plt.axis('off')
# plt.savefig("output/"+img_name, bbox_inches='tight')
# plt.show()
# for mask in masks:
#     img = cv2.rectangle(img,mask["bbox"])
