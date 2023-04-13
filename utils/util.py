import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.io import write_png
from torchvision import utils as vutils


def make_colors(indices):
    N = 20
    cmap = plt.get_cmap(f'tab{N}')

    colors = np.array(cmap(indices % N))[:, :3]   # alphaは使用しない
    colors = (255. * colors).astype(np.uint8).tolist()
    colors = [tuple(color) for color in colors]     # tupleを要求されるのでlistから変換

    return colors


label_to_class = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
class_to_label = {}
for i, label in enumerate(label_to_class):
    class_to_label[label] = i

def write_detection_image(image, boxes, scores, labels, filename):
    result_image = vutils.draw_bounding_boxes(
        (image.to('cpu') * 255.).to(torch.uint8),
        boxes=boxes,
        labels=[f'{label_to_class[label]}:{score:.3f}' for label, score in zip(labels, scores)],
        colors=make_colors(labels),
        font='/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf',
        font_size=24
    )
    write_png(result_image, filename)

def write_tracklet_image(image, active_boxes, active_ids, active_classnames, remain_boxes, remain_ids, remain_classnames, filename):
    result_image = vutils.draw_bounding_boxes(
        (image.to('cpu') * 255.).to(torch.uint8),
        boxes=torch.tensor(active_boxes),
        labels=[f'{name}:{id}' for id, name in zip(active_ids, active_classnames)],
        colors=make_colors(active_ids),
        fill=True,
        font='/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf',
        font_size=24
    )

    result_image = vutils.draw_bounding_boxes(
        result_image,
        boxes=torch.tensor(remain_boxes),
        labels=[f'{name}:{id}' for id, name in zip(remain_ids, remain_classnames)],
        colors=make_colors(remain_ids),
        fill=False,
        font='/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf',
        font_size=24
    )

    write_png(result_image, filename)