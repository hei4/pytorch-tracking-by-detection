import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
import torch
from torchvision import utils as vutils

from trackers.base_tracker import BaseTracker

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

N = 20
cmap = plt.get_cmap(f'tab{N}')

def make_colors(indices):
    colors = np.array(cmap(indices % N))[:, :3]   # alphaは使用しない
    colors = (255. * colors).astype(np.uint8).tolist()
    colors = [tuple(color) for color in colors]     # tupleを要求されるのでlistから変換

    return colors

def write_tracklet_image(image, active_tracklets, remain_tracklets):
    active_boxes, active_labels, active_ids = BaseTracker.get_pascal_boxes_labels_ids(active_tracklets)
    result_image = vutils.draw_bounding_boxes(
        (image.to('cpu') * 255.).to(torch.uint8),
        boxes=active_boxes,
        labels=[f'{label_to_class[label]}:{id}' for label, id in zip(active_labels, active_ids)],
        colors=make_colors(active_ids),
        fill=True,
        font='/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf',
        font_size=16
    )

    remain_boxes, remain_labels, remain_ids = BaseTracker.get_pascal_boxes_labels_ids(remain_tracklets)
    result_image = vutils.draw_bounding_boxes(
        result_image,
        boxes=remain_boxes,
        labels=[f'{label_to_class[label]}:{id}' for label, id in zip(remain_labels, remain_ids)],
        colors=make_colors(remain_ids),
        fill=False,
        font='/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf',
        font_size=16
    )

    return result_image


def write_tracklet_image_with_covariance(image, active_tracklets, remain_tracklets, filename):
    result_image = write_tracklet_image(image, active_tracklets, remain_tracklets)

    dpi = matplotlib.rcParams['figure.dpi']
    _, image_height, image_width = result_image.shape
    fig = plt.figure(figsize=(image_width/dpi, image_height/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    
    ax.imshow(result_image.permute(1, 2, 0).numpy(), vmin=0, vmax=255)

    for tracklet in active_tracklets + remain_tracklets:
        pos, width, height, angle = tracklet.get_ellipse()
        id = tracklet.get_id()
        ellipse = patches.Ellipse(
            xy=pos,
            width=width,
            height=height,
            angle=angle,
            color=cmap(id%N),
            fill=False,
            alpha=0.5,
            clip_on=True
        )
        ax.add_patch(ellipse)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    plt.savefig(filename)
    plt.close()


def write_tracklet_image_with_particle(image, active_tracklets, remain_tracklets, filename):
    result_image = write_tracklet_image(image, active_tracklets, remain_tracklets)

    dpi = matplotlib.rcParams['figure.dpi']
    _, image_height, image_width = result_image.shape
    fig = plt.figure(figsize=(image_width/dpi, image_height/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    
    ax.imshow(result_image.permute(1, 2, 0).numpy(), vmin=0, vmax=255)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    for tracklet in active_tracklets + remain_tracklets:
        points = tracklet.get_points()
        mask = (0 <= points[0]) & (points[0] < image_width-1) & (0 <= points[1]) & (points[1] < image_height-1)
        id = tracklet.get_id()
        ax.scatter(points[0, mask], points[1, mask], marker='.', s=1, color=cmap(id%N), alpha=0.05, clip_on=True)

    plt.savefig(filename)
    plt.close()
