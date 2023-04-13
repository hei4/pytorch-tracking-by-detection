import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import models as vmodels
from torchvision import transforms as vtf

from trackers.tracker import Tracker
from dataloaders.dataset import SequenceImagesDataset
from utils.util import write_tracklet_image
from utils.util import class_to_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str)
    parser.add_argument('result_root', type=str)
    args = parser.parse_args()
    
    os.makedirs(args.result_root, exist_ok=True)

    transform = vtf.Compose([
        vtf.Resize([540, 960]),
        vtf.Normalize(mean=[0., 0., 0.], std=[255., 255., 255.])
    ])

    dataset = SequenceImagesDataset(args.data_root, transform=transform)
    print(f'dataset... {len(dataset)}')

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False, pin_memory=True, collate_fn=None)
    print(f'dataloader... {len(dataloader)}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    detector = vmodels.detection.retinanet_resnet50_fpn(pretrained=True)
    detector.eval()
    detector.to(device)

    trackers = [
        Tracker('person'),
        Tracker('bicycle'),
        Tracker('handbag')
    ]

    frame = 0
    for images in dataloader:
        images = [image.to(device) for image in images]

        with torch.no_grad():
            batch_detections = detector(images)

        for image, detections in zip(images, batch_detections):
            print(f'frame: {frame}')

            detect_boxes = detections['boxes'].to('cpu')
            detect_scores = detections['scores'].to('cpu')
            detect_labels = detections['labels'].to('cpu').to(torch.int64)
            
            all_active_track_boxes = []
            all_active_track_ids = []
            all_active_track_classnames = []
            all_remain_track_boxes = []
            all_remain_track_ids = []
            all_remain_track_classnames = []
            
            for tracker in trackers:
                classname = tracker.get_classname()
                mask = detect_labels == class_to_label[classname]

                active_track_boxes, active_track_ids, remain_track_boxes, remain_track_ids = tracker(detect_boxes[mask], detect_scores[mask])
                
                all_active_track_boxes.append(active_track_boxes)
                all_active_track_ids.append(active_track_ids)
                all_active_track_classnames.extend([classname] * len(active_track_boxes))
                
                all_remain_track_boxes.append(remain_track_boxes)
                all_remain_track_ids.append(remain_track_ids)
                all_remain_track_classnames.extend([classname] * len(remain_track_boxes))

            all_active_track_boxes = torch.cat(all_active_track_boxes)
            all_active_track_ids = torch.cat(all_active_track_ids)
            all_remain_track_boxes = torch.cat(all_remain_track_boxes)
            all_remain_track_ids = torch.cat(all_remain_track_ids)

            write_tracklet_image(
                image,
                all_active_track_boxes,
                all_active_track_ids,
                all_active_track_classnames,
                all_remain_track_boxes,
                all_remain_track_ids,
                all_remain_track_classnames,
                f'{args.result_root}/{frame:06}.png'
            )

            frame += 1

if __name__ == '__main__':
    main()