import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import models as vmodels
from torchvision import transforms as vtf

from trackers.kalman_tracker import KalmanTracker
from trackers.particle_tracker import ParticleTracker
from trackers.byte_tracker import ByteTracker
from dataloaders.dataset import SequenceImagesDataset
from utils.util import write_tracklet_image_with_covariance, write_tracklet_image_with_particle
from utils.util import class_to_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tracking', type=str, choices=['kalman', 'particle', 'byte'])
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

    if args.tracking == 'kalman':
        Tracker = KalmanTracker
        write_image = write_tracklet_image_with_covariance
    elif args.tracking == 'particle':
        Tracker = ParticleTracker
        write_image = write_tracklet_image_with_particle
    elif args.tracking == 'byte':
        Tracker = ByteTracker
        write_image = write_tracklet_image_with_covariance

    else:
        None    # ここには到達しない筈

    trackers = [
        Tracker(class_to_label['person']),
        Tracker(class_to_label['bicycle']),
        Tracker(class_to_label['car']),
        Tracker(class_to_label['handbag'])
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
            
            all_active_tracklets = []
            all_remain_tracklets = []
            
            for tracker in trackers:
                mask = detect_labels == tracker.get_label()
                active_tracklets, remain_tracklets = tracker(detect_boxes[mask], detect_scores[mask])
                
                all_active_tracklets.extend(active_tracklets)
                all_remain_tracklets.extend(remain_tracklets)

            write_image(
                image,
                all_active_tracklets,
                all_remain_tracklets,
                f'{args.result_root}/{frame:06}.png'
            )

            frame += 1

if __name__ == '__main__':
    main()