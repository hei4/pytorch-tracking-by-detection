from torchvision.ops import box_convert

class Detection():
    def __init__(self, box, score) -> None:
        self.pascal_box = box.numpy()   # Pascal形式での保存
        self.yolo_box = box_convert(box, 'xyxy', 'cxcywh').numpy()  # YOLO形式での保存
        self.score = score

    def get_pascal_box(self):
        return self.pascal_box

    def get_yolo_box(self):
        return self.yolo_box
    
    def get_score(self):
        return self.score