import time
import cv2
from mmdet.apis import init_detector, inference_detector

from fivegaps.models.models import (
    AVAILABLE_MODELS, LABEL_MAPS, get_config_file, get_checkpoint_file,
    check_if_downloaded, download_model
)


def print_available_models():
    print("Available panoptic segmentation models:")
    for v in AVAILABLE_MODELS['panoptic_segmentation']:
        print(f"    {v}")


class Segmentor:
    def __init__(self, name, device='cuda:0', download=True):
        if name in AVAILABLE_MODELS['instance_segmentation']:
            self.pan_seg = False
        elif name in AVAILABLE_MODELS['panoptic_segmentation']:
            self.pan_seg = True
        else:
            raise ValueError(f"Model {name} not found.")

        self.label_map = LABEL_MAPS[name]

        if not check_if_downloaded(name):
            if download:
                download_model(name)
            else:
                raise ValueError(f"Model {name} not found. Use download=True "
                                 f"to download it.")

        config_file = get_config_file(name)
        checkpoint_file = get_checkpoint_file(name)

        self.model = init_detector(config_file, checkpoint_file, device=device)

    def segment(self, img):
        result = inference_detector(self.model, img)
        segments = result.pred_panoptic_seg.sem_seg[0]
        return segments

    def convert_panoptic_to_sem_seg(self, img):
        return img // 1000

    def get_label_map(self):
        raise self.label_map


if __name__ == "__main__":
    n = AVAILABLE_MODELS['panoptic_segmentation'][1]
    segmentor = Segmentor(n)
    i = cv2.imread(r"..\assets\demo.jpg")
    segmentor.segment(i)
    start = time.time()
    segmentor.segment(i)
