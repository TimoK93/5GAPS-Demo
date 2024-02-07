import time
import argparse
import cv2

from fivegaps.models.models import AVAILABLE_MODELS
from fivegaps.segmentor import Segmentor
from fivegaps import DEMO_IMAGE


def get_fps(name, device):
    img = cv2.imread(DEMO_IMAGE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    segmentor = Segmentor(name, device)
    # Init step
    segmentor.segment(img)
    t0 = time.time()
    for _ in range(10):
        segmentor.segment(img)
    del segmentor
    t1 = time.time()
    return 10 / (t1 - t0)


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='Speed-Test')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--repeats', type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("### This is the Speed Test from the fivegaps package ###")
    available_models = AVAILABLE_MODELS['panoptic_segmentation']
    print("The following models will be testet:")
    for model in available_models:
        print(f"    {model}")

    print("The speed test results...")
    for model in available_models:
        print(f"    {model}: {get_fps(model, 'cuda:0')} FPS")
