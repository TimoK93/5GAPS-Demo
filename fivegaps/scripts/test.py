import cv2

from fivegaps.models.models import AVAILABLE_MODELS
from fivegaps.segmentor import Segmentor


def is_running():
    try:
        name = AVAILABLE_MODELS['panoptic_segmentation'][1]
        segmentor = Segmentor(name)
        img = cv2.imread(r"..\assets\demo.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmentor.segment(img)
    except Exception as e:  # pylint: disable=broad-except
        print(e)
        return False
    return True


if __name__ == "__main__":
    print("This is the test script from the fivegaps package.")
    print("May it takes some time to execute...")
    if is_running():
        print("... seems to work!")
    else:
        print('\033[91m', "... seems like something went wrong!", '\033[0m')
