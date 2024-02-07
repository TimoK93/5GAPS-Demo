[![Pylint](https://github.com/TimoK93/5GAPS-Demo/actions/workflows/pylint.yml/badge.svg)](https://github.com/TimoK93/5GAPS-Demo/actions/workflows/pylint.yml)

# 5GAPS-Demo

---

## Installation

If you have no existing appllication or environment, you can install a 
predefined environment via conda and our environment.yml file:

```bash
conda env create -f environment.yml
```

Otherwise, you need to make sure that the following requirements are met:
- Python 3.8 or later
- PyTorch 1.8 or later (see instructions [here](https://pytorch.org/get-started/locally/))
- torchvision 0.9 or later
- CUDA 9.2 or later
- MMDetection (see instructions [here](https://mmdetection.readthedocs.io/en/latest/get_started.html or in the next section)
- CocoAPI (see instructions in the next sections)

If all other requirements are satisfied, MMDetection can be installed using the 
following commands:
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

The CocoAPI can be installed using the following commands:
```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```

The 5GAPS-Demo package can be installed via pip:
```bash
pip install https://github.com/TimoK93/5GAPS-Demo.git
```

Please verify your installation by running the following command:
```bash
fivegaps-test
```

---

## Usage

You can use the module by importing it in your Python code. The following
example shows how to import and use the module:

```python
import cv2

from fivegaps.models.models import AVAILABLE_MODELS
from fivegaps.segmentor import Segmentor

name = AVAILABLE_MODELS['panoptic_segmentation'][0]
segmentor = Segmentor(name)
img = cv2.imread(r"..\assets\demo.jpg")
# Get an array with the panoptic segmentation labels for each pixel
#     where a label is encoded as 1000 * category_id + instance_id
panoptic_segmentation = segmentor.segment(img)

# Convert it to semantic segmentation
#     where each pixel is assigned the category_id
semantic_segmentation = segmentor.convert_panoptic_to_sem_seg(panoptic_segmentation)

# Get the label to class mapping, e.g. 0: background, 1: person, 2: car, ...
label_map = segmentor.get_label_map()
print(label_map)
```

You can run a speed test by running the following command:
```bash
fivegaps-speed-test
```


---
## Contributing

Contributions are welcome! For bug reports or requests please
[submit an issue](www.github.com/TimoK93/ctc-metrics/issues). For new features
please [submit a pull request](www.github.com/TimoK93/ctc-metrics/pulls).

If you want to contribute, please check your code with pylint and the
pre-commit hooks before submitting a pull request:

```bash
pip install pre-commit, pylint
pre-commit run --all-files
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file 
for details.

---

## Acknowledgements

We want to acknowledge the following projects and their contributors for 
providing the open-source software that we use in our project:

- [OpenMMLab](https://github.com/open-mmlab)
- [PyTorch](https://pytorch.org/)
