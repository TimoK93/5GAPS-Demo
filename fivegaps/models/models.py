import os
from mim import download

from fivegaps.models.label_maps import coco


DEST_ROOT = os.path.join(os.path.dirname(__file__))


AVAILABLE_MODELS = {
    'instance_segmentation': [
        'rtmdet_tiny_8xb32-300e_coco',
    ],
    'panoptic_segmentation': [
        'panoptic_fpn_r50_fpn_mstrain_3x_coco',
        'mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic',
        'mask2former_r101_8xb2-lsj-50e_coco-panoptic',
        'mask2former_r50_8xb2-lsj-50e_coco-panoptic',
        'mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic',
        'mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic',
        'mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic',
        'mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic',
    ],
}

LABEL_MAPS = {
    'panoptic_fpn_r50_fpn_mstrain_3x_coco': coco,
    'rtmdet_tiny_8xb32-300e_coco': coco,
    'mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic': coco,
    'mask2former_r101_8xb2-lsj-50e_coco-panoptic': coco,
    'mask2former_r50_8xb2-lsj-50e_coco-panoptic': coco,
    'mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic': coco,
    'mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic': coco,
    'mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic': coco,
    'mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic': coco,
}


def get_config_file(name):
    return os.path.join(DEST_ROOT, f'{name}.py')


def get_checkpoint_file(name):
    all_files = [
        x for x in os.listdir(os.path.join(DEST_ROOT)) if x.endswith('.pth')]
    for file in all_files:
        if name in file:
            return os.path.join(DEST_ROOT, file)
    return None


def check_if_downloaded(name):
    return all([
        os.path.exists(get_config_file(name)),
        get_checkpoint_file(name)
    ])


def download_model(name):
    download('mmdet', [name], dest_root=DEST_ROOT)


def download_all_models():
    for _, models in AVAILABLE_MODELS.items():
        for model in models:
            if not check_if_downloaded(model):
                download_model(model)
