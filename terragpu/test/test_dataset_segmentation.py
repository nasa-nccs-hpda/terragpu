from terragpu.ai.deep_learning.datasets.segmentation_dataset \
    import SegmentationDataset

prepare_data = True
images_regex = '/Users/jacaraba/Desktop/development/ilab/vhr-cloudmask/data/images/*.tif'
labels_regex = '/Users/jacaraba/Desktop/development/ilab/vhr-cloudmask/data/labels/*.tif'
dataset_dir = '/Users/jacaraba/Desktop/development/ilab/vhr-cloudmask/data/dataset'

dataset = SegmentationDataset(
    prepare_data=prepare_data,
    images_regex=images_regex,
    labels_regex=labels_regex,
    dataset_dir=dataset_dir,
    tile_size=128,
    seed=24,
    max_patches=0.000001,
    augment=True,
    chunks={'band': 1, 'x': 2048, 'y': 2048},
    input_bands=['CB', 'B', 'G', 'Y', 'Red', 'RE', 'N1', 'N2'],
    output_bands=['B', 'G', 'Red'],
    pytorch=True)
