import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CracksDataset(CustomDataset):
    """Cracks dataset.

    In my segmentation map annotation for IGN, 0 represents background which is
    treated as a separate class so ``reduce_zero_label`` is fixed to False.
    You can choose to have only 1 class (i.e., crack) and set 
    ``reduce_zero_label`` to True and train your model with only 1 class.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('Background', 'Crack')

    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(CracksDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
