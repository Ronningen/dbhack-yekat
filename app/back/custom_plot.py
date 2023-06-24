from copy import deepcopy

import numpy as np
import torch

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils import deprecation_warn
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.engine.results import Results

def cuctom_plot(
        self: Results,
        conf=True,
        line_width=None,
        font_size=None,
        font='Arial.ttf',
        pil=False,
        img=None,
        img_gpu=None,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        **kwargs  # deprecated args TODO: remove support in 8.2
):
    """
    Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

    Args:
        conf (bool): Whether to plot the detection confidence score.
        line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
        font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
        font (str): The font to use for the text.
        pil (bool): Whether to return the image as a PIL Image.
        img (numpy.ndarray): Plot to another image. if not, plot to original image.
        img_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
        kpt_line (bool): Whether to draw lines connecting keypoints.
        labels (bool): Whether to plot the label of bounding boxes.
        boxes (bool): Whether to plot the bounding boxes.
        masks (bool): Whether to plot the masks.
        probs (bool): Whether to plot classification probability

    Returns:
        (numpy.ndarray): A numpy array of the annotated image.
    """
    # Deprecation warn TODO: remove in 8.2
    if 'show_conf' in kwargs:
        deprecation_warn('show_conf', 'conf')
        conf = kwargs['show_conf']
        assert type(conf) == bool, '`show_conf` should be of boolean type, i.e, show_conf=True/False'

    if 'line_thickness' in kwargs:
        deprecation_warn('line_thickness', 'line_width')
        line_width = kwargs['line_thickness']
        assert type(line_width) == int, '`line_width` should be of int type, i.e, line_width=3'

    names = self.names
    annotator = Annotator(deepcopy(self.orig_img if img is None else img),
                            line_width,
                            font_size,
                            font,
                            pil,
                            example=names)
    pred_boxes, show_boxes = self.boxes, boxes
    pred_masks, show_masks = self.masks, masks
    pred_probs, show_probs = self.probs, probs
    keypoints = self.keypoints
    if pred_masks and show_masks:
        if img_gpu is None:
            img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
            img_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
                2, 0, 1).flip(0).contiguous() / 255
        idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
        annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=img_gpu)

    if pred_boxes and show_boxes:
        for d in reversed(pred_boxes):
            c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            name = ('' if id is None else f'id:{id} ') + names[c]
            label = (f'{name} {conf:.2f}' if conf else name) if labels else None
            annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

    if pred_probs is not None and show_probs:
        n5 = min(len(names), 5)
        top5i = pred_probs.argsort(0, descending=True)[:n5].tolist()  # top 5 indices
        text = f"{', '.join(f'{names[j] if names else j} {pred_probs[j]:.2f}' for j in top5i)}, "
        annotator.text((32, 32), text, txt_color=(255, 255, 255))  # TODO: allow setting colors

    if keypoints is not None:
        for k in reversed(keypoints):
            annotator.kpts(k, self.orig_shape, kpt_line=kpt_line)

    return annotator.result()
