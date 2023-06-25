from copy import deepcopy

import numpy as np
from PIL.Image import Image
import cv2
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
        labels=True,
        boxes=True,
        alpha=0.5,
        states={},
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
    pred_boxes, show_boxes = self.boxes, boxes

    if pred_boxes and show_boxes:
        for d in reversed(pred_boxes):

            c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())

            name = ('' if id is None else f'{id}, ') + names[c] \
                + ('' if states.get(id, None) is None else f', {states[id]}')
            label = (f'{name} {conf:.2f}' if conf else name) if labels else None

            box = d.xyxy.squeeze()
            color = colors(c, True)


            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.orig_img, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
            if label:
                tf = max(line_width - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=font_size, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

                overlay = self.orig_img.copy()
                cv2.rectangle(overlay, p1, p2, color, -1, cv2.LINE_AA)  # filled
                self.orig_img = cv2.addWeighted(overlay, alpha, self.orig_img, 1 - alpha, 0)

                cv2.putText(self.orig_img,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            font_size,
                            (255, 255, 255),
                            thickness=tf,
                            lineType=cv2.LINE_AA)

    return np.asarray(self.orig_img)
