"""
    Scrypt for classifier model inference
"""
import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import os
import torchvision
resize_to = 224
class Padding(torch.nn.Module):
    def __init__(self, min_height, min_width):
        super().__init__()
        self.min_height = min_height
        self.min_width = min_width

    def forward(self, images):
        if images.shape[1] < self.min_height:
            images = torchvision.transforms.functional.pad(images, (0,0,0,(self.min_height-images.shape[1])))
        if images.shape[2] < self.min_width:
            images = torchvision.transforms.functional.pad(images, (0,0,self.min_width - images.shape[2], 0))
        return images

test_transform = Compose([
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    #RandomShortSideScale(min_size=100, max_size=128),
                    RandomHorizontalFlip(p=0.5),
                    RandomPerspective(distortion_scale=0.35, p=0.7),
                    Padding(min_height=200, min_width=200),
                    Resize((resize_to, resize_to))
                ]
)


class Classifier():
    def __init__(self, path, device='cpu') -> None:
        '''
        :param path: путь до сохраненной модели
        :param clip_duration:
        :param device:
        '''
        self.model = mvit_v2_s()
        self.model.head[1] = torch.nn.Linear(768, 2)
        self.model.eval()
        self.device = device
        self.transforms = test_transform
        self.checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.idx2class = {0: 'off', 1: 'on'}
        self.class2idx = {'off': 0, 'on': 1}

    def predict(self, video: torch.FloatTensor):
        '''
        :param video: должно быть формата [T C H W]
        :return: (pred idx, pred class)
        '''
        # full_video_tensor = handle_video_path(video_path)
        video = video.to(self.device)
        video = self.transforms(video)
        pred = torch.argmax(torch.nn.functional.softmax(self.model(video.unsqueeze(0)), dim=-1)).item()
        return pred, self.idx2class[int(pred)]

    # def handle_video_path(self, video_path: str) -> torch.FloatTensor:
        # video = EncodedVideo.from_path(video_path)
        # video_tensor = video.get_clip(0.0, 0.0 + self.clip_duration)['video'].permute(1, 0, 2, 3)
        # video_tensor = self.transforms(video_tensor)
        # return video_tensor

