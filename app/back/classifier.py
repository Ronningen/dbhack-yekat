"""
    Scrypt for classifier model inference
"""
import torch
import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


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
        self.transforms = MViT_V2_S_Weights.KINETICS400_V1.transforms()
        self.checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        pred = torch.argmax(self.model(video.unsqueeze(0)))
        return pred, idx2class[pred]

    # def handle_video_path(self, video_path: str) -> torch.FloatTensor:
        # video = EncodedVideo.from_path(video_path)
        # video_tensor = video.get_clip(0.0, 0.0 + self.clip_duration)['video'].permute(1, 0, 2, 3)
        # video_tensor = self.transforms(video_tensor)
        # return video_tensor

