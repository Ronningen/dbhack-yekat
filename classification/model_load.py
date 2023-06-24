import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

model = mvit_v2_s()
model.eval()
transforms = MViT_V2_S_Weights.KINETICS400_V1.transforms()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.head[1] = torch.nn.Linear(768, 2)

model.load_state_dict(torch.load(r'C:\Users\user\Downloads\model_class (2)', map_location=device))  # path to model state_dict
model.eval()
print(model)
#%%
