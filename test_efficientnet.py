import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.nn as nn
import timm
test_df = pd.read_csv('/Users/choetaewon/Documents/GitHub/bacon/data/open/sample_submission.csv')
CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 64,
    'SEED': 42,
    'WEIGHT_DECAY': 1e-4
}
class MultiViewDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.label_map = {'stable': 0, 'unstable': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]['id'])
        folder_path = os.path.join(self.root_dir, sample_id)

        views = []
        for name in ["front", "top"]:
            img_path = os.path.join(folder_path, f"{name}.png")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            views.append(image)

        if self.is_test:
            return views

        label = self.label_map[self.df.iloc[idx]['label']]
        return views, label

test_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_DS = MultiViewDataset(test_df, '/Users/choetaewon/Documents/GitHub/bacon/data/open/test', transform=test_transform, is_test=True)
test_DL =DataLoader(test_DS, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class Multiviewmodel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0'):
        super(Multiviewmodel, self).__init__()
        self.backbone_model = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        num_features = self.backbone_model.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features * 2, 1)
        )


    def forward(self, front_views, bottom_views):
        front_pred = self.backbone_model(front_views)
        bottom_pred = self.backbone_model(bottom_views)

        combined_pred = self.classifier(torch.cat((front_pred, bottom_pred), dim=1))

        return combined_pred


model = torch.load('/Users/choetaewon/Documents/GitHub/bacon/data/open/train_model.pt', map_location=device, weights_only=False)

model.eval()
all_probs = []


with torch.no_grad():
	for views in tqdm(test_DL, desc="Inference"):
		front_views = views[0].to(device)
		top_views = views[1].to(device)
		outputs = model(front_views, top_views).view(-1)
		probs = torch.sigmoid(outputs)
		all_probs.extend(probs.cpu().numpy())

all_probs = np.array(all_probs)
# 결과 저장 (컬럼  순서 중요)
submission = pd.DataFrame({
	'id': test_df['id'],
	'unstable_prob': all_probs,  # unstable일 확률 저장
	'stable_prob': 1.0 - all_probs  # stable일 확률 저장
})

submission.to_csv('submission.csv', encoding='UTF-8-sig', index=False)
print("submission.csv 저장 완료.")