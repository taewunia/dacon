import os
import pandas as pd
import numpy as np
import torchmetrics
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import timm
from torchmetrics import Accuracy, F1Score, MetricCollection

# 하이퍼파라미터 설정
CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-3,
    'BATCH_SIZE': 32,
    'SEED': 42
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

metrics = MetricCollection({
    "acc" : Accuracy(task="multiclass", num_classes=2),
    "f1" : F1Score(task="multiclass", num_classes=2, average="macro")
})
metrics = metrics.to(device)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG['SEED'])

train_df = pd.read_csv('/Users/choetaewon/Documents/GitHub/bacon/data/open/train.csv')
val_df = pd.read_csv('/Users/choetaewon/Documents/GitHub/bacon/data/open/dev.csv')

print(f"학습 데이터 개수: {len(train_df)}")
print(f"검증 데이터 개수: {len(val_df)}")

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


class Multiviewmodel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0'):
        super(Multiviewmodel, self).__init__()
        self.backbone_model = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        num_features = self.backbone_model.num_features

        self.classifier = nn.Linear(num_features * 2, 1)

    def forward(self, front_views, bottom_views):
        front_pred = self.backbone_model(front_views)
        bottom_pred = self.backbone_model(bottom_views)

        combined_pred = self.classifier(torch.cat((front_pred, bottom_pred), dim=1))

        return combined_pred


train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 1. 학습/검증 세트 준비 (is_test=False 설정)
train_dataset = MultiViewDataset(train_df, '/Users/choetaewon/Documents/GitHub/bacon/data/open/train', train_transform, is_test=False)
val_dataset = MultiViewDataset(val_df, '/Users/choetaewon/Documents/GitHub/bacon/data/open/dev', test_transform, is_test=False)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# 2. 테스트 세트 준비 (is_test=True 설정)
test_df = pd.read_csv('/Users/choetaewon/Documents/GitHub/bacon/data/open/sample_submission.csv')
test_dataset = MultiViewDataset(test_df, '/Users/choetaewon/Documents/GitHub/bacon/data/open/test', test_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)


def validate(model, loader, criterion, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for views, labels in tqdm(loader, desc="Validation"):
            views = [v.to(device) for v in views]
            labels = labels.to(device).float()

            outputs = model(views).view(-1)
            # 1. 시그모이드를 통과시켜 확률값(unstable일 확률)으로 변환
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs, dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.float64)


    eps = 1e-15
    p = np.clip(all_probs, eps, 1 - eps)
    # Binary Log Loss 공식 직접 적용
    logloss_score = -np.mean(all_labels * np.log(p) + (1 - all_labels) * np.log(1 - p))

    # Accuracy 계산
    acc_score = np.mean((all_probs > 0.5) == all_labels)

    return logloss_score, acc_score

model = Multiviewmodel()
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

train_loss_history = []
val_loss_history = []

for epoch in range(CFG['EPOCHS']):
    model.train()
    avg_train_loss = 0
    avg_val_loss = 0
    total_train_loss = 0
    total_val_loss = 0
    metrics.reset()
    train_process_bar = tqdm(train_loader, desc=f"Training {epoch + 1}/{CFG['EPOCHS']}", colour='green')
    for views, labels in train_process_bar:
        front_views = views[0].to(device)
        top_views = views[1].to(device)
        label_views = labels.float().to(device)

        optimizer.zero_grad()
        pred = model(front_views, top_views)
        pred = pred.view(-1)
        train_loss = criterion(pred, label_views)
        train_loss.backward()
        optimizer.step()
        train_process_bar.set_postfix(train_loss=train_loss.item())
        total_train_loss += train_loss.item()
        metrics.update(pred, label_views)
    metrics.compute()
    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    score_result = metrics.compute()
    print(f"loss={avg_train_loss:.3f} acc={score_result['acc'].item() * 100:.3f} f1={score_result['f1'].item() * 100:.3f}")
