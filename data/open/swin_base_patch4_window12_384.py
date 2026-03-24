import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm
from torchmetrics import Accuracy, F1Score, MetricCollection
import matplotlib.pyplot as plt
import cv2


BASE_PATH = '/Users/choetaewon/Documents/GitHub/bacon/data/open'


CFG = {
    'IMG_SIZE': 384,
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 4,
    'SEED': 42,
    'WEIGHT_DECAY': 1e-2
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"현재 딥러닝 장비: {device}")

metrics = MetricCollection({
    "acc" : Accuracy(task="binary"),
    "f1" : F1Score(task="binary")
})
metrics = metrics.to(device)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG['SEED'])

train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
val_df = pd.read_csv(os.path.join(BASE_PATH, 'dev.csv'))

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

i = 0

class Multiviewmodel(nn.Module):
    def __init__(self, backbone_name='swin_base_patch4_window12_384'):
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

train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=3, translate=(0.05, 0.05)),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MultiViewDataset(train_df, os.path.join(BASE_PATH, 'train'), train_transform, is_test=False)
val_dataset = MultiViewDataset(val_df, os.path.join(BASE_PATH, 'dev'), test_transform, is_test=False)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

test_df = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
test_dataset = MultiViewDataset(test_df, os.path.join(BASE_PATH, 'test'), test_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

model = Multiviewmodel()
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=CFG['WEIGHT_DECAY'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])

train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []

model.eval()
for epoch in range(CFG['EPOCHS']):
    model.train()
    avg_train_loss = 0
    total_train_loss = 0
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
        metrics.update(torch.sigmoid(pred), label_views)

    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    train_score_result = metrics.compute()
    scheduler.step()

    val_process_bar = tqdm(val_loader, desc=f"Validate {epoch + 1}/{CFG['EPOCHS']}", colour='red')
    metrics.reset()
    model.eval()
    total_val_loss = 0
    for views, labels in val_process_bar:
        front_views = views[0].to(device)
        top_views = views[1].to(device)
        label_views = labels.float().to(device)

        with torch.no_grad():
            pred = model(front_views, top_views)
            pred = pred.view(-1)
            val_loss = criterion(pred, label_views)
            val_process_bar.set_postfix(val_loss=val_loss.item())
            total_val_loss += val_loss.item()
            metrics.update(torch.sigmoid(pred), label_views)

    avg_val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    val_score_result = metrics.compute()

    train_f1_history.append(train_score_result['f1'].item())
    val_f1_history.append(val_score_result['f1'].item())
    print(f"\ntrain_loss={avg_train_loss:.4f} train_acc={train_score_result['acc'].item() * 100:.2f} train_f1={train_score_result['f1'].item() * 100:.2f}\nval_loss={avg_val_loss:.4f} val_acc={val_score_result['acc'].item() * 100:.2f} val_f1={val_score_result['f1'].item() * 100:.2f}")
    if val_score_result['f1'].item() > 0.8 and avg_val_loss <= 0.33:
      i += 1
      print(f"early stopping {val_score_result['f1'].item()*100:.2f} {avg_val_loss:.2f}")
      model.to("cpu")
      torch.save(model, os.path.join(BASE_PATH, f'train_model_eff{i}.pt'))
      model.to("mps")

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='train loss', color='green')
plt.title("avg_train_loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='val loss', color='red')
plt.title("avg_val_loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1, len(train_f1_history) + 1), train_f1_history, label='train f1', color='green')
plt.title("train_f1")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, len(val_f1_history) + 1), val_f1_history, label='val f1', color='red')
plt.title("val_f1")
plt.legend()

plt.tight_layout()
plt.show()