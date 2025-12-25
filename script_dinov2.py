"""DINOv2 version - Logo retrieval with batch-hard triplet mining"""
"""
Usage:
python script_dinov2.py
"""

import os
import random
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================
# Config
# =========================
class Config:
    BASE_DIR = "FlickrLogos-v2"
    CLASSES_DIR = os.path.join(BASE_DIR, "classes", "jpg")

    # DINOv2 models: vits14=384, vitb14=768, vitl14=1024, vitg14=1536
    DINOV2_MODEL = "dinov2_vits14"  # Small model - fastest
    EMBEDDING_DIM = 384  # Output dimension for vits14
    
    MARGIN = 0.4
    LR = 3e-5
    EPOCHS = 80
    BATCH_SIZE = 32
    EARLY_STOPPING = 20

    USE_BOUNDING_BOXES = True
    BBOX_PADDING = 10
    MIN_BBOX_SIZE = 20

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Dataset utilities
# =========================
def get_all_classes(classes_dir):
    return sorted([
        d for d in os.listdir(classes_dir)
        if os.path.isdir(os.path.join(classes_dir, d)) and d != "no-logo"
    ])

def get_partition_file(base_dir, partition):
    return {
        "P1": "trainset.spaces.txt",
        "P2": "valset-logosonly.spaces.txt",
        "P3": "testset-logosonly.spaces.txt"
    }[partition]

def get_class_images_from_partition(classes_dir, class_name, partition, base_dir):
    file_path = os.path.join(base_dir, get_partition_file(base_dir, partition))
    images = []

    with open(file_path) as f:
        for line in f:
            c, fname = line.strip().split()
            if c == class_name:
                path = os.path.join(classes_dir, class_name, fname)
                if os.path.exists(path):
                    images.append(path)
    return images

def get_bbox_from_file(bbox_path, padding):
    if not os.path.exists(bbox_path):
        return None
    with open(bbox_path) as f:
        x, y, w, h = map(int, f.readline().split()[1:])
    return max(0, x-padding), max(0, y-padding), w+2*padding, h+2*padding

def crop_logo_from_image(image_path, padding, min_size):
    img = Image.open(image_path).convert("RGB")
    bbox = get_bbox_from_file(image_path + ".bboxes.txt", padding)
    if bbox is None:
        return img
    x, y, w, h = bbox
    if w < min_size or h < min_size:
        return img
    return img.crop((x, y, x+w, y+h))

# =========================
# Dataset
# =========================
class LogoDataset(Dataset):
    def __init__(self, class_images: Dict[str, List[str]], transform, cfg):
        self.samples = []
        self.transform = transform
        self.cfg = cfg
        self.class_to_idx = {c: i for i, c in enumerate(class_images)}

        for c, imgs in class_images.items():
            label = self.class_to_idx[c]
            for p in imgs:
                self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = crop_logo_from_image(
            path, self.cfg.BBOX_PADDING, self.cfg.MIN_BBOX_SIZE
        )
        return self.transform(img), label

# =========================
# Model
# =========================
class DINOv2Embedding(nn.Module):
    def __init__(self, model_name, embedding_dim):
        super().__init__()
        # Load DINOv2 model from torch hub
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        
        self.head = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # DINOv2 outputs [B, embed_dim] directly (uses CLS token)
        x = self.backbone(x)
        x = self.head(x)
        return nn.functional.normalize(x, dim=1)

# =========================
# Batch-Hard Triplet Loss
# =========================
class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings)
        N = labels.size(0)

        mask_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask_neg = ~mask_pos

        dist_pos = dist.clone()
        dist_pos[~mask_pos] = -1e9
        hardest_pos = dist_pos.max(dim=1)[0]

        dist_neg = dist.clone()
        dist_neg[~mask_neg] = 1e9
        hardest_neg = dist_neg.min(dim=1)[0]

        loss = torch.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()

# =========================
# Embedding helper
# =========================
@torch.no_grad()
def embed_dataset(model, class_images, transform, cfg):
    embs, labels = [], []
    model.eval()

    for c, imgs in class_images.items():
        for p in imgs:
            img = crop_logo_from_image(p, cfg.BBOX_PADDING, cfg.MIN_BBOX_SIZE)
            x = transform(img).unsqueeze(0).to(cfg.DEVICE)
            e = model(x).cpu()
            embs.append(e)
            labels.append(c)

    return torch.cat(embs), labels

# =========================
# Retrieval metrics
# =========================
def recall_at_k(q_emb, q_lab, g_emb, g_lab, Ks=(1,5,10)):
    dist = torch.cdist(q_emb, g_emb)
    idx = dist.argsort(dim=1)

    out = {k: 0 for k in Ks}
    for i in range(len(q_lab)):
        ranked = [g_lab[j] for j in idx[i]]
        for k in Ks:
            if q_lab[i] in ranked[:k]:
                out[k] += 1

    return {k: out[k] / len(q_lab) for k in Ks}

def precision_at_k(q_emb, q_lab, g_emb, g_lab, Ks=(1,10,30)):
    dist = torch.cdist(q_emb, g_emb)
    idx = dist.argsort(dim=1)

    out = {k: 0 for k in Ks}
    for i in range(len(q_lab)):
        ranked = [g_lab[j] for j in idx[i]]
        for k in Ks:
            out[k] += ranked[:k].count(q_lab[i]) / k
    return {k: out[k] / len(q_lab) for k in Ks}

# =========================
# Main
# =========================
def main():
    cfg = Config()

    classes = get_all_classes(cfg.CLASSES_DIR)
    P1 = {c: get_class_images_from_partition(cfg.CLASSES_DIR, c, "P1", cfg.BASE_DIR) for c in classes}
    P2 = {c: get_class_images_from_partition(cfg.CLASSES_DIR, c, "P2", cfg.BASE_DIR) for c in classes}

    # DINOv2 uses specific image preprocessing
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = LogoDataset(P1, transform_train, cfg)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    print(f"Loading DINOv2 model: {cfg.DINOV2_MODEL}...")
    model = DINOv2Embedding(cfg.DINOV2_MODEL, cfg.EMBEDDING_DIM).to(cfg.DEVICE)
    print("✓ Model loaded successfully")
    
    loss_fn = BatchHardTripletLoss(cfg.MARGIN)
    opt = optim.Adam(model.parameters(), lr=cfg.LR)

    # History
    history = {'train_loss': [], 'score': [], 'recall': {1: [], 5: [], 10: []}, 'precision': {1: [], 10: [], 30: []}}

    best_score = -np.inf
    early_stop_counter = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in loader:
            imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            emb = model(imgs)
            loss = loss_fn(emb, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history['train_loss'].append(avg_loss)
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_loss:.4f}")

        # Embed datasets for evaluation
        q_emb, q_lab = embed_dataset(model, P1, transform_test, cfg)
        g_emb, g_lab = embed_dataset(model, P2, transform_test, cfg)

        rec = recall_at_k(q_emb, q_lab, g_emb, g_lab)
        prec = precision_at_k(q_emb, q_lab, g_emb, g_lab)

        history['recall'][1].append(rec[1])
        history['recall'][5].append(rec[5])
        history['recall'][10].append(rec[10])
        history['precision'][1].append(prec[1])
        history['precision'][10].append(prec[10])
        history['precision'][30].append(prec[30])

        score = rec[1]+rec[5]+rec[10] + prec[1]+prec[10]+prec[30]
        history['score'].append(score)
        print(f"  Score: {score:.4f} | Recall@1/5/10: {rec} | Precision@1/10/30: {prec}")

        # Save best model
        if score > best_score:
            best_score = score
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_logo_retrieval_dinov2.pth")
            print("✓ Saved best model.")
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= cfg.EARLY_STOPPING:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            break

    # =========================
    # Plotting
    # =========================
    epochs = range(1, len(history['train_loss'])+1)

    # Train loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss (DINOv2)'); plt.grid(True)
    plt.legend(); plt.savefig('train_loss_dinov2.png', dpi=300)

    # Score
    plt.figure()
    plt.plot(epochs, history['score'], 'g-', label='Score', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Combined Score (DINOv2)'); plt.grid(True)
    plt.legend(); plt.savefig('score_dinov2.png', dpi=300)

    # Precision
    plt.figure()
    plt.plot(epochs, history['precision'][1], 'r-', label='P@1', linewidth=2)
    plt.plot(epochs, history['precision'][10], 'b-', label='P@10', linewidth=2)
    plt.plot(epochs, history['precision'][30], 'g-', label='P@30', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Precision'); plt.title('Precision@K (DINOv2)'); plt.grid(True)
    plt.legend(); plt.savefig('precision_dinov2.png', dpi=300)

    # Recall
    plt.figure()
    plt.plot(epochs, history['recall'][1], 'r-', label='R@1', linewidth=2)
    plt.plot(epochs, history['recall'][5], 'b-', label='R@5', linewidth=2)
    plt.plot(epochs, history['recall'][10], 'g-', label='R@10', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Recall'); plt.title('Recall@K (DINOv2)'); plt.grid(True)
    plt.legend(); plt.savefig('recall_dinov2.png', dpi=300)

    # Save history JSON
    with open('training_history_dinov2.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("✓ Training complete. Plots and history saved.")

if __name__ == "__main__":
    main()
