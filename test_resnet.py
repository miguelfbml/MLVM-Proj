#!/usr/bin/env python3
"""

working version for script4.py
test_final.py
Test trained retrieval model vs ImageNet ResNet-50 baseline.
Uses P1 as prototypes and evaluates on P2 or P3.
"""

import os
import json
import argparse
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# Config
# =========================
class Config:
    BASE_DIR = "FlickrLogos-v2"
    CLASSES_DIR = os.path.join(BASE_DIR, "classes", "jpg")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "best_logo_retrieval.pth"
    EMBEDDING_DIM = 2048
    BBOX_PADDING = 10
    MIN_BBOX_SIZE = 20

# =========================
# Dataset utilities
# =========================
def get_all_classes(classes_dir):
    return sorted([
        d for d in os.listdir(classes_dir)
        if os.path.isdir(os.path.join(classes_dir, d)) and d != "no-logo"
    ])

def get_partition_file(partition):
    return {
        "P1": "trainset.spaces.txt",
        "P2": "valset-logosonly.spaces.txt",
        "P3": "testset-logosonly.spaces.txt"
    }[partition]

def get_class_images(classes_dir, class_name, partition):
    file_path = os.path.join(Config.BASE_DIR, get_partition_file(partition))
    images = []
    with open(file_path) as f:
        for line in f:
            c, fname = line.strip().split()
            if c == class_name:
                p = os.path.join(classes_dir, class_name, fname)
                if os.path.exists(p):
                    images.append(p)
    return images

def parse_bbox(bbox_file):
    if not os.path.exists(bbox_file):
        return None
    with open(bbox_file) as f:
        parts = f.readline().split()
        return tuple(map(int, parts[1:5]))

def crop_logo(img_path):
    img = Image.open(img_path).convert("RGB")
    bbox = parse_bbox(img_path + ".bboxes.txt")
    if bbox is None:
        return img
    x, y, w, h = bbox
    if w < Config.MIN_BBOX_SIZE or h < Config.MIN_BBOX_SIZE:
        return img
    x = max(0, x - Config.BBOX_PADDING)
    y = max(0, y - Config.BBOX_PADDING)
    return img.crop((x, y, x + w + 2*Config.BBOX_PADDING, y + h + 2*Config.BBOX_PADDING))

# =========================
# Models
# =========================
class TrainedEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(Config.EMBEDDING_DIM)
        )

    def forward(self, x):
        x = self.head(self.backbone(x))
        return nn.functional.normalize(x, dim=1)

class ImageNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        x = torch.flatten(self.backbone(x), 1)
        return nn.functional.normalize(x, dim=1)

# =========================
# Embedding helpers
# =========================
@torch.no_grad()
def compute_embeddings(model, paths, transform):
    embs = []
    for p in paths:
        img = transform(crop_logo(p)).unsqueeze(0).to(Config.DEVICE)
        embs.append(model(img).cpu())
    return torch.cat(embs)

def build_prototypes(model, classes, transform):
    proto, labels = [], []
    for c in classes:
        imgs = get_class_images(Config.CLASSES_DIR, c, "P1")
        if len(imgs) == 0:
            continue
        emb = compute_embeddings(model, imgs, transform)
        proto.append(emb.mean(dim=0, keepdim=True))
        labels.append(c)
    return torch.cat(proto), labels

# =========================
# Metrics
# =========================
def retrieval_metrics(q_emb, q_lab, g_emb, g_lab):
    dist = torch.cdist(q_emb, g_emb)
    idx = dist.argsort(dim=1)

    recall_K = {1:0, 5:0, 10:0}
    precision_K = {1:0, 10:0, 30:0}
    n = len(q_lab)
    
    # Calculate mAP
    average_precisions = []

    for i in range(n):
        ranked = [g_lab[j] for j in idx[i]]
        
        # Recall@K
        for k in recall_K:
            recall_K[k] += q_lab[i] in ranked[:k]
        
        # Precision@K
        for k in precision_K:
            precision_K[k] += ranked[:k].count(q_lab[i]) / k
        
        # Average Precision for this query
        relevant_positions = [j+1 for j, label in enumerate(ranked) if label == q_lab[i]]
        if relevant_positions:
            precisions_at_relevant = [sum(1 for r in relevant_positions if r <= pos) / pos 
                                     for pos in relevant_positions]
            average_precisions.append(sum(precisions_at_relevant) / len(relevant_positions))
        else:
            average_precisions.append(0.0)

    recall_K = {k: v/n for k,v in recall_K.items()}
    precision_K = {k: v/n for k,v in precision_K.items()}
    mAP = sum(average_precisions) / len(average_precisions)
    
    return recall_K, precision_K, mAP

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["P2", "P3"], default="P2")
    parser.add_argument("--model", default=Config.MODEL_PATH)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    classes = get_all_classes(Config.CLASSES_DIR)

    # Load models
    trained = TrainedEmbeddingModel().to(Config.DEVICE)
    trained.load_state_dict(torch.load(args.model, map_location=Config.DEVICE))
    trained.eval()

    baseline = ImageNetBaseline().to(Config.DEVICE)
    baseline.eval()

    # Prototypes
    trained_proto, labels = build_prototypes(trained, classes, transform)
    base_proto, _ = build_prototypes(baseline, classes, transform)

    # Queries
    q_imgs, q_labels = [], []
    for c in classes:
        imgs = get_class_images(Config.CLASSES_DIR, c, args.partition)
        q_imgs += imgs
        q_labels += [c]*len(imgs)

    trained_q = compute_embeddings(trained, q_imgs, transform)
    base_q = compute_embeddings(baseline, q_imgs, transform)

    trained_rec, trained_prec, trained_map = retrieval_metrics(trained_q, q_labels, trained_proto, labels)
    base_rec, base_prec, base_map = retrieval_metrics(base_q, q_labels, base_proto, labels)

    # Save JSON
    results = {
        "partition": args.partition,
        "trained": {"recall": trained_rec, "precision": trained_prec, "mAP": trained_map},
        "baseline": {"recall": base_rec, "precision": base_prec, "mAP": base_map}
    }
    with open(f"results_{args.partition}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print(f"\n{'='*80}")
    print(f"Evaluation Results on {args.partition}")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<20} {'Trained':<15} {'Baseline':<15} {'Improvement':<15}")
    print(f"{'-'*65}")
    
    # Recall metrics
    for k in sorted(trained_rec.keys()):
        trained_val = trained_rec[k]
        base_val = base_rec[k]
        improvement = trained_val - base_val
        print(f"Recall@{k:<14} {trained_val:<15.4f} {base_val:<15.4f} {improvement:+.4f}")
    
    print()
    
    # Precision metrics
    for k in sorted(trained_prec.keys()):
        trained_val = trained_prec[k]
        base_val = base_prec[k]
        improvement = trained_val - base_val
        print(f"Precision@{k:<11} {trained_val:<15.4f} {base_val:<15.4f} {improvement:+.4f}")
    
    print()
    
    # mAP
    improvement = trained_map - base_map
    print(f"{'mAP':<20} {trained_map:<15.4f} {base_map:<15.4f} {improvement:+.4f}")
    
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: results_{args.partition}.json")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
