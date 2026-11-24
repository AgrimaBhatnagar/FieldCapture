"""Pretrained ResNet18 damage classifier.

The backbone is ImageNet pretrained. A four class inspection head is trained from
an explicitly labelled manifest. No synthetic labels or random image splits.
"""
from __future__ import annotations
import json, os
from pathlib import Path

CLASSES = ["crack", "none", "stain", "standing_water"]

def build_model(pretrained=True):
    import torch
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model

def load_checkpoint(path, device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload

def predict_image(image, checkpoint, device=None):
    import cv2, numpy as np, torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint, device)
    bgr = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
    if bgr is None: raise ValueError("Could not read damage image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = ResNet18_Weights.DEFAULT.transforms()(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    order = np.argsort(probs)[::-1]
    return {"class": CLASSES[int(order[0])], "confidence": float(probs[order[0]]),
            "margin": float(probs[order[0]] - probs[order[1]]),
            "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
            "model": "resnet18_imagenet_transfer"}

def make_dataset(manifest_path):
    import cv2
    from PIL import Image
    rows = [json.loads(x) for x in Path(manifest_path).read_text().splitlines() if x.strip()]
    images, labels, groups = [], [], []
    for row in rows:
        path = Path(row["image"])
        if not path.is_absolute(): path = Path(manifest_path).parent / path
        im = cv2.imread(str(path))
        if im is None: raise ValueError(f"Unreadable image: {path}")
        images.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        labels.append(CLASSES.index(row["label"]))
        groups.append(str(row["property_id"]))
    return images, labels, groups

def evaluate_grouped(manifest_path, epochs=12, batch_size=8, seed=17):
    """Leave one property out. Requires multiple real property_id groups."""
    import numpy as np, torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
    images, labels, groups = make_dataset(manifest_path)
    if len(set(groups)) < 3: raise ValueError("Need at least 3 independently identified properties")
    if set(labels) != set(range(len(CLASSES))):
        raise ValueError("Manifest must contain all four damage classes")
    transforms = ResNet18_Weights.DEFAULT.transforms()
    class DS(Dataset):
        def __init__(self, ids, aug=False): self.ids=ids; self.aug=aug
        def __len__(self): return len(self.ids)
        def __getitem__(self, j):
            i=self.ids[j]; x=transforms(images[i])
            if self.aug and torch.rand(()) < .5: x=torch.flip(x, dims=[2])
            return x, int(labels[i])
    device="cuda" if torch.cuda.is_available() else "cpu"
    y_true=[]; y_pred=[]; fold_rows=[]
    splitter=LeaveOneGroupOut()
    for fold,(tr,te) in enumerate(splitter.split(np.zeros(len(labels)),labels,groups),1):
        if set(labels[i] for i in tr) != set(range(len(CLASSES))):
            raise ValueError(f"Fold {fold} training data must contain all four classes; add labelled examples from more properties")
        torch.manual_seed(seed+fold)
        model=build_model(pretrained=True).to(device)
        for p in model.parameters(): p.requires_grad=False
        for p in model.layer4.parameters(): p.requires_grad=True
        for p in model.fc.parameters(): p.requires_grad=True
        opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=2e-4,weight_decay=1e-3)
        loader=DataLoader(DS(tr,True),batch_size=batch_size,shuffle=True)
        loss_fn=nn.CrossEntropyLoss()
        for _ in range(epochs):
            model.train()
            for xb,yb in loader:
                xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
                loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
        model.eval(); preds=[]
        with torch.no_grad():
            for i in te:
                logits=model(transforms(images[i]).unsqueeze(0).to(device))
                preds.append(int(logits.argmax(1).item()))
        y_true.extend(labels[i] for i in te); y_pred.extend(preds)
        fold_rows.append({"held_out_property":str(groups[te[0]]),"n_test":len(te),
                          "accuracy":float(accuracy_score([labels[i] for i in te],preds))})
    return {"model":"ResNet18 ImageNet pretrained backbone, fine tuned layer4 and classifier",
            "split":"LeaveOneGroupOut by property_id","n":len(y_true),
            "accuracy":float(accuracy_score(y_true,y_pred)),
            "balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
            "classes":CLASSES,
            "classification_report":classification_report(y_true,y_pred,labels=list(range(4)),target_names=CLASSES,output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(y_true,y_pred,labels=list(range(4))).tolist(),
            "folds":fold_rows}

def train_final(manifest_path, output_path, epochs=12, batch_size=8):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights
    images, labels, groups=make_dataset(manifest_path)
    if len(set(groups)) < 3 or set(labels) != set(range(len(CLASSES))):
        raise ValueError("Final training requires at least 3 properties and all four damage classes")
    transforms=ResNet18_Weights.DEFAULT.transforms()
    class DS(Dataset):
        def __len__(self): return len(images)
        def __getitem__(self,i): return transforms(images[i]),int(labels[i])
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=build_model(pretrained=True).to(device)
    for p in model.parameters(): p.requires_grad=False
    for p in model.layer4.parameters(): p.requires_grad=True
    for p in model.fc.parameters(): p.requires_grad=True
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=2e-4,weight_decay=1e-3)
    loader=DataLoader(DS(),batch_size=batch_size,shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            loss=nn.CrossEntropyLoss()(model(xb),yb); loss.backward(); opt.step()
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.cpu().state_dict(),"classes":CLASSES,
                "architecture":"resnet18","pretrained":"ImageNet1K_V1",
                "training_samples":len(labels),"property_groups":len(set(groups))},output_path)
    return {"saved":str(output_path),"samples":len(labels),"property_groups":len(set(groups))}
