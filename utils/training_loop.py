import torch
import wandb
import torch.optim as optim
import torch.nn as nn
from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm


def calculate_metrics(loader, model, device, id_normal, num_classes):
    predictions, labels = [], []
    with torch.no_grad():
        for inputs, label, lengths in loader:
            logits = model(inputs.to(device), lengths)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu()
            predictions.append(preds)
            labels.append(label)
    predictions = torch.cat(predictions).to(dtype=torch.int64)
    labels = torch.cat(labels).to(dtype=torch.int64)
    normal_idx = [i for i in range(len(labels)) if labels[i] == id_normal]
    recall_normal = (predictions[normal_idx] == id_normal).sum() / len(normal_idx)
    return multiclass_f1_score(predictions, labels, average='weighted', num_classes=num_classes), recall_normal


def training_loop(model, model_name, device, train_loader, test_loader, id_normal=1, num_classes=7, 
                  optimizer=optim.Adam, lr=0.001, num_epochs=20):
    wandb.init(
        project="ml_sys_design",
        config={
            "model": model_name
        }
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optimizer(model.parameters(), lr=lr)
    num_epochs = num_epochs

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels, lengths) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        train_f1, train_recall_n = calculate_metrics(train_loader, model, device, id_normal, num_classes)
        test_f1, test_recall_n = calculate_metrics(test_loader, model, device, id_normal, num_classes)
        wandb.log({
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "train_recall_n": train_recall_n,
            "train_f1": train_f1,
            "test_recall_n": test_recall_n,
            "test_f1": test_f1,
        })