# TODO: run with PYTHONUNBUFFERED=1
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import time
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from glove import Glove
from tokenizer import Tokenizer
from utils import build_datasets, count_parameters
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from imdb_model import IMDBModel
from jsd import JSD

NUM_EMBED_DIMS = 100
SEQ_LEN = 512
BATCH_SIZE = 64
LR = 5e-3
EPS = 1e-7
BETAS = (0.9, 0.999)
NUM_EPOCHS = 10
NUM_EPOCHS2 = 50

THETA = 0.45
ALPHA = 0.1
filename = "baseline_glove_cnn.pt"
filename2 = "baseline_glove_cnn_spans.pt"
SPAN_MODEL = True
NUM_SPANS = 4
FREEZE_WORD_EMBEDDINGS = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

glove = Glove()
glove_fname = f'glove.6B.{NUM_EMBED_DIMS}d.txt'
glove_path = os.path.join('models', 'glove', glove_fname)
glove.load_embeddings(glove_path, NUM_EMBED_DIMS)

tokenizer = Tokenizer(glove_embeds=glove)
M = torch.tensor(tokenizer.build_nn_embedding_matrix()).float().to(device)
print('embed matrix', M.shape)

num_spans = NUM_SPANS if SPAN_MODEL else 0
model = IMDBModel(embedding_matrix=M, num_spans=num_spans, freeze_word_embeddings=FREEZE_WORD_EMBEDDINGS)


train_val_dataset, test_dataset = build_datasets(tokenizer)

# train/validation split (stratified)
# y is third in tuple, it's a tensor of size [1,1]
all_labels = np.array([sample[2][0].item() for sample in train_val_dataset])
train_inds, val_inds, _, _ = train_test_split(
    range(len(train_val_dataset)),
    all_labels,
    stratify=all_labels,
    test_size=0.2,
    random_state=42
)
train_subset = Subset(train_val_dataset, train_inds)
val_subset = Subset(train_val_dataset, val_inds)

# data loaders
train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


print('f_pos', np.mean(all_labels))

# optimizer = optim.SGD(model.parameters(), lr=LR)
optimizer = optim.AdamW(model.parameters(), lr=LR, eps=EPS, betas=BETAS)
loss_fn = nn.BCEWithLogitsLoss()

num_params = count_parameters(model)
print(f"{num_params} trainable model params")
model.to(device)


def eval_epoch(loader, training=False, alpha=ALPHA):
    if training:
        model.train()
    else:
        model.eval()
    epoch_loss = 0
    epoch_bce = 0
    epoch_jsd = 0
    eval_y_preds = []
    eval_ys = []

    if training:
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)

    for (X_batch, mask_batch, y_batch) in loader:
        (X_batch, mask_batch, y_batch) = (X_batch.to(device), mask_batch.to(device), y_batch.to(device))
        optimizer.zero_grad()

        y_pred, r = model(X_batch, mask_batch)

        y_pred_p = torch.sigmoid(y_pred.detach()).squeeze(1).float()
        y_binary = (y_batch.detach() > 0.5).squeeze(1).int()
        eval_ys.append(y_binary.cpu().numpy())
        eval_y_preds.append(y_pred_p.cpu().numpy())

        bce_loss = loss_fn(y_pred, y_batch)
        if SPAN_MODEL:
            jsds = JSD(r, THETA)
            jsd_loss = torch.mean(jsds)
            epoch_jsd += jsd_loss.item()

            loss = (1.0-alpha)*bce_loss - alpha*jsd_loss
        else:
            loss = bce_loss
        if training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_bce += bce_loss.item()

    mean_loss = epoch_loss / len(loader)
    mean_bce = epoch_bce / len(loader)
    if SPAN_MODEL:
        mean_jsd = epoch_jsd / len(loader)
    else:
        mean_jsd = 0
    _ys = np.concatenate(eval_ys)
    _yhats = np.concatenate(eval_y_preds)
    roc_auc = roc_auc_score(_ys, _yhats)
    return mean_loss, mean_bce, mean_jsd, roc_auc


# Train the model initially without JSD loss
best_loss = float("inf")
best_roc = float("-inf")
epoch_times = []
for epoch in range(NUM_EPOCHS):
    """
    ramp_start = 0
    ramp_end = 100
    ramp_width = float(ramp_end - ramp_start)
    if epoch < ramp_start:
        alpha = 0.0
    elif epoch > ramp_end:
        alpha = ALPHA
    else:
        alpha = ((epoch-ramp_start)/ramp_width)*ALPHA
    """
    alpha = 0.0

    start = time.time()
    mean_train_loss,  mean_train_bce, mean_train_jsd, train_roc_auc = eval_epoch(train_loader, training=True, alpha=alpha)
    mean_val_loss, mean_val_bce, mean_val_jsd, val_roc_auc = eval_epoch(val_loader, training=False, alpha=alpha)
    stop = time.time()
    epoch_times.append(stop-start)
    best = val_roc_auc > best_roc

    print(f"{'* ' if best else '  '}Epoch {epoch+1:3d} | L* {mean_train_loss:.4f}, {mean_val_loss:.4f} | L1  {mean_train_bce:.4f}, {mean_val_bce:.4f} | L2 {mean_train_jsd: .4f}, {mean_val_jsd: .4f} | AUC {val_roc_auc:.3f} | alpha {alpha:.3f}")
    if epoch % 5 == 0:
        mean_epoch_time = np.mean(np.array(epoch_times))
        print(f"{mean_epoch_time:.2f} seconds per epoch average")
    if best:
        # best_loss = mean_val_loss
        best_roc = val_roc_auc
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_train_loss,
            'val_loss': mean_val_loss
        }
        torch.save(checkpoint, filename)

print('Done Training, loading best checkpoint')
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

mean_test_loss, mean_test_bce, mean_test_jsd, test_roc_auc = eval_epoch(test_loader, training=False)
print(f"  Test | L* {mean_test_loss:.4f} | L1 {mean_test_bce:.4f} | L2 {mean_test_jsd:.4f} | AUC {test_roc_auc:.3f}")


# Training round 2 - Spans
# Freezing embedding weights.
# TODO: I'm experimenting with freezing convolution weights too
for conv in model.convs:
    conv.requires_grad_(False)
num_params = count_parameters(model)
print(f"{num_params} trainable model params")


best_loss = float('inf')
epoch_times = []
ramp_start = 0
ramp_end = NUM_EPOCHS2 / 2
ramp_width = float(ramp_end - ramp_start)
for epoch in range(NUM_EPOCHS2):
    if epoch < ramp_start:
        alpha = 0.0
    elif epoch > ramp_end:
        alpha = ALPHA
    else:
        alpha = ((epoch-ramp_start)/ramp_width)*ALPHA

    start = time.time()
    mean_train_loss,  mean_train_bce, mean_train_jsd, train_roc_auc = eval_epoch(train_loader, training=True, alpha=alpha)
    mean_val_loss, mean_val_bce, mean_val_jsd, val_roc_auc = eval_epoch(val_loader, training=False, alpha=alpha)
    stop = time.time()
    epoch_times.append(stop - start)
    best = mean_val_loss < best_loss

    print(f"{'* ' if best else '  '}Epoch {epoch+1:3d} | L* {mean_train_loss:.4f}, {mean_val_loss:.4f} | L1  {mean_train_bce:.4f}, {mean_val_bce:.4f} | L2 {mean_train_jsd: .4f}, {mean_val_jsd: .4f} | AUC {val_roc_auc:.3f} | alpha {alpha:.3f}")
    if epoch % 5 == 0:
        mean_epoch_time = np.mean(np.array(epoch_times))
        print(f"{mean_epoch_time:.2f} seconds per epoch average")

    if best:
        best_loss = mean_val_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_train_loss,
            'val_loss': mean_val_loss
        }
        torch.save(checkpoint, filename2)

print('Done Training, loading best checkpoint')
checkpoint = torch.load(filename2)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

mean_test_loss, mean_test_bce, mean_test_jsd, test_roc_auc = eval_epoch(test_loader, training=False)
print(f"  Test | L* {mean_test_loss:.4f} | L1 {mean_test_bce:.4f} | L2 {mean_test_jsd:.4f} | AUC {test_roc_auc:.3f}")
