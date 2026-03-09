import torch
import torch.nn as nn

class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def train(model, train_loader, train_N, random_trans, optimizer, loss_function, device):
    total_loss = 0.0
    correct = 0
    seen = 0

    model.train()
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if random_trans is not None:
            x = random_trans(x)

        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += batch_loss.item() * bs
        correct += (output.argmax(1) == y).sum().item()
        seen += bs

    print(f"Train - Loss: {total_loss/seen:.4f} Accuracy: {correct/seen:.4f}")

def validate(model, valid_loader, loss_function, device):
    total_loss = 0.0
    correct = 0
    seen = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            output = model(x)
            bs = y.size(0)
            total_loss += loss_function(output, y).item() * bs
            correct += (output.argmax(1) == y).sum().item()
            seen += bs

    print(f"Valid - Loss: {total_loss/seen:.4f} Accuracy: {correct/seen:.4f}")