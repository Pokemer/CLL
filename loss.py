import torch
import torch.nn as nn
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class CLL(nn.Module):
  def __init__(self, sigma=1, epoch=15, num_classes = 10, reduction='mean'):
    super().__init__()
    self.ce = nn.CrossEntropyLoss(reduction=reduction)
    self.sigma=sigma
    self.epoch=epoch
    self.num_classes=num_classes
    self.reduction = reduction
  def forward(self,y_hat, y, epoch=None):
    if flag:=(epoch is None or self.epoch > epoch):
      loss = self.ce(y_hat, y)
    else:
      y_hat = F.softmax(y_hat, dim=1)
      loss = 1-torch.exp(-(1-y_hat[range(y_hat.size(0)),y])**2/2/self.sigma**2)
    if self.reduction=='mean':
      loss = torch.mean(loss)
    elif self.reduction=='sum':
      return torch.sum(loss)
    if flag:
      return loss
    else:
      return loss/(1-torch.exp(torch.tensor(-1/self.sigma**2)))
  def updeate_sigma(self, factor=0.5):
    self.sigma*=factor
    return