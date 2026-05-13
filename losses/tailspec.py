import torch
import torch.nn as nn
import torch.nn.functional as F


class TailSpecLoss(nn.Module):
    def __init__(
        self,
        cls_num_list=None,
        num_classes=100,
        reweight_epoch=0,
        max_m=0.1,
        s=70.0,
        temperature=0.1,
        weight=3.25,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.reweight_epoch = int(reweight_epoch)
        self.max_m = float(max_m)
        self.s = float(s)
        self.temperature = float(temperature)
        self.weight = float(weight) if weight is not None else 1.0

        if cls_num_list is None:
            cls_num = torch.ones(self.num_classes, dtype=torch.float32)
        else:
            cls_num = torch.tensor(cls_num_list, dtype=torch.float32)
            if cls_num.numel() < self.num_classes:
                pad = torch.ones(self.num_classes - cls_num.numel(), dtype=torch.float32)
                cls_num = torch.cat([cls_num, pad])
            elif cls_num.numel() > self.num_classes:
                cls_num = cls_num[:self.num_classes]
        cls_num = cls_num + 1e-6

        freq_min = cls_num.min()
        freq_max = cls_num.max()
        if freq_max - freq_min > 0:
            freq_norm = (cls_num - freq_min) / (freq_max - freq_min)
        else:
            freq_norm = torch.zeros_like(cls_num)
        m_list = self.max_m * (1.0 - freq_norm)

        tmp = torch.log(cls_num + 1.0)
        tmp = tmp - tmp.min()
        tmp = tmp / (tmp.max() + 1e-8)
        s_list = tmp * (self.s - 1.0) + 1.0

        self.register_buffer("cls_num", cls_num)
        self.register_buffer("s_list", s_list)
        self.register_buffer("m_list", m_list)

    def forward(self, logits, targets, epoch=None, reduction="mean", features=None, centers=None):
        if epoch is None:
            epoch = 0
        device = logits.device
        labels = targets.long().view(-1)
        n, c = logits.size()

        cosine = logits * self.s_list.to(device)
        phi = cosine - self.m_list.to(device)
        one_hot = torch.zeros((n, c), dtype=logits.dtype, device=device)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = torch.where(one_hot > 0, phi, cosine)

        if self.temperature is not None and self.temperature > 0:
            output = output / self.temperature

        cls_w = 1.0 / (self.cls_num + 1e-6)
        cls_w = cls_w / (cls_w.max() + 1e-8)
        cls_w = cls_w * (1.6 * self.weight)
        cls_w = torch.clamp(cls_w, max=2.0).to(device)

        ce_loss = F.cross_entropy(output, labels, weight=cls_w, reduction="none")

        freq_min = self.cls_num.min()
        freq_max = self.cls_num.max()
        freq_norm = (self.cls_num - freq_min) / ((freq_max - freq_min) + 1e-8)
        is_tail = freq_norm.to(device)[labels] <= 0.25
        sample_weights = torch.ones_like(ce_loss)
        sample_weights[is_tail] = 1.0 + 0.15 * self.weight
        if epoch is not None and epoch >= self.reweight_epoch:
            sample_weights[is_tail] = 1.0 + 0.25 * self.weight

        loss = ce_loss * sample_weights

        if features is not None and centers is not None:
            feats = features.to(device)
            cts = centers.to(device)
            center_loss = ((feats - cts[labels]) ** 2).sum(dim=1)
            loss = loss + 0.003 * center_loss

        if reduction == "none":
            return loss
        return loss.mean()

