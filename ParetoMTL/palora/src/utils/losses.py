import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskCrossEntropyLoss(nn.modules.loss._Loss):
    def __init__(self, label_smoothing=0.0, fim_temperature=1.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.fim_temperature = fim_temperature
        if label_smoothing > 0:
            logging.info(f"MultiTaskCrossEntropyLoss: label_smoothing={label_smoothing}")
        if fim_temperature != 1.0:
            logging.info(f"MultiTaskCrossEntropyLoss: fim_temperature={fim_temperature}")

    def forward(self, input: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.loss_fn(input, target)

        loss = {k: self.loss_fn(input[k], target[k]) for k in input.keys()}

        # Temperature-scaled Fisher: compute tempered CE for FIM estimation
        fim_loss = None
        if self.fim_temperature != 1.0:
            fim_loss = {}
            for k in input.keys():
                # Divide logits by temperature before CE — softer predictions for Fisher
                tempered_logits = input[k] / self.fim_temperature
                fim_loss[k] = F.cross_entropy(tempered_logits, target[k])

        return loss, fim_loss


class MultiTaskMSELoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, input: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.loss_fn(input, target)

        return {k: self.loss_fn(input[k].squeeze(), target[k].squeeze()) for k in input.keys()}


class CityScapesLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn_semantic = SemanticSegmentationLoss()
        self.loss_fn_instance = InstanceSegmentationLoss()
        self.loss_fn_depth = DepthLoss()

    def forward(self, input, target) -> List[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.loss_fn(input, target)

        s = self.loss_fn_semantic(input[0], target[0])
        i = self.loss_fn_instance(input[1], target[1])
        d = self.loss_fn_depth(input[2], target[2])
        return [s, i, d]


class UTKFaceMultiTaskLoss(nn.modules.loss._Loss):
    def __init__(self, use_gender_task=True, label_smoothing=0.0,
                 fim_label_smoothing=0.0, adaptive_label_smoothing=0.0,
                 entropy_reg=0.0, elr_lambda=0.0, elr_beta=0.7,
                 fim_temperature=1.0):
        super().__init__()
        self.use_gender_task = use_gender_task
        self.huber = nn.HuberLoss(delta=1)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_fim = nn.CrossEntropyLoss(label_smoothing=fim_label_smoothing) if fim_label_smoothing > 0 else None
        self.adaptive_ls = adaptive_label_smoothing
        self.entropy_reg = entropy_reg
        self.elr_lambda = elr_lambda
        self.elr_beta = elr_beta
        self._current_lam = None  # set by trainer before forward
        self.fim_temperature = fim_temperature
        if elr_lambda > 0:
            logging.info(f"UTKFaceMultiTaskLoss: ELR lambda={elr_lambda}, beta={elr_beta}")
        if fim_temperature != 1.0:
            logging.info(f"UTKFaceMultiTaskLoss: fim_temperature={fim_temperature}")
        if label_smoothing > 0:
            logging.info(f"UTKFaceMultiTaskLoss: label_smoothing={label_smoothing}")
        if fim_label_smoothing > 0:
            logging.info(f"UTKFaceMultiTaskLoss: fim_label_smoothing={fim_label_smoothing}")
        if adaptive_label_smoothing > 0:
            logging.info(f"UTKFaceMultiTaskLoss: adaptive_label_smoothing={adaptive_label_smoothing}")
        if entropy_reg > 0:
            logging.info(f"UTKFaceMultiTaskLoss: entropy_reg={entropy_reg}")

    def forward(self, input, target) -> List[torch.Tensor]:
        lam = self._current_lam

        if self.adaptive_ls > 0 and lam is not None:
            # λ-adaptive smoothing: ε_t = ε_max · (1 - λ_t) for CE tasks
            # Task order matches UTKFaceDataModule/UtkFaceResnet: age(0), gender(1), race(2)
            eps_gender = self.adaptive_ls * (1 - lam[1].item())
            eps_race = self.adaptive_ls * (1 - lam[2].item())
            loss = {
                "age": self.huber(input["age"].squeeze(), target["age"].squeeze()),
                "gender": F.cross_entropy(input["gender"], target["gender"], label_smoothing=eps_gender),
                "race": F.cross_entropy(input["race"], target["race"], label_smoothing=eps_race),
            }
        else:
            loss = {
                "age": self.huber(input["age"].squeeze(), target["age"].squeeze()),
                "gender": self.ce(input["gender"], target["gender"]),
                "race": self.ce(input["race"], target["race"]),
            }

        # Entropy regularization (Option C): -beta * sum H(p_i) for CE tasks
        # Added OUTSIDE the per-task scalarization to preserve MAdam decomposition.
        # The entropy penalty is stored separately and added to total loss by the trainer.
        self._entropy_penalty = None
        if self.entropy_reg > 0:
            entropy_sum = 0.0
            for task_key in ["race", "gender"]:
                p = torch.softmax(input[task_key], dim=-1)
                # H(p) = -sum p_k log p_k. We want to MAXIMIZE entropy → subtract from loss.
                task_entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
                entropy_sum = entropy_sum + task_entropy
            # Store as penalty: trainer adds -beta * entropy_sum to total loss AFTER scalarization
            self._entropy_penalty = self.entropy_reg * entropy_sum

        # ELR-style regularization: log(1 - <p, t>) per CE task
        # t is an EMA of batch-level predictions (no per-sample tracking needed).
        # Keeps CE gradient alive without modifying targets. Preserves MAdam decomposition.
        self._elr_reg_value = None
        if self.elr_lambda > 0:
            elr_total = 0.0
            for task_key in ["race", "gender"]:
                p_grad = torch.softmax(input[task_key], dim=-1)
                p_detach = p_grad.detach()
                # Update EMA target with batch mean prediction
                if not hasattr(self, "_elr_ema"):
                    self._elr_ema = {}
                if task_key not in self._elr_ema:
                    self._elr_ema[task_key] = p_detach.mean(dim=0)
                else:
                    self._elr_ema[task_key] = (
                        self.elr_beta * self._elr_ema[task_key] + (1 - self.elr_beta) * p_detach.mean(dim=0)
                    )
                t = self._elr_ema[task_key].unsqueeze(0).expand_as(p_grad)
                # ELR loss: log(1 - <p, t>) — penalizes agreement with historical average
                inner = (p_grad * t.detach()).sum(dim=-1)
                elr_term = torch.log(1.0 - inner + 1e-7).mean()
                # Add directly to this task's loss (inside scalarization → preserves MAdam)
                loss[task_key] = loss[task_key] + self.elr_lambda * elr_term
                elr_total = elr_total + elr_term.item()
            self._elr_reg_value = self.elr_lambda * elr_total

        fim_loss = None
        if self.ce_fim is not None:
            fim_loss = {
                "age": loss["age"],
                "gender": self.ce_fim(input["gender"], target["gender"]),
                "race": self.ce_fim(input["race"], target["race"]),
            }
        elif self.fim_temperature != 1.0:
            fim_loss = {
                "age": loss["age"],
                "gender": F.cross_entropy(input["gender"] / self.fim_temperature, target["gender"]),
                "race": F.cross_entropy(input["race"] / self.fim_temperature, target["race"]),
            }

        return loss, fim_loss


class DepthLoss(nn.Module):
    """Loss for depth prediction. By default L1 loss is used."""

    def __init__(self, loss="l1", ignore_value=0.0):
        super(DepthLoss, self).__init__()
        self.ignore_value = ignore_value
        if loss == "l1":
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError("Loss {} currently not supported in DepthLoss".format(loss))

    def forward(self, out, label):
        if out.shape != label.shape:
            label = label.unsqueeze(1)
        mask = label != self.ignore_value
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class InstanceSegmentationLoss(nn.Module):
    """ """

    def __init__(self, loss="l1", ignore_value=250):
        super(InstanceSegmentationLoss, self).__init__()
        self.ignore_value = ignore_value
        if loss == "l1":
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError("Loss {} currently not supported in InstanceSegmentationLoss".format(loss))

    def forward(self, out, label):
        mask = label != self.ignore_value
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class SemanticSegmentationLoss(nn.CrossEntropyLoss):
    def __init__(self, size_average=True, ignore_index: int = 250, reduction: str = "mean") -> None:
        super().__init__(size_average=size_average, ignore_index=ignore_index, reduction=reduction)


class CityscapesTwoTaskLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def loss_fn_semantic(self, pred, target):
        target = target.long()
        return F.nll_loss(pred, target, ignore_index=-1)

    def loss_fn_depth(self, pred, target):
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(target, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = torch.sum(torch.abs(pred - target) * binary_mask)
        loss = loss / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return loss

    def forward(self, input, target) -> List[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.loss_fn(input, target)

        s = self.loss_fn_semantic(input["sem"], target["sem"])
        d = self.loss_fn_depth(input["depth"], target["depth"])

        # s = self.loss_fn_semantic(input[0], target[0])
        # d = self.loss_fn_depth(input[1], target[1])
        return {"sem": s, "depth": d}


class NYUv2Loss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def loss_fn_semantic(self, pred, target):
        target = target.long()
        return F.cross_entropy(pred, target, ignore_index=-1)

    def loss_fn_depth(self, pred, target):
        return self.loss_fn_normal(pred, target)

    def loss_fn_normal(self, pred, target):
        invalid_idx = 0  # make this -1 for task 'disp' (does not apply for NYUv2)
        valid_mask = (torch.sum(target, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        loss = torch.sum(F.l1_loss(pred, target, reduction="none").masked_select(valid_mask)) / torch.nonzero(
            valid_mask, as_tuple=False
        ).size(0)
        return loss

    def forward(self, input, target) -> List[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            return self.loss_fn(input, target)

        s = self.loss_fn_semantic(input["sem"], target["sem"])
        d = self.loss_fn_depth(input["depth"], target["depth"])
        n = self.loss_fn_normal(input["normal"], target["normal"])
        return {"sem": s, "depth": d, "normal": n}


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "sem":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(
            0
        )

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss
