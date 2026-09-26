import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-4


class PaLoRALayer:
    is_palora_layer = True

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        num_members: int,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.num_members = num_members
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class PaLinear(nn.Linear, PaLoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_members: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        PaLoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            num_members=num_members,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.ParameterDict(
                {str(i): nn.Parameter(self.weight.new_zeros((r, in_features))) for i in range(num_members)}
            )
            self.lora_B = nn.ParameterDict(
                {str(i): nn.Parameter(self.weight.new_zeros((out_features, r))) for i in range(num_members)}
            )
            self.scaling = self.lora_alpha / self.r

            # Compared to the original implementation, we DO NOT freeze the weight matrix of the backbone
            self.weight.requires_grad = True
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in self.lora_A.keys():
                nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[i])
                # nn.init.ones_(self.lora_B[i])

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor, ray: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # assert ray.shape[0] == self.num_members

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            x = self.lora_dropout(x)
            # not the most efficient implementation. this is a linear operation, can take mult x out
            skip = [
                x @ self.lora_A[k].transpose(0, 1) @ self.lora_B[k].transpose(0, 1)
                for i, k in enumerate(self.lora_A.keys())
            ]

            result = result + sum([a * s for a, s in zip(ray, skip)]) * self.scaling
            # result = result + sum([a.unsqueeze(1) * s for a, s in zip(ray.T, skip)]) * self.scaling
            return result
        else:
            raise NotImplementedError("go over this again")
            return F.linear(x, T(self.weight), bias=self.bias)

    @classmethod
    def from_module(
        cls,
        module,
        num_members,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=False,
    ):
        return cls(
            in_features=module.in_features,
            out_features=module.out_features,
            num_members=num_members,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            bias=module.bias is not None,
        )


class PaConvLoRA(nn.Module, PaLoRALayer):
    def __init__(
        self,
        conv_module,
        in_channels,
        out_channels,
        kernel_size,
        num_members: int,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=True,
        **kwargs,
    ):
        super(PaConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        PaLoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            num_members=num_members,
        )
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.ParameterDict(
                {
                    str(i): nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
                    for i in range(num_members)
                }
            )
            self.lora_B = nn.ParameterDict(
                {
                    str(i): nn.Parameter(
                        self.conv.weight.new_zeros(
                            (
                                out_channels // self.conv.groups * kernel_size,
                                r * kernel_size,
                            )
                        )
                    )
                    for i in range(num_members)
                }
            )

            self.scaling = self.lora_alpha / self.r

            # Compared to the original implementation, we DO NOT freeze the weight matrix of the backbone
            self.conv.weight.requires_grad = True
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in self.lora_A.keys():
                nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[i])
                # nn.init.ones_(self.lora_B[i])

    def train(self, mode=True):
        super(PaConvLoRA, self).train(mode)

    def forward(self, x: torch.Tensor, ray: torch.Tensor):

        assert ray.shape[0] == self.num_members, f"Ray shape {ray.shape} does not match num_members {self.num_members}"

        if self.r > 0 and not self.merged:
            skip = [a * lb @ la for a, lb, la in zip(ray, self.lora_B.values(), self.lora_A.values())]
            skip = sum(skip).view(self.conv.weight.shape) * self.scaling

            return self.conv._conv_forward(x, self.conv.weight + skip, self.conv.bias)
        return self.conv(x)

    @classmethod
    def from_module(
        cls,
        module: nn.Conv1d | nn.Conv2d | nn.Conv3d,
        num_members,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=False,
    ):
        return cls(
            # conv_module=module.__class__,
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=(module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]),
            num_members=num_members,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
        )


class PaConv2d(PaConvLoRA):
    def __init__(self, *args, **kwargs):
        super(PaConv2d, self).__init__(nn.Conv2d, *args, **kwargs)

    @classmethod
    def from_module(cls, *args, **kwargs):
        return super(PaConv2d, cls).from_module(*args, **kwargs)


class PaConv1d(PaConvLoRA):
    def __init__(self, *args, **kwargs):
        super(PaConv1d, self).__init__(nn.Conv1d, *args, **kwargs)


# Can Extend to other ones like this


class PaConv3d(PaConvLoRA):
    def __init__(self, *args, **kwargs):
        super(PaConv3d, self).__init__(nn.Conv3d, *args, **kwargs)


class PaSequential(nn.Sequential):
    is_palora_layer = True

    def forward(self, x: torch.Tensor, ray: torch.Tensor):
        for module in self:
            # Check if module accepts ray argument (PaLoRA layers, DumbWrapper, or modules with ray param)
            if getattr(module, "is_palora_layer", False) or hasattr(module, 'module'):
                x = module(x, ray=ray)
            else:
                # Check if module's forward method accepts 'ray' as a parameter
                sig = inspect.signature(module.forward)
                if 'ray' in sig.parameters:
                    x = module(x, ray=ray)
                else:
                    x = module(x)
        return x

    def __repr__(self):
        return "Pa" + super().__repr__()

    @classmethod
    def from_module(cls, module: nn.Sequential):
        return cls(module._modules)
