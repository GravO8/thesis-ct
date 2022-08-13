import torch


class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, Y_pred, Y):
        fl1 = self.alpha     * ((1-Y_pred)**self.gamma) * Y     * torch.log(Y_pred)   # Y == 1
        fl0 = (1-self.alpha) * (Y_pred**self.gamma)     * (1-Y) * torch.log(1-Y_pred) # Y == 0
        fl  = -fl1 -fl0
        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        return fl


if __name__ == "__main__":
    loss   = BinaryFocalLoss(gamma = 2, alpha = .25, reduction = "mean")
    input  = torch.tensor([0.1, 0.9, 0.9, .1], requires_grad = True)
    target = torch.Tensor([0, 0, 1, 1])
    output = loss(input, target)
    output.backward()
    print(output)
