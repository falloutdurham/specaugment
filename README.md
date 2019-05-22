# SpecAugment / PyTorch

Implements the frequency and time masking transforms from [SpecAugment](https://arxiv.org/abs/1904.08779) in PyTorch.

## Example

    transforms.Compose([
         transforms.ToTensor(),
         FrequencyMask(max_width=10, use_mean=False)])