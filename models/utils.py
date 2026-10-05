import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import default_collate

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def generate_2d_sincos_embedding(grid_size, embed_dim, step=1.0):
    """
    Generates a 2D sine-cosine positional embedding for a grid of patches.

    Args:
        grid_size (int): The height/width of the grid (e.g., 16 or 8).
        embed_dim (int): The total embedding dimension (must be divisible by 4).
        step (float): The spatial step size. Crucial for multi-scale alignment!

    Returns:
        torch.Tensor: A tensor of shape [grid_size * grid_size, embed_dim]
    """
    assert embed_dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D sin-cos."

    coords = torch.arange(0, grid_size * step, step, dtype=torch.float32)

    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')

    grid_y = grid_y.flatten()
    grid_x = grid_x.flatten()

    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4 - 1)
    omega = 1.0 / (10000.0 ** omega)

    out_y = grid_y.unsqueeze(1) * omega.unsqueeze(0)
    out_x = grid_x.unsqueeze(1) * omega.unsqueeze(0)

    pos_emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1) 
    pos_emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)  

    pos_embed = torch.cat([pos_emb_y, pos_emb_x], dim=1)

    return pos_embed


def top_accuracy(output, target, topk=(1, 5)):
    """Computes the top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  
    correct = pred.eq(target.view(1, -1).expand_as(pred))  

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc_k = correct_k * (100.0 / batch_size)
        results.append(acc_k.item())
    return results


def mixup_cutmix_collate_fn(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=1000):
    """
    Wrapper function to add Mixup and Cutmix to our image processing pipelines.

    Args:
        mixup_alpha: Alpha parameter for Beta distribution from which mixup lambda is sampled
        cutmix_alpha: Alpha parameter for Beta distribution from which cutmix lambda is samples
        num_classes: How many classes are there to predict from

    Note!

    Normally, we have a single label for each image (and our dataloader returns an index representing
    what class its in). But now instead of returning a single tensor of size (Batch, ) that has these indexes,
    we will instead return (Batch x Num Classes), as each image (after transformation) will be a mixture of two images
    so we return the proportion of pixels of each image represented in each image.
    """
    mix_cut_transform = None

    mixup_cutmix = []
    if mixup_alpha > 0:
        print("Enabling MixUp!")
        mixup_cutmix.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        print("Enabling CutMix!")
        mixup_cutmix.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))

    if len(mixup_cutmix) > 0:
        mix_cut_transform = v2.RandomChoice(mixup_cutmix)

    def collate_fn(batch):
        collated = default_collate(batch)

        if mix_cut_transform is not None:
            collated = mix_cut_transform(collated)
        return collated

    return collate_fn


def train_transformations(image_size=(224, 224),
                          image_mean=IMAGENET_MEAN,
                          image_std=IMAGENET_STD,
                          hflip_probability=0.5,
                          interpolation=InterpolationMode.BILINEAR,
                          random_aug_magnitude=12):
    """
    Dataloader with Random Augmentation

    Args:
        image_size: What size image do we want to return?
        image_mean: Mean of the image channels
        image_std: Standard deviation of the image channels
        hflip_probability: Probability of random horizontal flipping
        interpolation: What interpolation model do you want to use?
        random_aug_magnitude: Strength of augmentations (Valid values between 0 and 30)
    """
    transformation_chain = []

    transformation_chain.append(v2.RandomResizedCrop(image_size, interpolation=interpolation, antialias=True))

    if hflip_probability > 0:
        transformation_chain.append(v2.RandomHorizontalFlip(p=hflip_probability))

    if random_aug_magnitude > 0:
        print("Enabling Random Augmentation!")
        transformation_chain.append(v2.RandAugment(magnitude=random_aug_magnitude, interpolation=interpolation))

    transformation_chain.append(v2.PILToTensor())

    transformation_chain.append(v2.ToDtype(torch.float32, scale=True))

    transformation_chain.append(v2.Normalize(mean=(image_mean), std=image_std))

    return v2.Compose(transformation_chain)


def eval_transformations(image_size=(224, 224),
                         resize_size=(256, 256),
                         image_mean=IMAGENET_MEAN,
                         image_std=IMAGENET_STD,
                         interpolation=InterpolationMode.BILINEAR):
    """
    Quick evaluation dataloader, no fancy transformations in this one

    Args:
        image_size: What size image do we want to pass to model, this is a center crop size?
        resize_size: Original size we resize image to before center crop
        image_mean: Mean of the image channels
        image_std: Standard deviation of the image channels
        interpolation: What interpolation model do you want to use?
    """

    transformations = v2.Compose(
        [
            v2.Resize(resize_size, interpolation=interpolation, antialias=True),
            v2.CenterCrop(image_size),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_mean, std=image_std)
        ]
    )

    return transformations


