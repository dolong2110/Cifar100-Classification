import torch
import torchvision.transforms as tt

def get_cifar100_mean_std(images, digit_after_decimal) -> (float, float):
    average = torch.Tensor([0, 0, 0])
    standard_dev = torch.Tensor([0, 0, 0])
    length_data = 0
    for image_type in images:
        length_data += len(image_type[0])
        for image in image_type[0]:
            average += image.mean([1, 2])
            standard_dev += image.std([1, 2])

    return (torch.round((average / length_data) * 10**digit_after_decimal) / (10**digit_after_decimal)).tolist(), \
           (torch.round((standard_dev / length_data) * 10**digit_after_decimal) / (10**digit_after_decimal)).tolist()

def augment_cifar100(image_resolution, mean: float, std: float):
    transform = tt.Compose([
        tt.RandomCrop(image_resolution, padding=4, padding_mode='reflect'),  # image resolution is 32 for cifar100
        tt.RandomHorizontalFlip(),
        tt.RandomRotation(15),
        tt.ToTensor(),
        tt.Normalize(mean, std, inplace=True)
    ])

    return transform

def augment_general(image_resolution, mean, std):
    transform = tt.Compose([
        # tt.ToPILImage(),
        tt.RandomResizedCrop(256, scale=(0.5, 0.9), ratio=(1, 1)),
        tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        tt.RandomCrop(image_resolution, padding=4, padding_mode='reflect'),  # image resolution is 32 for cifar100
        tt.RandomHorizontalFlip(),
        tt.RandomRotation(15),
        tt.ToTensor(),
        tt.Normalize(mean, std)
    ])

    return transform

def to_tensor():
    transform = tt.Compose([tt.ToTensor()])
    return transform

def normalize_data(mean, std):
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(mean, std)
    ])

    return transform

def augment_contrastive(image_resolution):
    color_jitter = tt.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
    # 10% of the image usually, but be careful with small image sizes
    # blur = tt.GaussianBlur((3, 3), (0.1, 2.0))
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Resize((256, 256)),
        tt.RandomResizedCrop(size=(image_resolution, image_resolution)),
        tt.RandomHorizontalFlip(p=0.5),
        tt.RandomApply([color_jitter], p=0.8),
        # tt.RandomApply([blur], p=0.5),
        tt.RandomGrayscale(p=0.2),
        tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transform