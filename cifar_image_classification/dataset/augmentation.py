import torchvision.transforms as tt

def augment_cifar100_train(image_resolution, mean, std):
    transform = tt.Compose([
        # tt.ToPILImage(),
        # tt.RandomRotate
        # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
        # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        tt.RandomCrop(image_resolution, padding=4), # image resolution is 32 for cifar100
        tt.RandomHorizontalFlip(),
        tt.RandomRotation(15),
        tt.ToTensor(),
        tt.Normalize(mean, std)
    ])

    return transform

def augment_cifar100_basic(mean, std):
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(mean, std)
    ])

    return transform

def augment_contrastive(image_resolution):
    color_jitter = tt.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
    # 10% of the image usually, but be careful with small image sizes
    blur = tt.GaussianBlur((3, 3), (0.1, 2.0))
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Resize((256, 256)),
        tt.RandomResizedCrop(size=(image_resolution, image_resolution)),
        tt.RandomHorizontalFlip(p=0.5),
        tt.RandomApply([color_jitter], p=0.8),
        # transforms.RandomApply([blur], p=0.5),
        tt.RandomGrayscale(p=0.2),
        tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transform