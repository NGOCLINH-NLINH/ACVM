import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar100_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def build_task_loaders(data_dir='./data', num_tasks=10, batch_size=64, num_workers=2):
    train_transform, test_transform = get_cifar100_transforms()

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    classes_per_task = num_classes // num_tasks

    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)

    task_loaders = []
    task_classes_list = []

    for t in range(num_tasks):
        start_class = t * classes_per_task
        end_class = (t + 1) * classes_per_task
        current_classes = list(range(start_class, end_class))

        class_names_str = [class_names[c].replace('_', ' ') for c in current_classes]
        task_classes_list.append(class_names_str)

        train_indices = np.where(np.isin(train_labels, current_classes))[0]
        test_indices = np.where(np.isin(test_labels, current_classes))[0]

        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        task_loaders.append({'train': train_loader, 'test': test_loader})

    return task_loaders, task_classes_list
