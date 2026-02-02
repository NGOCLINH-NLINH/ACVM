from pathlib import Path
from torchvision.datasets import CIFAR100


def ensure_labels_file(labels_file):
    p = Path(labels_file)
    if not p.exists():
        print("Labels file not found. Creating from CIFAR100 dataset")
        ds = CIFAR100(root="data", train=True, download=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("\n".join(ds.classes))
        print("Saved labels to", labels_file)