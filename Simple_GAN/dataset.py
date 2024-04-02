import torchvision.datasets as datasets
import torchvision.transforms as transforms


class GANDataset:

    def __init__(self):
        transforms_ = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.dataset = datasets.MNIST(root="dataset/", transform=transforms_, download=True)