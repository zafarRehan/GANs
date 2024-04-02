import torchvision.datasets as datasets
import torchvision.transforms as transforms


class GANDataset:

    def __init__(self, hyp):
        transforms_ = transforms.Compose(
            [
                transforms.Resize(hyp.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(hyp.CHANNELS_IMG)], [0.5 for _ in range(hyp.CHANNELS_IMG)]
                ),
            ]
        )
        self.dataset = datasets.MNIST(root="dataset/", transform=transforms_, download=True)