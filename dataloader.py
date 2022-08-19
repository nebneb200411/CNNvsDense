from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])

train_dataset_cnn = datasets.MNIST(
        './data',               
        train = True,           
        download = True,        
        transform = transform_cnn
    )

test_dataset_cnn = datasets.MNIST(
        './data',               
        train = False,               
        transform = transform_cnn
    )

train_dataloader_cnn = DataLoader(
    train_dataset_cnn,
    batch_size = 16,
    shuffle = True
)

test_dataloader_cnn = DataLoader(
    test_dataset_cnn,
    batch_size = 16,
    shuffle = True
)

transform_dense = transforms.Compose([
        #transforms.Resize((28 * 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

train_dataset_dense = datasets.MNIST(
        './data',               
        train = True,           
        download = True,        
        transform = transform_dense
    )

test_dataset_dense = datasets.MNIST(
        './data',               
        train = False,               
        transform = transform_dense
    )

train_dataloader_dense = DataLoader(
    train_dataset_dense,
    batch_size = 16,
    shuffle = True
)

test_dataloader_dense = DataLoader(
    test_dataset_dense,
    batch_size = 16,
    shuffle = True
)