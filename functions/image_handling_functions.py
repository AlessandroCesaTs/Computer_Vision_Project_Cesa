import torch
import torchvision.transforms as transforms

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def resize_transformation(img):
  resize = transforms.Compose([transforms.Resize([64,64]),
  transforms.ToTensor(),transforms.Grayscale()])
  i = resize(img)
  i =i*255.0 #back to 0-255
  return i

def transformation_for_AlexNet(img):
  resize = transforms.Compose([transforms.Resize([224,224]),
  transforms.ToTensor()])
  i = resize(img)
  return i


class TransformedDataSet():
    """Wrap a datset (created with imagefolder) to apply a transformation"""
    def __init__(self, ds):
        self.ds = ds
        self.transformation=transforms.RandomHorizontalFlip(1)

    def __getitem__(self, index):
        """Get a sample from the dataset at the given index"""
        img, label = self.ds[index]

        # Apply the transformation if it is provided
        if self.transformation:
            img = self.transformation(img)

        return img, label

    def __len__(self):
        """Number of batches"""
        return len(self.ds)
