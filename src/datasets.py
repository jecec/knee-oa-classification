from torch.utils.data import Dataset
import numpy as np
import cv2

def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    xray = xray.astype(np.float32) / 255.0

    xray_3ch = np.zeros((3, xray.shape[0], xray.shape[1]), dtype=xray.dtype) # (3, H, W)
    xray_3ch[0] = xray
    xray_3ch[1] = xray
    xray_3ch[2] = xray

    return xray_3ch

class Knee_xray_dataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = read_xray(self.dataset['Directory'].iloc[idx])
        label = self.dataset["KL"].iloc[idx]

        if self.transforms is not None:
            for t in self.transforms:
                if hasattr(t, 'randomize'):
                    t.randomize()
                img = t(img)

        res = {
            'img' : img,
            'label' : label
        }
        return res