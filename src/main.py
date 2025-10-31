from args import get_args
import pandas as pd
import os
from datasets import Knee_xray_dataset
from torch.utils.data import DataLoader
from models import MyModel
from trainer import train_model, validate_model
import torch
import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. We need some arguments
    args = get_args()

    # 2. Iterate among the folds
    for fold in range(5):
        print(f"Training on fold {fold+1}")
        train_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{str(fold)}_train.csv'))
        val_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{str(fold)}_val.csv'))

        # 3. Preparing datasets
        train_dataset = Knee_xray_dataset(
            dataset=train_set,
            transforms = [transforms.HorizontalFlip(prob=0.5),
                          transforms.OneOf([
                              transforms.NoTransform(),
                              transforms.Rotate(),
                              transforms.DualCompose([
                                  transforms.Scale(ratio_range=(0.7, 1.2), prob=1),
                                  transforms.Rescale()
                              ])
                          ])
            ])
        val_dataset = Knee_xray_dataset(
            dataset=val_set,)

        # 4. Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

        # 5. Initializing the model
        model = MyModel(args.backbone).to(device)

        # 6. Training the model
        train_model(model, train_loader, val_loader, fold)
        print()

if __name__ == '__main__':
    main()