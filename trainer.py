import torch
import numpy as np
import tqdm
import csv
import os
from timm.models import create_model
import torch.autograd.profiler as profiler
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from style_augmentation.styleaug import StyleAugmentor

# Borrow from Improved Regularization of Convolutional Neural Networks with Cutout (https://github.com/uoguelph-mlrg/Cutout)
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Trainer:
    def __init__(self, epoch, 
               dataset_path='./drive/My Drive/datasets/car classification/train_data', 
               val_path='./drive/My Drive/datasets/car classification/val_data', 
               val_crop='five', batch_size=128, model_name='tf_efficientnet_b3_ns', 
               lr=0.001, lr_min=0.0001, weight_decay=1e-4, momentum=0.9, log_step=25, save_step=10,
               log_path='./drive/My Drive/cars_log.txt', cutout=True, style_aug=False,
               resume=False, resume_path='./drive/My Drive/ckpt/'):

        # initialize attributes
        self.epoch = epoch
        self.dataset_path = dataset_path
        self.val_path = val_path
        self.val_crop = val_crop
        self.batch_size = batch_size
        self.model_name = model_name
        self.lr = lr
        self.lr_mi = lr_min
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.log_step = log_step
        self.save_step = save_step
        self.log_path = log_path
        self.cutout = cutout
        self.style_aug = style_aug
        self.resume = resume
        self.resume_path = resume_path
        if model_name == 'tf_efficientnet_b0_ns':
            self.input_size = (224, 224)
        elif model_name == 'tf_efficientnet_b3_ns':
            self.input_size = (300, 300)
        elif model_name == 'tf_efficientnet_b4_ns':
          scaleelf.input_size = (380, 380)
        elif model_name == 'tf_efficientnet_b6_ns':
            self.input_size = (528, 528)
        else:
            raise Exception('non-valid model name')
        
        # Compose transforms
        transform = []
        val_transform = []

        transform += [transforms.RandomResizedCrop(self.input_size, scale=(0.125, 1.0))]

        if not self.style_aug:
            transform.append(transforms.ColorJitter(0.3, 0.3, 0.3, 0.3))

        transform += [transforms.RandomHorizontalFlip(), transforms.ToTensor()]

        if self.style_aug:
            transform.append(StyleAugmentor())
            transform.append([])

        transform += [Cutout(n_holes=1, length=int(self.input_size[0]/3)), transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])]
        if self.val_crop == 'center':
            val_transform += [transforms.Resize(int(self.input_size[0]*(1.14))), transforms.CenterCrop(self.input_size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])]
        else:
            val_transform += [transforms.Resize(int(self.input_size[0]*(1.14))), transforms.FiveCrop(self.input_size)]
            val_transform.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            val_transform.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])(crop) for crop in crops])))

        self.transform = transforms.Compose(transform)
        self.val_transform = transforms.Compose(val_transform)

        self.dataset = ImageFolder(self.dataset_path, transform=self.transform)
        self.val_dataset = ImageFolder(self.val_path, transform=self.val_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(model_name, pretrained=True, num_classes=196).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
        self.start_epoch = 0

        if resume:
            ckpt = torch.load(self.resume_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_epoch = ckpt['epoch']

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch, last_epoch=self.start_epoch-1,
                  eta_min=lr_min)

    def train(self):
        for epoch in range(self.start_epoch, self.epoch):
            pbar = tqdm.tqdm(self.dataloader)
            pbar.set_description('training process')
            epoch_loss_mean = 0 
            epoch_acc_mean = 0
            loss_mean = 0
            acc_mean = 0
            self.model.train()
            self.scheduler.step()
            batch_number = len(pbar)
            for it, data in enumerate(pbar):

                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                preds = self.model(inputs)
                loss = self.criterion(preds, labels)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_mean += loss.item()
                epoch_loss_mean += loss.item()
                acc = (preds.argmax(-1) == labels).sum().item() / labels.size()[0]
                acc_mean += acc
                epoch_acc_mean += acc

                if (it+1) % self.log_step == 0:
                    loss_mean /= self.log_step
                    acc_mean /= self.log_step
                    with open(self.log_path, 'a+') as f:
                        f.write('epoch: ' + str(epoch) + '\n')
                        f.write('loss: ' + str(loss_mean) + '\n')
                        f.write('acc: ' + str(acc_mean) + '\n')
                        f.write('\n')
                    loss_mean = 0
                    acc_mean = 0
            epoch_loss_mean /= len(pbar)
            epoch_acc_mean /= len(pbar)
            # validate
            pbar = tqdm.tqdm(self.val_dataloader)
            pbar.set_description('validating process')
            val_loss_mean = 0
            val_acc_mean = 0
            self.model.eval()
            with torch.no_grad():
                for it, data in enumerate(pbar):
                    inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)

                    if self.val_crop == 'center':
                        preds = self.model(inputs)
                    elif self.val_crop == 'five':
                        bs, ncrops, c, h, w = inputs.size()
                        preds = self.model(inputs.view(-1, c, h, w))
                        preds = preds.view(bs, ncrops, -1).mean(1)

                    loss = self.criterion(preds, labels)
                    val_loss_mean += loss.item()
                    acc = (preds.argmax(-1) == labels).sum().item() / labels.size()[0]
                    val_acc_mean += acc
                
            val_loss_mean /= len(pbar)
            val_acc_mean /= len(pbar)


            print('loss_mean:', epoch_loss_mean, 'acc_mean:', epoch_acc_mean)
            print('val_loss_mean:', val_loss_mean, 'val_acc_mean:', val_acc_mean)
            
            with open(self.log_path, 'a+') as f:
                f.write('epoch summary\n')
                f.write('epoch: ' + str(epoch) + '\n')
                f.write('loss: ' + str(epoch_loss_mean) + '\n')
                f.write('acc: ' + str(epoch_acc_mean) + '\n')
                f.write('val_loss: ' + str(val_loss_mean) + '\n')
                f.write('val_acc: ' + str(val_acc_mean) + '\n')
                f.write('\n')
            if (epoch+1) % self.save_step == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch + 1
                }, './drive/My Drive/ckpt/%d.pth'%(epoch+1))
        

def criterion(self, preds, trues):
    return torch.nn.CrossEntropyLoss()(preds, trues)