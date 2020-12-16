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

class MultiScaleFiveCrop(object):
    def __init__(self, sizes=(680, 600, 528)):
        self.sizes = sizes # 680, 528, 410

    def __call__(self, img):
        c, h, w = img.size()
        if h<w:
            s = h
        else:
            s = w
        scales = tuple(map(lambda x: x/s, self.sizes))
        img = img.repeat(3, 1, 1, 1)
        cc = []
        
        for i in range(len(self.sizes)):
            c = ()
            im = transforms.Resize(tuple(map(lambda x: int(x*scales[i]*(256/224)), (h, w))))(img[i, :, :, :]) # c, h, w, 3
            crops = transforms.FiveCrop(tuple(map(lambda x: int(x*scales[i]), (h, w))))(im.unsqueeze(0))
            c += crops
            cc.append(torch.cat(c))

        return cc


class Tester:
    def __init__(self, epoch, dataset_path='./drive/My Drive/datasets/car classification/train_dataset', 
                 val_path='./drive/My Drive/datasets/car classification/val_data',
                 batch_size=128, model_name='tf_efficientnet_b0_ns', ckpt_path='./drive/My Drive/ckpt/190.pth', 
                 test_number=5000,
                 pseudo_test=True, crop='five', csv_path='', mode='fix', sizes=(680, 600, 528)):
        self.epoch = epoch
        self.dataset_path = dataset_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.test_number = test_number
        self.pseudo_test = pseudo_test
        self.crop = crop
        self.csv_path = csv_path
        self.mode = mode
        self.sizes = sizes

        if model_name == 'tf_efficientnet_b0_ns':
            self.input_size = (224, 224)
        elif model_name == 'tf_efficientnet_b3_ns':
            self.input_size = (300, 300)
        elif model_name == 'tf_efficientnet_b4_ns':
            self.input_size = (480, 480)
        elif model_name == 'tf_efficientnet_b6_ns':
            self.input_size = (680, 680) # 528
        else:
            raise Exception('non-valid model name')
        
        # Compose transforms
        transform = []
        fill = lambda i: transforms.Resize((i.size[1]*(2**torch.ceil(torch.log2(torch.tensor(self.input_size[1]/i.size[1])))), 
                  i.size[0]*(2**torch.ceil(torch.log2(torch.tensor(self.input_size[1]/i.size[1]))))))(i) if i.size[0] < self.input_size[0] or i.size[1] < self.input_size[1] else i
        if crop == 'center':
            transform.append(transforms.CenterCrop(self.input_size[0]))
            transform.append(transforms.ToTensor())
            transform.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        elif crop == 'five':
            transform.append(transforms.Lambda(fill))
            transform.append(transforms.FiveCrop(self.input_size[0]))
            transform.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            transform.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(crop) for crop in crops])))
        self.transform = transforms.Compose(transform)
        
        if self.pseudo_test:
            if crop == 'multi':
                self.transform_val = []
                self.dataset = []
                self.dataloader = []
                for i in range(len(self.sizes)):
                    self.transform_val.append(self.get_transform_val((self.sizes[i], self.sizes[i])))
                    self.dataset.append(ImageFolder(self.dataset_path, transform=self.transform_val[i]))
                    self.dataloader.append(DataLoader(self.dataset[i], batch_size=self.batch_size, num_workers=1, shuffle=False))
            else:
                self.dataset = ImageFolder(self.dataset_path, transform=self.transform_val)
                self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(model_name, num_classes=196).to(self.device)
        if self.mode == 'fix':
            ckpt = torch.load(self.ckpt_path)
            self.model.load_state_dict(ckpt['model'])
        else:
            ckpt = torch.load(self.ckpt_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
        self.start_epoch = 0

        l = [d.name for d in os.scandir(self.val_path) if d.is_dir()]
        l.sort()
        l[l.index('Ram CV Cargo Van Minivan 2012')] = 'Ram C/V Cargo Van Minivan 2012'
        self.label_texts = l

    def get_transform_val(self, size):
        if self.crop == 'five' or self.crop == 'multi':
            transform_val = [transforms.Resize(int(size[0]*(1.14))), transforms.FiveCrop(size)]
            transform_val.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            transform_val.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])))
        else:
            transform_val = [transforms.Resize(int(size[0]*(1.14))), transforms.CenterCrop(size)]
            transform_val.append(transforms.ToTensor())
            transform_val.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        return transforms.Compose(transform_val)

    def label_to_text(self, labels):
        texts = []
        names = sorted([i for _, _, i in os.walk(os.path.join(self.dataset_path, 'dummy'))][0])
        for i, l in enumerate(labels):
            texts.append([names[i].split('.')[0], self.label_texts[l.item()]])
        texts.insert(0, ['id', 'label'])
        return texts

    def p_test(self):
        acc_mean = 0
        loss_mean = 0

        for epoch in range(self.start_epoch, self.epoch):
            pbar = tqdm.tqdm(zip(*self.dataloader))
            pbar.set_description('testing process')
            self.model.eval()
            tested_number = 0
            with torch.no_grad():
                for it, data in enumerate(pbar):
                    if self.crop != 'multi':
                        inputs = data[0].to(self.device)
                        labels = data[1].to(self.device)
                        
                        if self.crop == 'center':
                            preds = self.model(inputs)
                        elif self.crop == 'five':
                            bs, ncrops, c, h, w = inputs.size()
                            preds = self.model(inputs.view(-1, c, h, w))
                            preds = preds.view(bs, ncrops, -1).mean(1)
                    else:
                        labels = data[0][1].to(self.device)
                        preds_l = []
                        for i in range(len(self.sizes)):
                            bs, ncrops, c, h, w = data[i][0].size()
                            inputs = data[i][0].to(self.device)
                            preds_l.append(self.model(inputs.view(-1, c, h, w)))
                            preds_l[i] = preds_l[i].view(bs, ncrops, -1).mean(1)

                    preds = torch.stack(preds_l, dim=1).mean(1)
                    loss = self.criterion(preds, labels)
                    loss_mean += loss.item()
                    acc = (preds.argmax(-1) == labels).sum().item() / labels.size()[0]
                    acc_mean += acc
                    tested_number += self.batch_size

        acc_mean /= len(self.dataloader[0]) #(len(pbar))
        loss_mean /= len(self.dataloader[0]) #(len(pbar))
        print('acc_mean:', acc_mean, 'loss_mean:', loss_mean)

    def test(self):
        for epoch in range(self.start_epoch, self.epoch):
            pbar = tqdm.tqdm(zip(*self.dataloader))
            pbar.set_description('testing process')
            self.model.eval()
            pred_labels = None
            with torch.no_grad():
                for it, data in enumerate(pbar):
                    if self.crop != 'multi':
                        inputs = data[0].to(self.device)
                        if self.crop == 'center':
                            preds = self.model(inputs)
                        elif self.crop == 'five':
                            bs, ncrops, c, h, w = inputs.size()
                            preds = self.model(inputs.view(-1, c, h, w))
                            preds = preds.view(bs, ncrops, -1).mean(1)
                    else:
                        preds_l = []
                        for i in range(len(self.sizes)):
                            bs, ncrops, c, h, w = data[i][0].size()
                            inputs = data[i][0].to(self.device)
                            preds_l.append(self.model(inputs.view(-1, c, h, w)))
                            preds_l[i] = preds_l[i].view(bs, ncrops, -1).mean(1)

                        preds = torch.stack(preds_l, dim=1).mean(1)

                    if pred_labels == None:
                        pred_labels = preds.argmax(-1)
                    else:
                        pred_labels = torch.cat([pred_labels, preds.argmax(-1)])
                        
            pred_text = self.label_to_text(pred_labels)
            with open(self.csv_path, 'w', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerows(pred_text)

    def criterion(self, preds, trues):
        return torch.nn.CrossEntropyLoss()(preds, trues)