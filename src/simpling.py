import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

class FaceDataset(Dataset):
    def __init__(self,anno_path,positive_size=20000,part_size=20000,negative_size=60000):
        self.composed = transforms.Compose([
            transforms.ToTensor()
        ])
        self.anno_path = anno_path
        negative_anno_file = os.path.join(anno_path,"negative.txt")
        positive_anno_file = os.path.join(anno_path, "positive.txt")
        part_anno_file = os.path.join(anno_path, "part.txt")

        with open(negative_anno_file) as f:
            negative_datas = f.readlines()
            # f0 = open(r"E:\save_path\12\negative00.txt", "w")
            # for data in negative_datas:
            #     da = data.split(' ')
            #     if da.__len__() < 3:
            #         f0.write(data.split('\n')[0] + ' 0 0 0 0\n')
            #     else:
            #         f0.write(data.split('\n')[0] + '\n')
            # f0.close()

        with open(positive_anno_file) as f:
            positive_datas = f.readlines()

        with open(part_anno_file) as f:
            part_datas = f.readlines()

        self.dataset = []
        self.dataset.extend(np.random.choice(negative_datas,size=negative_size))
        self.dataset.extend(np.random.choice(positive_datas,size=positive_size))
        self.dataset.extend(np.random.choice(part_datas,size=part_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        lines = self.dataset[idx].split(" ")
        face_img = os.path.join(self.anno_path,lines[0])
        img_data = self.composed(Image.open(face_img))

        category = torch.FloatTensor([float(lines[1].strip())])
        offset = torch.FloatTensor(
            [float(lines[2].strip()), float(lines[3].strip()), float(lines[4].strip()), float(lines[5].strip())])

        return img_data, category, offset

class LandarkDataset(Dataset):
    def __init__(self,anno_path,positive_size=200000,part_size=20000,negative_size=60000):
        self.composed = transforms.Compose([
            transforms.ToTensor()
        ])
        self.anno_path = anno_path
        negative_anno_file = os.path.join(anno_path, "negative.txt")
        positive_anno_file = os.path.join(anno_path, "positive.txt")
        part_anno_file = os.path.join(anno_path, "part.txt")

        with open(negative_anno_file) as f:
            negative_datas = f.readlines()
            # f0 = open(r"E:\save_path\48\negative00.txt", "w")
            # for data in negative_datas:
            #     da = data.split(' ')
            #     if da.__len__() < 8:
            #         f0.write(data.split('\n')[0] + ' 0 0 0 0 0 0 0 0 0 0\n')
            #     else:
            #         f0.write(data.split('\n')[0] + '\n')
            # f0.close()

        with open(part_anno_file) as f:
            part_datas = f.readlines()
            # f0 = open(r"E:\save_path\48\part00.txt", "w")
            # for data in part_datas:
            #     da = data.split(' ')
            #     if da.__len__() < 8:
            #         f0.write(data.split('\n')[0] + ' 0 0 0 0 0 0 0 0 0 0\n')
            #     else:
            #         f0.write(data.split('\n')[0] + '\n')
            # f0.close()

        with open(positive_anno_file) as f:
            positive_datas = f.readlines()

        self.dataset = []
        self.dataset.extend(np.random.choice(negative_datas, size=negative_size))
        self.dataset.extend(np.random.choice(positive_datas, size=positive_size))
        self.dataset.extend(np.random.choice(part_datas, size=part_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        lines = self.dataset[idx].split(" ")
        face_img = os.path.join(self.anno_path,lines[0])
        img_data = self.composed(Image.open(face_img))

        category = torch.FloatTensor([float(lines[1].strip())])
        offset = torch.FloatTensor(
            [float(lines[2].strip()), float(lines[3].strip()), float(lines[4].strip()), float(lines[5].strip())])
        landark = torch.FloatTensor(
            [float(lines[6].strip()), float(lines[7].strip()), float(lines[8].strip()), float(lines[9].strip()),float(lines[10].strip()),float(lines[11].strip()), float(lines[12].strip()), float(lines[13].strip()), float(lines[14].strip()),float(lines[15].strip())])

        return img_data, category, offset,landark


class FaceMatchDataset(Dataset):
    # def __init__(self,anno_path,identity_size=197325):
    def __init__(self, anno_path, identity_size=150000):
        self.composed = transforms.Compose([
            transforms.ToTensor()
        ])
        self.anno_path = anno_path
        identity_anno_file = os.path.join(anno_path, "identity.txt")
        with open(identity_anno_file) as f:
            identity_datas = f.readlines()
        self.dataset = []
        self.dataset.extend(np.random.choice(identity_datas, size=identity_size))

    def __len__(self):
        return len(self.dataset)

    def cls_num(self):
        return 10177

    def __getitem__(self, idx):
        lines = self.dataset[idx].split(" ")
        face_img = os.path.join(self.anno_path,"images", lines[0])
        img_data = self.composed(Image.open(face_img))

        identity = torch.IntTensor([int(lines[1].strip())])
        # ide = [0 for i in range(10177)]
        # ide[identity - 1] = 1
        # ide = np.array(ide)
        # ide = np.float32(ide)
        #ide = ide.astype(np.int64)
        return img_data,identity


if __name__ == '__main__':
    # faceDataset = FaceDataset(r"E:\save_path\12")
    # dataloader = DataLoader(faceDataset, batch_size=128,
    #                         shuffle=True, num_workers=4)

    # landarkDataset = LandarkDataset(r"E:\save_path\48")
    # dataloader = DataLoader(landarkDataset, batch_size=128,
    #                         shuffle=True, num_workers=4)

    faceMatchDataset = FaceMatchDataset(r"E:\save_path\identity\48")
    dataloader = DataLoader(faceMatchDataset, batch_size=128,
                             shuffle=True, num_workers=4)

    for x in dataloader:
        print("...........")