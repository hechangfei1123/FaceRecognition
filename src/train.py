import os

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim

from simpling import FaceDataset
from simpling import LandarkDataset


class Trainer:
    def __init__(self, net, save_path, dataset_path,  isCuda=True):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda

        if self.isCuda:
            self.net.cuda()

        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()
        self.landmark_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        # if os.path.exists(self.save_path):
        #     net.load_state_dict(torch.load(self.save_path))

    def train(self):
        faceDataset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=128, shuffle=True, num_workers=4)
        for j in range(100):
            for i, (_img_data, _category, _offset) in enumerate(dataloader):
                img_data_ = Variable(_img_data)
                category_ = Variable(_category)
                offset_ = Variable(_offset)

                if self.isCuda:
                    img_data_ = img_data_.cuda()
                    category_ = category_.cuda()
                    offset_ = offset_.cuda()

                _output_category, _output_offset = self.net(img_data_)
                output_category = _output_category.view(-1, 1)
                # output_offset = _output_offset.view(-1, 4)
                # output_landmark = _output_landmark.view(-1, 10)
                # 计算分类的损失
                category_mask = torch.lt(category_, 2)  # part样本不参与分类损失计算
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                cls_loss = self.cls_loss_fn(output_category, category)

                # 计算bound的损失
                offset_mask = torch.gt(category_, 0)  # 负样本不参与计算
                # offset = offset_[offset_mask]
                # offset = torch.masked_select(offset_, offset_mask)
                # output_offset = _output_offset[offset_mask]
                # output_offset = torch.masked_select(_output_offset, offset_mask)
                offset_index = torch.nonzero(offset_mask)[:,0]
                offset = offset_[offset_index]
                output_offset = _output_offset[offset_index]
                offset_loss = self.offset_loss_fn(output_offset, offset)  # 损失

                loss = cls_loss * 1.0 + offset_loss * 0.5

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(" loss:", loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(), " offset_loss",
                      offset_loss.cpu().data.numpy())
            torch.save(self.net.state_dict(), os.path.join(self.save_path,"pnet_epoch_%d.pt" % j))
            torch.save(self.net, os.path.join(self.save_path, "pnet_epoch_model_%d.pkl" % j))
            print("save success")


    def trainOnet(self):
        faceDataset = LandarkDataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=128, shuffle=True, num_workers=4)
        for j in range(100):
            for i, (_img_data, _category, _offset,_landark) in enumerate(dataloader):
                img_data_ = Variable(_img_data)
                category_ = Variable(_category)
                offset_ = Variable(_offset)
                landark_ = Variable(_landark)

                if self.isCuda:
                    img_data_ = img_data_.cuda()
                    category_ = category_.cuda()
                    offset_ = offset_.cuda()
                    landark_ = landark_.cuda()

                _output_category, _output_offset,_output_landmark = self.net(img_data_)
                output_category = _output_category.view(-1, 1)
                # output_offset = _output_offset.view(-1, 4)
                # output_landmark = _output_landmark.view(-1, 10)
                # 计算分类的损失
                category_mask = torch.lt(category_, 2)  # part样本不参与分类损失计算
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                cls_loss = self.cls_loss_fn(output_category, category)

                # 计算bound的损失
                offset_mask = torch.gt(category_, 0)  # 负样本不参与计算
                # offset = offset_[offset_mask]
                # offset = torch.masked_select(offset_, offset_mask)
                # output_offset = _output_offset[offset_mask]
                # output_offset = torch.masked_select(_output_offset, offset_mask)
                offset_index = torch.nonzero(offset_mask)[:, 0]
                offset = offset_[offset_index]
                output_offset = _output_offset[offset_index]
                offset_loss = self.offset_loss_fn(output_offset, offset)  # 损失

                landark_mask = torch.eq(category_,1)
                landark_index = torch.nonzero(offset_mask)[:, 0]
                landark = landark_[offset_index]
                output_landmark = _output_landmark[offset_index]
                # landark = torch.masked_select(landark_, landark_mask)
                # output_landmark = torch.masked_select(_output_landmark, landark_mask)
                landark_loss = self.landmark_loss_fn(output_landmark,landark)

                loss = cls_loss * 0.8 + offset_loss * 0.6 + landark_loss * 1.5

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(" loss:", loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(), " offset_loss",
                      offset_loss.cpu().data.numpy(),"landark_loss",landark_loss.cpu().data.numpy())
            torch.save(self.net.state_dict(), os.path.join(self.save_path, "pnet_epoch_%d.pt" % j))
            torch.save(self.net, os.path.join(self.save_path, "pnet_epoch_model_%d.pkl" % j))
            print("save success")