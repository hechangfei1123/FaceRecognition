import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import traceback
from simpling import FaceMatchDataset
from torch.utils.data import DataLoader
import numpy as np


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU()
    )


def conv_bn_nopad(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride,1, bias=False),
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.PReLU(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(),
    )

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class Net(nn.Module):
    def __init__(self, cls_num):
        super(Net, self).__init__()

        self._sequential = nn.Sequential(
            conv_dw(3, 32, 2),  # 48
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),  # 24
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),  # 12
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),  # 6
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 2),  # 3
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_bn_nopad(512, 512, 1),
        )
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, cls_num)

        #self.apply(weights_init)



    def forward(self, x):
        y1 = self._sequential(x)
        y2 = y1.view(-1, 512)
        y3 = self.fc1(y2)
        # print(torch.mean(y3).cpu().data.numpy(),torch.max(y3).cpu().data.numpy(),torch.min(y3).cpu().data.numpy())
        y4 = F.log_softmax(self.fc2(y3), dim=1)
        return y3, y4


class CenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1.0, use_cuda=True):
        super(CenterLoss, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))
        self.use_cuda = use_cuda

    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_centers = self.centers.index_select(dim=0, index=y)
        intra_distances = hidden.dist(expanded_centers)
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
        return loss

    def cuda(self, device_id=None):
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))

if __name__ == '__main__':
    # dataset = FaceMatchDataset(r"E:\save_path\identity\48")
    cls_num = 10177
    # cls_num =10000
    print(cls_num)
    # cls_num = 10000
    net = Net(cls_num)
    net.cuda()
    net.train()

    cls_loss_fn = nn.NLLLoss()
    center_loss_fn = CenterLoss( 128,cls_num)
    center_loss_fn.cuda()

    state_dict = net.state_dict()
    state_dict.update(torch.load("E:/save_path/identity/param/fnet_model_v18_full.pt"))
    net.load_state_dict(state_dict)
    center_loss_fn.load_state_dict(torch.load("E:/save_path/identity/param/fnet_center_v18.pt"))

    optimizer0 = optim.Adam(net.parameters())
    optimizer1 = optim.Adam(center_loss_fn.parameters())

    k = 0

    for _ in range(1000):
        dataset = FaceMatchDataset(r"E:\save_path\identity\48")
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=2, drop_last=True)
        try:
            for i, (_img_data, _cls) in enumerate(dataloader):
                img_data_ = Variable(_img_data)
                cls_ = Variable(_cls)


                img_data_ = img_data_.cuda()
                cls_ = cls_.cuda()

                _output_feature, _output_cls = net(img_data_)
                print(_output_cls.shape)
                print(cls_.shape)
                cls_ = cls_.type(torch.cuda.LongTensor).squeeze()
                # break

                cls_loss = cls_loss_fn(_output_cls, cls_)

                center_loss = center_loss_fn(cls_,_output_feature)

                loss = cls_loss + center_loss

                optimizer0.zero_grad()
                optimizer1.zero_grad()
                loss.backward()
                optimizer0.step()
                optimizer1.step()

                print(k, i, "............................................................................... loss:", loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(), " center_loss:",
                      center_loss.cpu().data.numpy())

                k += 1
                if k % 100 == 0:
                    #分类层权重
                    torch.save(net.state_dict(), "E:/save_path/identity/param/fnet_model_v18_full.pt")
                    dic = net.state_dict()
                    dic.pop('fc2.weight')
                    dic.pop('fc2.bias')

                    torch.save(dic, "E:/save_path/identity/param/fnet_model_v18.pt")
                    torch.save(center_loss_fn.state_dict(), "E:/save_path/identity/param/fnet_center_v18.pt")
                    print("save ........")
                    # exit()
        except Exception:
            traceback.print_exc()
                # exit()
#