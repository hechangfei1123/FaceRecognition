import torch
from torch.autograd import Variable

from PIL import Image
from PIL import ImageDraw
import numpy as np

import utils

import net

from torchvision import transforms
import time


class Detector:
    def __init__(self, pnet_param="E:/save_path/12/param/pnet_epoch_99.pt", rnet_param="E:/save_path/24/param/pnet_epoch_99.pt", onet_param="E:/save_path/48/param/pnet_epoch_99.pt",
                 isCuda=True):

        self.isCuda = isCuda

        self.pnet = net.PNet()
        self.rnet = net.RNet()
        self.onet = net.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self, image):

        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])

        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])

        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])

        return onet_boxes

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = Variable(torch.stack(_img_dataset))
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.7)
        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.5)

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = Variable(torch.stack(_img_dataset))
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset,landmark = self.onet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.7)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            px1 = _x1 + ow * landmark[idx][0]
            py1 = _y1 + oh * landmark[idx][1]
            px2 = _x1 + ow * landmark[idx][2]
            py2 = _y1 + oh * landmark[idx][3]
            px3 = _x1 + ow * landmark[idx][4]
            py3 = _y1 + oh * landmark[idx][5]
            px4 = _x1 + ow * landmark[idx][6]
            py4 = _y1 + oh * landmark[idx][7]
            px5 = _x1 + ow * landmark[idx][8]
            py5 = _y1 + oh * landmark[idx][9]

            boxes.append([x1, y1, x2, y2, cls[idx][0],px1,py1,px2,py2,px3,py3,px4,py4,px5,py5])

        return utils.nms(np.array(boxes), 0.7, isMin=True)

    def __pnet_detect(self, image):

        boxes = []

        img = image
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len > 12:

            img_data = self.__image_transform(img)
            img_data = Variable(img_data)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)

            _cls, _offest = self.pnet(img_data)

            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            idxs = torch.nonzero(torch.gt(cls, 0.6))

            for idx in idxs:
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

        return utils.nms(np.array(boxes), 0.5)

    # 将回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = float(start_index[1] * stride) / scale
        _y1 = float(start_index[0] * stride) / scale
        _x2 = float(start_index[1] * stride + side_len) / scale
        _y2 = float(start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        x1 = _x1 + ow* _offset[0].long()
        y1 = _y1 + oh* _offset[1].long()
        x2 = _x2 + ow* _offset[2].long()
        y2 = _y2 + oh* _offset[3].long()

        return [float(x1), float(y1), float(x2), float(y2), cls]


if __name__ == '__main__':

    image_file = r"E:\BaiduNetdiskDownload\WIDER_test\images\11--Meeting\11_Meeting_Meeting_11_Meeting_Meeting_11_41.jpg"
    detector = Detector()

    with Image.open(image_file) as im:
        # boxes = detector.detect(im)
        # print("----------------------------")
        boxes = detector.detect(im)
        print(im.size)
        imDraw = ImageDraw.Draw(im)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])



            print(box[4])
            imDraw.rectangle((x1, y1, x2, y2), outline='red')
            imDraw.point((box[5],box[6]),fill='red')
            imDraw.point((box[7], box[8]), fill='red')
            imDraw.point((box[9], box[10]), fill='red')
            imDraw.point((box[11], box[12]), fill='red')
            imDraw.point((box[13], box[14]), fill='red')

        im.show()
