import facenet
from PIL import Image
from PIL import ImageDraw
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
import math
import numpy as np
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity

def detectFace(im_path):
    image = face_recognition.load_image_file(im_path)
    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        width = right - left
        height = bottom - top
        if (width > height):
            right -= (width - height)
        elif (height > width):
            bottom -= (height - width)

        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        face_resize = pil_image.resize((48, 48), Image.ANTIALIAS)
        return face_resize

# from sklearn.metrics.pairwise import cosine_similarity
def getMold(vec):
    sum = 0.0
    for i in range(len(vec)):
        sum += vec[i] * vec[i]
    return math.sqrt(sum)

def getSimilarity(lhs,rhs):
    tmp = 0.0
    for i in range(len(lhs)):
        tmp += lhs[i] * rhs[i]
    return tmp/(getMold(lhs) * getMold(rhs))

if __name__ == '__main__':
    # dataset = FaceMatchDataset(r"E:\save_path\identity\48")
    cls_num = 10177
    # cls_num =10000
    # print(cls_num)
    # cls_num = 10000
    net = facenet.Net(cls_num)
    net.cuda()
    net.eval()

    cls_loss_fn = nn.NLLLoss()
    center_loss_fn = facenet.CenterLoss( 128,cls_num)
    center_loss_fn.cuda()

    state_dict = net.state_dict()
    state_dict.update(torch.load("E:/save_path/identity/param-ok/fnet_model_v18_full.pt"))
    net.load_state_dict(state_dict)
    center_loss_fn.load_state_dict(torch.load("E:/save_path/identity/param-ok/fnet_center_v18.pt"))
    composed_fn = transforms.Compose([
        transforms.ToTensor()
    ])

    im1 = detectFace(r"E:\img\6.jpg")
    # face_img1 = r"E:\save_path\identity\48\images\000019.jpg"
    # im1 = Image.open(face_img1)
    im1.show()
    im1 = composed_fn(im1)
    im1 = im1.cuda()
    im1.unsqueeze_(0)
    _output_feature1, _output_cls1 = net(im1)
    _output_feature1 = _output_feature1.cpu().data.numpy().reshape(128)
    # print(np.argmax(_output_cls1.cpu().data.numpy()))

    # face_img2 = r"E:\save_path\identity\48\images\000022.jpg"
    # im2 = Image.open(face_img2)
    im2 = detectFace(r"E:\img\8.jpg")
    im2.show()
    im2 = composed_fn(im2)
    im2 = im2.cuda()
    im2.unsqueeze_(0)
    _output_feature2, _output_cls2 = net(im2)
    _output_feature2 = _output_feature2.cpu().data.numpy().reshape(128)
    # print(np.argmax(_output_cls2.cpu().data.numpy()))
    #
    print(getSimilarity(_output_feature1,_output_feature2))
    a = [_output_feature1, _output_feature2]
    #
    print(cosine_similarity(a))


