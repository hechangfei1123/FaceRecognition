import os
from PIL import Image
import numpy as np
import sys
sys.path.append('./')

import utils
import traceback

anno_src = 'wider_face_train.txt'
img_dir = 'E:/BaiduNetdiskDownload/WIDER_train/images/'

save_path = 'E:/save_path'

for face_size in [12]:

    print("gen %i image" % face_size)
    positive_image_dir = os.path.join(save_path,str(face_size),"positive")
    negative_image_dir = os.path.join(save_path,str(face_size),"negative")
    part_image_dir = os.path.join(save_path,str(face_size),"part")

    for dir_path in [positive_image_dir,negative_image_dir,part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i,line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                strs = line.strip().split(" ")
                strs = list(filter(bool, strs))
                image_filename = strs[0].strip()
                print(image_filename)
                image_file = os.path.join(img_dir, image_filename)

                image_file += ".jpg"

                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    x2 = float(strs[3].strip())
                    y2 = float(strs[4].strip())
                    # w = float(strs[3].strip())
                    # h = float(strs[4].strip())
                    # x2 = float(x1 + w)
                    # y2 = float(y1 + h)

                    # px1 = float(strs[5].strip())
                    # py1 = float(strs[6].strip())
                    # px2 = float(strs[7].strip())
                    # py2 = float(strs[8].strip())
                    # px3 = float(strs[9].strip())
                    # py3 = float(strs[10].strip())
                    # px4 = float(strs[11].strip())
                    # py4 = float(strs[12].strip())
                    # px5 = float(strs[13].strip())
                    # py5 = float(strs[14].strip())

                    if max(x2-x1,y2-y1) < 40 or x1 < 0 or y1 < 0 or x2-x1 < 0 or y2-y1 < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]

                    # 计算出人脸中心点位置
                    cx = (x2-x1)/ 2
                    cy = (y2 + y1) / 2

                    # 使正样本和部分样本数量翻倍
                    for _ in range(5):
                    # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-(x2-x1) * 0.2, (x2-x1) * 0.2)
                        h_ = np.random.randint(-(y2-y1) * 0.2, (y2-y1) * 0.2)
                        cx_ = cx + w_
                        cy_ = cy + h_

                    # 让人脸形成正方形，并且让坐标也有少许的偏离
                    side_len = np.random.randint(int(min((x2-x1), (y2-y1)) * 0.8), np.ceil(1.25 * max((x2-x1), (y2-y1))))
                    x1_ = np.max(cx_ - side_len / 2, 0)
                    y1_ = np.max(cy_ - side_len / 2, 0)
                    x2_ = x1_ + side_len
                    y2_ = y1_ + side_len

                    crop_box = np.array([x1_, y1_, x2_, y2_])

                    # 计算坐标的偏移值
                    offset_x1 = (x1 - x1_) / side_len
                    offset_y1 = (y1 - y1_) / side_len
                    offset_x2 = (x2 - x2_) / side_len
                    offset_y2 = (y2 - y2_) / side_len
                    #
                    # offset_px1 = (px1 - x1_) / side_len
                    # offset_py1 = (py1 - y1_) / side_len
                    # offset_px2 = (px2 - x1_) / side_len
                    # offset_py2 = (py2 - y1_) / side_len
                    # offset_px3 = (px3 - x1_) / side_len
                    # offset_py3 = (py3 - y1_) / side_len
                    # offset_px4 = (px4 - x1_) / side_len
                    # offset_py4 = (py4 - y1_) / side_len
                    # offset_px5 = (px5 - x1_) / side_len
                    # offset_py5 = (py5 - y1_) / side_len

                    # 剪切下图片，并进行大小缩放
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                    iou = utils.iou(crop_box, np.array(boxes))[0]
                    if iou > 0.65:  # 正样本
                        positive_anno_file.write(
                            "positive/{0}.jpg {1} {2} {3} {4} {5} \n".format(
                                positive_count, 1, offset_x1, offset_y1,
                                offset_x2, offset_y2))
                        positive_anno_file.flush()
                        face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                        positive_count += 1
                    elif iou > 0.4:  # 部分样本
                        part_anno_file.write(
                            "part/{0}.jpg {1} {2} {3} {4} {5} \n".format(
                                part_count, 2, offset_x1, offset_y1, offset_x2,
                                offset_y2))
                        part_anno_file.flush()
                        face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                        part_count += 1
                    elif iou < 0.3:
                        negative_anno_file.write(
                            "negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                        negative_anno_file.flush()
                        face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                        negative_count += 1

                    # 生成负样本
                    _boxes = np.array(boxes)

                for i in range(5):
                    side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                    x_ = np.random.randint(0, img_w - side_len)
                    y_ = np.random.randint(0, img_h - side_len)
                    crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                    if np.max(utils.iou(crop_box, _boxes)) < 0.3:
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                        negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                        negative_anno_file.flush()
                        face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                        negative_count += 1
            except Exception as e:
                traceback.print_exc()


    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()