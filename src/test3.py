from PIL import Image
import face_recognition
import os

landark_file = str("identity_CelebA.txt")
data_file = r"E:\BaiduNetdiskDownload\CelebA\Img\img_align_celeba"
save_path = r'E:\save_path\identity'
for face_size in [48]:
    landmark_save_dir = os.path.join(save_path, str(face_size),"images")
    if not os.path.exists(landmark_save_dir):
        os.makedirs(landmark_save_dir)

    landark_save_file = os.path.join(save_path,str(face_size),"identity.txt")

    f1 = open(landark_save_file, 'w')

    with open(landark_file, 'r') as f:
        annotations = f.readlines()

    for annotation in annotations:
        annotation = annotation.strip().split(" ")
        annotation = list(filter(bool, annotation))
        image_filename = annotation[0].strip()

        identity = int(annotation[1].strip())

        im_path = os.path.join(data_file, image_filename)
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
            face_resize = pil_image.resize((face_size,face_size),Image.ANTIALIAS)

            save_im_path = os.path.join(landmark_save_dir, image_filename)
            face_resize.save(save_im_path)

            f1.write(image_filename + ' %d\n' % (identity))

            # rs_lefteye_x = (lefteye_x - left) * (face_size)/(right - left)
            # rs_lefteye_y = (lefteye_y - top) * (face_size) / (bottom - top)
            # rs_righteye_x = (righteye_x - left) * (face_size) / (right - left)
            # rs_righteye_y = (righteye_y - top) * (face_size) / (bottom - top)
            # rs_nose_x = (nose_x - left) * (face_size) / (right - left)
            # rs_nose_y = (nose_y - top) * (face_size) / (bottom - top)
            # rs_leftmouth_x = (leftmouth_x - left) * (face_size) / (right - left)
            # rs_leftmouth_y = (leftmouth_y - top) * (face_size) / (bottom - top)
            # rs_rightmouth_x = (rightmouth_x - left) * (face_size) / (right - left)
            # rs_rightmouth_y = (rightmouth_y - top) * (face_size) / (bottom - top)
            #
            # face_image = image[top:bottom, left:right]
            # pil_image = Image.fromarray(face_image)
            # face_resize = pil_image.resize((face_size,face_size),Image.ANTIALIAS)
            #
            # save_im_path = os.path.join(landmark_save_dir, image_filename)
            # face_resize.save(save_im_path)
            # f1.write(image_filename + ' %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (rs_lefteye_x, rs_lefteye_y, rs_righteye_x, rs_righteye_y,
            #                                                                                     rs_nose_x,rs_nose_y,rs_leftmouth_x,rs_leftmouth_y,rs_rightmouth_x,rs_rightmouth_y))
    f1.close();

    # im = Image.open(im_path)
    # im.show(im)
# for (int i = 1; i <= 10; i++)
# list = os.listdir("./")
# for i in range(0, len(list)):
#     imgName = os.path.basename(list[i])
#     if (os.path.splitext(imgName)[1] != ".jpg"): continue
#
#     image = face_recognition.load_image_file(imgName)
#
#     face_locations = face_recognition.face_locations(image)
#
#     for face_location in face_locations:
#
#         # Print the location of each face in this image
#         top, right, bottom, left = face_location
#         # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
#
#         # You can access the actual face itself like this:
#         width = right - left
#         height = bottom - top
#         if (width > height):
#             right -= (width - height)
#         elif (height > width):
#             bottom -= (height - width)
#         face_image = image[top:bottom, left:right]
#         pil_image = Image.fromarray(face_image)
#         pil_image.save('face%s'%imgName)