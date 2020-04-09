import os
import sys
import caffe
import argparse
import numpy as np
import scipy.misc
import scipy.io
from PIL import Image

from scipyold import imresize
from util import *
from cityscapes import cityscapes
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--cityscapes_dir", type=str, required=True, help="Path to the original cityscapes dataset")
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--output_dir", type=str, required=True, help="Where to save the evaluation results")
parser.add_argument("--caffemodel_dir", type=str, default='./scripts/eval_cityscapes/caffemodel/',
                    help="Where the FCN-8s caffemodel stored")
parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu id to use")
parser.add_argument("--split", type=str, default='val', help="Data split to be evaluated")
parser.add_argument("--save_output_images", type=int, default=0, help="Whether to save the FCN output images")
parser.add_argument("--fcn_input_size", type=int, default=710,
                    help="The (square) size images will be cropped to before they are fed into the FCN")
args = parser.parse_args()


def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_output_images > 0:
        output_image_dir = args.output_dir + 'image_outputs/'
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)
    CS = cityscapes(args.cityscapes_dir)
    n_cl = len(CS.classes)
    label_frames = CS.list_label_frames(args.split)
    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(args.caffemodel_dir + '/deploy.prototxt',
                    args.caffemodel_dir + 'fcn-8s-cityscapes.caffemodel',
                    caffe.TEST)

    hist_perframe = np.zeros((n_cl, n_cl))
    for i, idx in enumerate(label_frames):
        if i % 10 == 0:
            print('Evaluating: %d/%d' % (i, len(label_frames)))
        city = idx.split('_')[0]
        # idx is city_shot_frame
        label = CS.load_label(args.split, city, idx)
        # label = imresize(label, (400, 800))
        # im = imresize(im, (256, 256))
        im_file = args.result_dir + '/' + idx + '_leftImg8bit.png'
        im = np.array(Image.open(im_file))

        # Ground truth: center crop like is done to generated imgs
        size_1 = 286
        size_2 = 256
        cc_d = (size_1 - size_2) // 2
        if im.shape == (1024, 2048, 3):
            im = imresize(im, (size_1, size_1))
            im = im[cc_d:-cc_d, cc_d:-cc_d]
        elif im.shape == (256, 256, 3):
            pass  # im already correct size: do nothing
        else:
            print(f"Unrecognized image size {im.shape}")
            exit(1)

        # Center crop label with same proportion
        cc_d_l_y = int(cc_d * label.shape[1] / size_2)
        cc_d_l_x = int(cc_d * label.shape[2] / size_2)
        label = label[:, cc_d_l_y:-cc_d_l_y, cc_d_l_x:-cc_d_l_x]

        # Scale up to label size
        im = imresize(im, (label.shape[1], label.shape[2]))

        # Crop label and im to fit in GPU memory
        in_size = 700
        im = im[:in_size, :in_size]
        label = label[:, :in_size, :in_size]
        out = segrun(net, CS.preprocess(im))
        hist_perframe += fast_hist(label.flatten(), out.flatten(), n_cl)
        if args.save_output_images > 0:
            label_im = CS.palette(label)
            pred_im = CS.palette(out)
            imageio.imwrite(output_image_dir + '/' + str(i) + '_pred.jpg', pred_im.astype(np.uint8))
            imageio.imwrite(output_image_dir + '/' + str(i) + '_gt.jpg', label_im.astype(np.uint8))
            imageio.imwrite(output_image_dir + '/' + str(i) + '_input.jpg', im.astype(np.uint8))

    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    with open(args.output_dir + '/evaluation_results.txt', 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            while len(cl) < 15:
                cl = cl + ' '
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))


main()