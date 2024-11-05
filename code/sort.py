# -*- coding:utf-8 -*-
"""
@file name  : sort.py
@author     : Ma sihai
@date       : 2024-7-02
@brief      : 分类脚本
"""
import time
import os
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import platform
import shutil

if platform.system() == 'Linux':
    matplotlib.use('Agg')

classes =  ["normal", "边缘扩散", "边缘圆环", "抽检正常", "规则类型", "均匀分布，数量较多", "块状图片", "全损", "圆弧状", "直线", "中心环状"]

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification", add_help=add_help)
    parser.add_argument("--img-path", default=r"D:\map\4月map", type=str, help="dataset path")
    parser.add_argument("--ckpt-path", default=r"D:\pyspaces\recz\Result\2024-07-20_00-37-01\checkpoint_49.pth", type=str, help="ckpt path")
    parser.add_argument("--model", default="resnet50", type=str,
                        help="model name; resnet50/convnext/convnext-tiny")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")

    return parser

def get_predict(path_img, transform, cuda_cpu, model):
    #img = Image.open(path_img).convert('L')
    img = Image.open(path_img)
    img_tensor = transform(img)
    img_tensor = img_tensor.to(cuda_cpu)
    predict = None

    with torch.no_grad():
        ss = time.time()
        for i in range(1):
            s = time.time()
            img_tensor_batch = img_tensor.unsqueeze(dim=0)
            bs = 128
            img_tensor_batch = img_tensor_batch.repeat(bs, 1, 1, 1)  # 128 or 100 or 1
            outputs = model(img_tensor_batch)
            outputs_prob = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs_prob.data, 1)
            pred_idx = predicted.cpu().data.numpy()[0]
            time_c = time.time() - s
            #print('\r', 'model predict: {},  speed: {:.4f} s/batch, Throughput: {:.0f} frame/s'.format(classes[pred_idx], time_c, 1*bs/time_c), end='\n')
            predict = classes[pred_idx]

    return predict    

def move_file(root, file, subdir):
    fulldir = os.path.join(root, subdir)
    if not os.path.exists(fulldir):
        os.makedirs(fulldir,  exist_ok=False)
    
    fullfile = os.path.join(root, file)
    shutil.move(fullfile, fulldir)

def main(args):
    device = args.device
    path_img = args.img_path
    result_dir = args.output_dir
    # ------------------------------------ step1: img preprocess ------------------------------------

    normMean = [0.5]
    normStd = [0.5]
    input_size = (200, 200)
    normTransform = transforms.Normalize(normMean, normStd)

    valid_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(200),
        transforms.ToTensor(),
        normTransform
    ])

    # ------------------------------------ step2: model init ------------------------------------
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif args.model == 'convnext':
        model = torchvision.models.convnext_base(pretrained=True)
    elif args.model == 'convnext-tiny':
        model = torchvision.models.convnext_tiny(pretrained=True)
    else:
        print('unexpect model --> :{}'.format(args.model))

    model_name = model._get_name()

    if 'ResNet' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        model.conv1 = nn.Conv2d(4, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features  # 替换最后一层
        model.fc = nn.Linear(2048, 11)
    elif 'ConvNeXt' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        num_kernel = 128 if args.model == 'convnext' else 96
        model.features[0][0] = nn.Conv2d(1, num_kernel, (4, 4), stride=(4, 4))  # convnext base/ tiny
        # 替换最后一层
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 2)

    state_dict = torch.load(args.ckpt_path)
    model_sate_dict = state_dict['model_state_dict']
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    model.to(device)
    model.eval()

    for file in os.listdir(path_img):
        if os.path.isfile(os.path.join(path_img, file)):
            if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg"):
                predict = get_predict(os.path.join(path_img, file), valid_transform, device, model)
                if predict is not None:
                    print('\r', '{} predict: {}'.format(file, predict))
                    move_file(path_img, file, predict)    # 移动到相关目录
                    

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)