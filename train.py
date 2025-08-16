import os
import argparse
from tqdm import tqdm
import pandas as pd
import glob
from collections import OrderedDict
import paddle
import joblib
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.transforms as transforms
from PIL import Image
from paddle.io import DataLoader, Dataset
import math

# 修改导入路径
from network import (MODEL as net)

from loss import ssim_loss_ir, ssim_loss_vi, sf_loss_ir, sf_loss_vi

use_gpu = paddle.device.is_compiled_with_cuda()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model_name', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    # Modified loss weights as requested: [1, 0.5, 0.0005, 0.00065]
    parser.add_argument('--weight', default=[1, 0.5, 0.0005, 0.00065], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--patch_size', default=120, type=int, help='size of patches for training')
    parser.add_argument('--stride', default=60, type=int, help='stride between patches (for overlap)')
    # 新增参数: 保存间隔，num_heads和window_size
    parser.add_argument('--save_interval', default=1, type=int, help='save model every N epochs')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads')
    parser.add_argument('--window_size', default=1, type=int, help='window size for attention')
    # 新增参数: 训练集路径
    parser.add_argument('--train_ir_dir', default='./datasets/train/ir/', type=str, help='path to IR training images')
    parser.add_argument('--train_vi_dir', default='./datasets/train/vi/', type=str, help='path to VI training images')
    args = parser.parse_args()
    return args


class PatchDataset(Dataset):
    """
    Dataset that returns overlapping patches from images
    """

    def __init__(self, image_folder_dataset, patch_size=120, stride=60, transform=None, is_ir=True,
                 ir_base_path="./datasets/train/ir", vi_base_path="./datasets/train/vi"):
        self.image_folder_dataset = image_folder_dataset
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.is_ir = is_ir

        # 定义基础路径（现在从参数接收）
        self.ir_base_path = ir_base_path
        self.vi_base_path = vi_base_path

        # 确保数据集目录存在
        os.makedirs(self.ir_base_path, exist_ok=True)
        os.makedirs(self.vi_base_path, exist_ok=True)

        # 计算每个图像的patch索引映射
        self.patch_indices = []
        for img_idx, img_path in enumerate(self.image_folder_dataset):
            # 获取原始图像尺寸以计算patch数量
            if self.is_ir:
                img = Image.open(img_path).convert('L')
            else:
                filename = os.path.basename(img_path)
                ir_path = os.path.join(self.ir_base_path, filename)
                img = Image.open(ir_path).convert('L')

            w, h = img.size

            # 计算横向和纵向的patch数量
            num_patches_h = max(1, math.floor((h - self.patch_size) / self.stride) + 1)
            num_patches_w = max(1, math.floor((w - self.patch_size) / self.stride) + 1)

            # 为每个patch创建索引映射
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    self.patch_indices.append((img_idx, i, j))

    def __getitem__(self, index):
        # 从索引获取图像索引和patch位置
        img_idx, i, j = self.patch_indices[index]
        image_path = self.image_folder_dataset[img_idx]
        filename = os.path.basename(image_path)

        # 获取IR和VI图像路径
        if self.is_ir:
            ir_path = image_path
            vi_path = os.path.join(self.vi_base_path, filename)
        else:
            vi_path = image_path
            ir_path = os.path.join(self.ir_base_path, filename)

        # 加载图像
        ir_img = Image.open(ir_path).convert('L')
        vi_img = Image.open(vi_path).convert('L')

        # 计算patch的左上角坐标
        top = i * self.stride
        left = j * self.stride

        # 如果patch会超出图像边界，调整为贴边
        w, h = ir_img.size
        if top + self.patch_size > h:
            top = h - self.patch_size
        if left + self.patch_size > w:
            left = w - self.patch_size

        # 裁剪patch
        ir_patch = ir_img.crop((left, top, left + self.patch_size, top + self.patch_size))
        vi_patch = vi_img.crop((left, top, left + self.patch_size, top + self.patch_size))

        # 转换为张量
        if self.transform is not None:
            ir_patch = self.transform(ir_patch)
            vi_patch = self.transform(vi_patch)
        else:
            tran = transforms.ToTensor()
            ir_patch = tran(ir_patch)
            vi_patch = tran(vi_patch)

        return ir_patch, vi_patch

    def __len__(self):
        return len(self.patch_indices)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion_ssim_ir, criterion_ssim_vi, criterion_sf_ir,
          criterion_sf_vi, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_ssim_ir = AverageMeter()
    losses_ssim_vi = AverageMeter()
    losses_sf_ir = AverageMeter()
    losses_sf_vi = AverageMeter()
    weight = args.weight
    model.train()

    for i, (ir, vi) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if use_gpu:
            ir = paddle.to_tensor(ir)
            vi = paddle.to_tensor(vi)

        out = model(ir, vi)

        loss_ssim_ir = weight[0] * criterion_ssim_ir(out, ir)
        loss_ssim_vi = weight[1] * criterion_ssim_vi(out, vi)
        loss_sf_ir = weight[2] * criterion_sf_ir(out, ir)
        loss_sf_vi = weight[3] * criterion_sf_vi(out, vi)
        loss = loss_ssim_ir + loss_ssim_vi + loss_sf_ir + loss_sf_vi

        losses.update(loss.item(), ir.shape[0])
        losses_ssim_ir.update(loss_ssim_ir.item(), ir.shape[0])
        losses_ssim_vi.update(loss_ssim_vi.item(), ir.shape[0])
        losses_sf_ir.update(loss_sf_ir.item(), ir.shape[0])
        losses_sf_vi.update(loss_sf_vi.item(), ir.shape[0])

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ssim_ir', losses_ssim_ir.avg),
        ('loss_ssim_vi', losses_ssim_vi.avg),
        ('loss_sf_ir', losses_sf_ir.avg),
        ('loss_sf_vi', losses_sf_vi.avg),
    ])
    return log


def main():
    args = parse_args()

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)
    paddle.device.set_device('gpu' if use_gpu else 'cpu')

    # 使用命令行参数设置的路径
    training_dir_ir = os.path.join(args.train_ir_dir, '*')
    folder_dataset_train_ir = glob.glob(training_dir_ir)
    training_dir_vi = os.path.join(args.train_vi_dir, '*')
    folder_dataset_train_vi = glob.glob(training_dir_vi)

    # 打印数据集路径信息
    print(f"IR训练图像路径: {args.train_ir_dir}")
    print(f"VI训练图像路径: {args.train_vi_dir}")
    print(f"找到IR图像: {len(folder_dataset_train_ir)}张")
    print(f"找到VI图像: {len(folder_dataset_train_vi)}张")

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 使用新的PatchDataset类，并传入路径参数
    dataset_train_ir = PatchDataset(
        image_folder_dataset=folder_dataset_train_ir,
        patch_size=args.patch_size,
        stride=args.stride,
        transform=transform_train,
        is_ir=True,
        ir_base_path=args.train_ir_dir.rstrip('/'),
        vi_base_path=args.train_vi_dir.rstrip('/')
    )

    # 只需要一个数据加载器
    train_loader = DataLoader(
        dataset_train_ir,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True
    )

    # 使用正确的参数初始化模型，并传入命令行参数中的num_heads和window_size
    model = net(in_channel=1, num_heads=args.num_heads, window_size=args.window_size)

    if use_gpu:
        # 在paddle中不需要显式调用cuda()
        pass

    criterion_ssim_ir = ssim_loss_ir
    criterion_ssim_vi = ssim_loss_vi
    criterion_sf_ir = sf_loss_ir
    criterion_sf_vi = sf_loss_vi

    optimizer = optim.Adam(
        parameters=model.parameters(),
        learning_rate=args.lr,
        beta1=args.betas[0],
        beta2=args.betas[1],
        epsilon=args.eps
    )

    log = pd.DataFrame(
        index=[],
        columns=[
            'epoch',
            'loss',
            'loss_ssim_ir',
            'loss_ssim_vi',
            'loss_sf_ir',
            'loss_sf_vi',
        ]
    )

    # 打印数据集信息
    print(f"训练数据集包含 {len(dataset_train_ir)} 个补丁")

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

        train_log = train(
            args,
            train_loader,
            model,
            criterion_ssim_ir,
            criterion_ssim_vi,
            criterion_sf_ir,
            criterion_sf_vi,
            optimizer,
            epoch
        )

        print('loss: %.4f - loss_ssim_ir: %.4f - loss_ssim_vi: %.4f - loss_sf_ir: %.4f- loss_sf_vi: %.4f '
              % (
                  train_log['loss'],
                  train_log['loss_ssim_ir'],
                  train_log['loss_ssim_vi'],
                  train_log['loss_sf_ir'],
                  train_log['loss_sf_vi'],
              ))

        tmp = pd.Series([
            epoch + 1,
            train_log['loss'],
            train_log['loss_ssim_ir'],
            train_log['loss_ssim_vi'],
            train_log['loss_sf_ir'],
            train_log['loss_sf_vi'],
        ], index=['epoch', 'loss', 'loss_ssim_ir', 'loss_ssim_vi', 'loss_sf_ir', 'loss_sf_vi'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' % args.name, index=False)

        # 使用新参数save_interval来决定模型保存频率
        if (epoch + 1) % args.save_interval == 0:
            paddle.save(model.state_dict(), 'models/%s/model_%d.pdparams' % (args.name, epoch + 1))


if __name__ == '__main__':
    main()