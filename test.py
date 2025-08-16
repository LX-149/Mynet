from PIL import Image
import numpy as np
import os
import paddle
import time
import imageio
import paddle.vision.transforms as transforms
import glob
import argparse
import cv2
# 修改导入路径
from network import MODEL as net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_gpu = paddle.device.is_compiled_with_cuda()




def rgb_to_ycrcb(rgb_img):
    """将RGB图像转换为YCrCb"""
    return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2YCrCb)


def ycrcb_to_rgb(ycrcb_img):
    """将YCrCb图像转换为RGB"""
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)


def load_and_preprocess_image(image_path):
    """加载图像并预处理"""
    img = Image.open(image_path)

    # 如果是单通道图像，直接转换为灰度
    if img.mode == 'L':
        return img, None, None, False

    # 如果是三通道图像，转换为YCrCb
    if img.mode == 'RGB' or img.mode == 'RGBA':
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        ycrcb = rgb_to_ycrcb(img)
        y_channel = Image.fromarray(ycrcb[:, :, 0])  # Y通道
        cr_channel = ycrcb[:, :, 1]  # Cr通道
        cb_channel = ycrcb[:, :, 2]  # Cb通道

        return y_channel, cr_channel, cb_channel, True

    # 其他情况转换为灰度
    return img.convert('L'), None, None, False
def parse_args():
    parser = argparse.ArgumentParser(description='图像融合测试程序')
    parser.add_argument('--model_path', default='./models/model_name/model_1.pdparams',
                        type=str, help='模型权重文件路径')

    parser.add_argument('--model_name', default='',
                        type=str, help='模型名称，用于查找模型文件（如指定则会覆盖model_path）')
    parser.add_argument('--model_epoch', default=1,
                        type=int, help='要加载的模型训练轮次（仅当指定model_name时有效）')
    parser.add_argument('--models_dir', default='./models/',
                        type=str, help='模型存储目录（仅当指定model_name时有效）')

    parser.add_argument('--test_ir_dir', default='./datasets/test/ir/',
                        type=str, help='IR测试图像路径')
    parser.add_argument('--test_vi_dir', default='./datasets/test/vi/',
                        type=str, help='VI测试图像路径')
    parser.add_argument('--output_dir', default='./fusion_result/',
                        type=str, help='融合结果保存路径')
    parser.add_argument('--num_heads', default=8, type=int, help='注意力头数量')
    parser.add_argument('--window_size', default=1, type=int, help='窗口大小')
    return parser.parse_args()


args = parse_args()

# 使用正确的参数初始化模型，并使用命令行参数中的num_heads和window_size
model = net(in_channel=1, num_heads=args.num_heads, window_size=args.window_size)

# 确定模型路径：如果指定了model_name，则根据model_name和model_epoch构建路径
if args.model_name:
    model_path = os.path.join(args.models_dir, args.model_name, f"model_{args.model_epoch}.pdparams")
    print(f"根据model_name和model_epoch构建模型路径: {model_path}")
else:
    model_path = args.model_path
    print(f"使用命令行指定的模型路径: {model_path}")

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误：模型文件不存在: {model_path}")
    print(f"请检查以下内容：")
    print(f"1. 是否指定了正确的模型路径 --model_path")
    print(f"2. 如果使用了--model_name, 请检查模型名称和轮次是否正确")
    print(f"3. 模型文件是否已经训练生成")
    exit(1)
else:
    print(f"找到模型文件: {model_path}")

# 加载模型
try:
    if use_gpu:
        paddle.device.set_device('gpu')
    else:
        paddle.device.set_device('cpu')

    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
    print(f"模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)

# 打印测试配置信息
print("\n测试配置信息:")
print(f"设备: {'GPU' if use_gpu else 'CPU'}")
print(f"IR测试图像路径: {args.test_ir_dir}")
print(f"VI测试图像路径: {args.test_vi_dir}")
print(f"融合结果输出路径: {args.output_dir}")
print(f"注意力头数量: {args.num_heads}")
print(f"窗口大小: {args.window_size}")


def batch_fusion():
    """
    对测试集中的所有图像进行批量推理
    从vi文件夹获取所有可见光图像，并在ir文件夹中获取对应的红外图像
    """
    # 使用命令行参数设置的路径
    test_vi_dir = args.test_vi_dir
    test_ir_dir = args.test_ir_dir
    output_dir = args.output_dir

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有可见光图像路径
    vi_images = glob.glob(os.path.join(test_vi_dir, '*.*'))

    # 记录处理时间
    tic = time.time()

    # 计数处理的图像对
    processed_images = 0

    print(f"找到 {len(vi_images)} 张可见光图像待处理")

    for vi_path in vi_images:
        filename = os.path.basename(vi_path)
        base_filename, ext = os.path.splitext(filename)
        ir_path = os.path.join(test_ir_dir, filename)

        # 加载和预处理图像
        vi_y, vi_cr, vi_cb, vi_is_color = load_and_preprocess_image(vi_path)
        ir_y, ir_cr, ir_cb, ir_is_color = load_and_preprocess_image(ir_path)

        # 转换为张量（只处理Y通道或灰度通道）
        transform = transforms.ToTensor()
        vi_tensor = transform(vi_y).unsqueeze(0)
        ir_tensor = transform(ir_y).unsqueeze(0)

        # 模型推理
        model.eval()
        with paddle.no_grad():
            fused_result = model(paddle.to_tensor(ir_tensor), paddle.to_tensor(vi_tensor))

        # 后处理
        fused_y = np.squeeze(fused_result.numpy())
        fused_y = (fused_y * 255).astype(np.uint8)

        # 如果原图是彩色的，重新组合通道
        if vi_is_color or ir_is_color:
            # 选择保留哪个图像的色度信息（这里选择VI图像的色度）
            if vi_is_color:
                cr_to_use, cb_to_use = vi_cr, vi_cb
            else:
                cr_to_use, cb_to_use = ir_cr, ir_cb

            # 重新组合YCrCb
            fused_ycrcb = np.stack([fused_y, cr_to_use, cb_to_use], axis=2)

            # 转换回RGB
            fused_rgb = ycrcb_to_rgb(fused_ycrcb)

            # 保存为彩色图像
            output_path = os.path.join(output_dir, f"fused_{base_filename}{ext}")
            imageio.imwrite(output_path, fused_rgb)
        else:
            # 保存为灰度图像
            output_path = os.path.join(output_dir, f"fused_{base_filename}{ext}")
            imageio.imwrite(output_path, fused_y)

        processed_images += 1
        print(f"已处理: {filename}")

    toc = time.time()
    processing_time = toc - tic

    print(f"批量融合完成。成功融合 {processed_images} 对图像。")
    print(f"总处理时间: {processing_time:.2f} 秒")
    if processed_images > 0:
        print(f"平均每对图像处理时间: {processing_time / processed_images:.2f} 秒")


def single_fusion():
    """
    原始的单图像对处理函数，保留以便兼容
    现在支持从命令行参数获取路径
    """
    tic = time.time()
    for num in range(1):
        # 使用示例文件名，但从命令行参数获取目录路径
        test_file = 'IR_002.bmp'
        path1 = os.path.join(args.test_ir_dir, test_file)
        vis_file = 'VIS_002.bmp'
        path2 = os.path.join(args.test_vi_dir, vis_file)

        # 检查文件是否存在
        if not os.path.exists(path1):
            print(f"错误: IR图像文件不存在: {path1}")
            return
        if not os.path.exists(path2):
            print(f"错误: VI图像文件不存在: {path2}")
            return

        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')

        img1_org = img1
        img2_org = img2
        tran = transforms.ToTensor()
        img1_org = tran(img1_org)
        img2_org = tran(img2_org)

        img1_org = paddle.to_tensor(img1_org).unsqueeze(0)
        img2_org = paddle.to_tensor(img2_org).unsqueeze(0)

        model.eval()
        with paddle.no_grad():
            out = model(img1_org, img2_org)

        d = np.squeeze(out.numpy())
        result = (d * 255).astype(np.uint8)

        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{num}.bmp")
        imageio.imwrite(output_path, result)

    toc = time.time()
    print('end {}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))


if __name__ == '__main__':
    # 使用批量处理函数
    batch_fusion()