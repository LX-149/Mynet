import paddle
import paddle.nn.functional as F
from math import exp


def ssim_loss_vi(fused_result, input_vi):
    ssim_loss = ssim(fused_result, input_vi)
    return ssim_loss


def ssim_loss_ir(fused_result, input_ir):
    ssim_loss_ir = ssim(fused_result, input_ir)
    return ssim_loss_ir


def sf_loss_vi(fused_result, input_vi):
    SF_loss = paddle.norm(sf(fused_result) - sf(input_vi))
    return SF_loss


def sf_loss_ir(fused_result, input_ir):
    SF_loss = paddle.norm(sf(fused_result) - sf(input_ir))
    return SF_loss


def sf(f1, kernel_radius=5):
    b, c, h, w = f1.shape

    # Create shift kernels
    r_shift_kernel = paddle.zeros([c, 1, 3, 3])
    r_shift_kernel[:, :, 1, 0] = 1.0  # equivalent to [0, 0, 0], [1, 0, 0], [0, 0, 0]

    b_shift_kernel = paddle.zeros([c, 1, 3, 3])
    b_shift_kernel[:, :, 0, 1] = 1.0  # equivalent to [0, 1, 0], [0, 0, 0], [0, 0, 0]

    # Apply convolutions with the shift kernels
    f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)

    # Calculate gradient
    f1_grad = paddle.pow((f1_r_shift - f1), 2) + paddle.pow((f1_b_shift - f1), 2)

    # Create kernel for spatial filtering
    kernel_size = kernel_radius * 2 + 1
    add_kernel = paddle.ones([c, 1, kernel_size, kernel_size])
    kernel_padding = kernel_size // 2

    # Apply spatial filtering
    f1_sf = paddle.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), axis=1)

    return 1 - f1_sf


def gaussian(window_size, sigma):
    gauss = paddle.to_tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(0)
    _2D_window = paddle.matmul(_1D_window.transpose([1, 0]), _1D_window).unsqueeze([0, 0])
    window = _2D_window.expand([channel, 1, window_size, window_size])
    return window


def ssim(img1, img2, window_size=11, window=None, val_range=None):
    # Check value range
    if val_range is None:
        if paddle.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if paddle.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    _, channel, height, width = img1.shape

    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)

    # Mean of img1 and img2
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    # Square of mean
    mu1_sq = paddle.pow(mu1, 2)
    mu2_sq = paddle.pow(mu2, 2)
    mu1_mu2 = mu1 * mu2

    # Variance of img1 and img2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ret = paddle.mean(ssim_map)

    return 1 - ret