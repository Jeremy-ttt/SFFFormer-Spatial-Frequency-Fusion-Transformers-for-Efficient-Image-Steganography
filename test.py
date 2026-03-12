import torch
import torch.optim
from thop import profile
import os
import time
import random
from model_SFFFormer import SFFFormer
from critic import *
from datasets import DIV2K_test_cover_loader, DIV2K_test_secret_loader


# from datasets import COCO_test_cover_loader, COCO_test_secret_loader # 根据需要取消注释

# 辅助函数：计算指标 (假设这些函数在 utils 或当前环境中定义，如果未定义需补充)
# from utils import calculate_psnr_skimage, calculate_ssim_skimage, calculate_mae, calculate_rmse

class Args:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_model = 'stegTransX_Lite'  # 更新模型名称
        self.num_secret = 1
        self.dim = 16

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # 确保卷积算法的可重复性（虽然会稍微牺牲一点速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42) # 在最开始调用
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    USE_MULTI_GPU = False

    if USE_MULTI_GPU and torch.cuda.device_count() > 1:
        MULTI_GPU = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        device_ids = [0, 1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 1. 模型初始化 (修改部分)
    # ==========================================
    print(f"Initializing {args.use_model}...")

    # Encoder:
    # 输入 = Cover(3) + Secret(3) = 6 channels
    # 输出 = Stego(3)
    encoder = SFFFormer(
        in_channels=(args.num_secret + 1) * 3,
        out_channels=3,
        dim=args.dim,

    )

    # Decoder:
    # 输入 = Stego(3)
    # 输出 = Secret(3) (假设 num_secret=1)
    decoder = SFFFormer(
        in_channels=3,
        out_channels=args.num_secret * 3,
        dim=args.dim,
    )

    encoder.cuda()
    decoder.cuda()

    # ==========================================
    # 2. 加载模型 (修改部分)
    # ==========================================
    # 注意：旧的 StegFormer 权重与 stegTransX_Lite 结构不匹配。
    # 请在训练新模型后，指向新的 .pt 文件。
    model_path = r'checkpoint\best.pt'
    state_dicts = torch.load(model_path)

    def clean_state_dict(obj):
        new_obj = {}
        for k, v in obj.items():
            # 1. 处理 torch.compile 的前缀
            name = k.replace('_orig_mod.', '')

            # 2. 过滤掉 thop 注入的统计键
            if "total_ops" in name or "total_params" in name:
                continue

            new_obj[name] = v
        return new_obj

    # # 加载时使用 strict=True 来验证是否真的只剩下核心权重了
    encoder.load_state_dict(clean_state_dict(state_dicts['encoder']), strict=True)
    decoder.load_state_dict(clean_state_dict(state_dicts['decoder']), strict=True)

    # print("✅ 权重已成功加载（已处理 _orig_mod 前缀）")
    # encoder.load_state_dict(state_dicts['encoder'], strict=False)
    # decoder.load_state_dict(state_dicts['decoder'], strict=False)
    print("Model weights loaded successfully.")


    # 数据并行
    if USE_MULTI_GPU:
        encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)
        decoder = torch.nn.DataParallel(decoder, device_ids=device_ids)
    encoder.to(device)
    decoder.to(device)

    # ==========================================
    # 3. 计算参数量 (THOP)
    # ==========================================
    print("Calculating FLOPs and Params...")
    try:
        with torch.no_grad():

            # Encoder 输入是 6 通道
            test_encoder_input = torch.randn(1, (args.num_secret + 1) * 3, 256, 256).to(device)
            # Decoder 输入是 3 通道
            test_decoder_input = torch.randn(1, 3, 256, 256).to(device)

            encoder_mac, encoder_params = profile(encoder, inputs=(test_encoder_input,), verbose=False)
            decoder_mac, decoder_params = profile(decoder, inputs=(test_decoder_input,), verbose=False)

            print(f"Encoder: FLOPs={encoder_mac * 2 / 1e9:.3f}G, Params={encoder_params / 1e6:.3f}M")
            print(f"Decoder: FLOPs={decoder_mac * 2 / 1e9:.3f}G, Params={decoder_params / 1e6:.3f}M")
    except Exception as e:
        print(f"THOP calculation failed: {e}")

    # ==========================================
    # 4. 推理/验证循环
    # ==========================================
    print('Start Validation (Clamp version)...')

    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        psnr_secret_y = []
        psnr_cover_y = []
        ssim_secret = []
        ssim_cover = []
        rmse_cover = []
        rmse_secret = []
        mae_cover = []
        mae_secret = []
        encode_times = []
        decode_times = []

        # 在 DIV2K 上测试
        i = 0
        # 注意：原本的 break 逻辑会导致只跑 1 个 batch，这里我注释掉了外层多余的循环，仅保留 loader 循环
        for (cover, secret) in zip(DIV2K_test_cover_loader, DIV2K_test_secret_loader):
            # if i!= 91:
            #     i += 1
            #     continue
            cover = cover.to(device)
            secret = secret.to(device)

            # 1. Encode
            # 拼接 Cover 和 Secret -> [Batch, 6, H, W]
            msg = torch.cat([cover, secret], 1)

            start_time = time.perf_counter()
            encode_img = encoder(msg)
            end_time = time.perf_counter()

            encode_time = end_time - start_time
            encode_times.append(encode_time)

            # 截断到 [0, 1] 模拟真实图像保存
            encode_img = torch.clamp(encode_img, 0, 1)

            # 2. Decode
            start_time = time.perf_counter()
            decode_img = decoder(encode_img)
            end_time = time.perf_counter()

            decode_time = end_time - start_time
            decode_times.append(decode_time)

            decode_img = decode_img.clamp(0, 1)

            # 3. 计算指标
            cover = cover.cpu()
            secret = secret.cpu()
            encode_img = encode_img.cpu()
            decode_img = decode_img.cpu()

            # 这里的 calculate_xxx 函数需要你自己保证已导入
            # 如果报错 NameError，请确保你在这个脚本里定义了它们或者 import 了它们
            try:
                # 计算 Y 通道 PSNR
                psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
                psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
                psnr_cover_y.append(psnry_encode_temp)
                psnr_secret_y.append(psnry_decode_temp)

                # 计算 SSIM
                ssim_cover_temp = calculate_ssim_skimage(cover, encode_img)
                ssim_secret_temp = calculate_ssim_skimage(secret, decode_img)
                ssim_cover.append(ssim_cover_temp)
                ssim_secret.append(ssim_secret_temp)

                # 计算 RMSE
                rmse_cover_temp = calculate_rmse(cover, encode_img)
                rmse_secret_temp = calculate_rmse(secret, decode_img)
                rmse_cover.append(rmse_cover_temp)
                rmse_secret.append(rmse_secret_temp)

                # 计算 MAE
                mae_cover_temp = calculate_mae(cover, encode_img)
                mae_secret_temp = calculate_mae(secret, decode_img)
                mae_cover.append(mae_cover_temp)
                mae_secret.append(mae_secret_temp)

                i += 1
                print(f'Item {i} | Encode T: {encode_time:.4f}s | Decode T: {decode_time:.4f}s')
                print(f'   Cover  -> PSNR: {psnry_encode_temp:.2f}, SSIM: {ssim_cover_temp:.4f}')
                print(f'   Secret -> PSNR: {psnry_decode_temp:.2f}, SSIM: {ssim_secret_temp:.4f}')

            except NameError:
                print("Error: Metric calculation functions (calculate_psnr_skimage, etc.) are not defined.")
                break

        # 汇总输出
        if len(psnr_cover_y) > 0:
            print('\n' + '=' * 20 + ' DIV2K RESULTS ' + '=' * 20)
            print(f'Encode Time: {np.sum(encode_times):.4f} s')
            print(f'Decode Time: {np.sum(decode_times):.4f} s')
            print(f'Average Encode Time: {np.mean(encode_times):.4f} s')
            print(f'Average Decode Time: {np.mean(decode_times):.4f} s')

            print(f'Cover Image:')
            print(f'  PSNR: {np.mean(psnr_cover_y):.4f}')
            print(f'  SSIM: {np.mean(ssim_cover):.4f}')
            print(f'  MAE:  {np.mean(mae_cover):.4f}')
            print(f'  RMSE: {np.mean(rmse_cover):.4f}')

            print(f'Secret Image:')
            print(f'  PSNR: {np.mean(psnr_secret_y):.4f}')
            print(f'  SSIM: {np.mean(ssim_secret):.4f}')
            print(f'  MAE:  {np.mean(mae_secret):.4f}')
            print(f'  RMSE: {np.mean(rmse_secret):.4f}')
            print('=' * 55)
        else:
            print("No data processed.")
if __name__ == "__main__":
    main()