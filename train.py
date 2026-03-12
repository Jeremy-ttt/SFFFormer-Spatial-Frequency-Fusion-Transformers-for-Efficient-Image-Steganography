import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from thop import profile
import timm
import timm.scheduler
import kornia
# 导入配置和数据集
import config
from datasets import *
from skimage.metrics import structural_similarity as calculate_ssim
from model_SFFFormer import SFFFormer


# ==========================================
# Loss Functions
# ==========================================

class L1_Charbonnier_loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        X = X.float()
        Y = Y.float()
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class Restrict_Loss(nn.Module):
    def __init__(self):
        super(Restrict_Loss, self).__init__()

    def forward(self, X):
        X = X.float()
        count1 = torch.sum(X > 1)
        count0 = torch.sum(X < 0)
        if count1 == 0: count1 = 1.0
        if count0 == 0: count0 = 1.0

        one = torch.ones_like(X)
        zero = torch.zeros_like(X)

        X_one = torch.where(X <= 1, one, X)
        X_zero = torch.where(X >= 0, zero, X)

        diff_one = X_one - one
        diff_zero = zero - X_zero

        loss = torch.sum(0.5 * (diff_one ** 2)) / count1 + \
               torch.sum(0.5 * (diff_zero ** 2)) / count0
        return loss


# ==========================================
# Helper Functions
# ==========================================

def safe_psnr(img1, img2):
    """
    img1, img2: numpy array, range [0, 1]
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)

    if mse < 1e-10:
        return 100.0

    return 10 * math.log10(1.0 / mse)


# ==========================================
# Main Training Function
# ==========================================

def main():
    args = config.Args()
    # 随机种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 路径设置
    model_version_name = args.model_name
    save_path = os.path.join(args.path, 'checkpoint', model_version_name)
    log_path = os.path.join(args.path, 'tensorboard_log', args.model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    writer = SummaryWriter(log_path)

    print(f"=== 开始训练 SFFFormer (FP32 Mode) ===")

    # 1. 模型初始化
    encoder = SFFFormer(in_channels=6, out_channels=3, dim=16).cuda()
    decoder = SFFFormer(in_channels=3, out_channels=3, dim=16).cuda()

    # 2. FLOPs 验证 (训练前可选)
    try:
        dummy_in_enc = torch.randn(1, 6, 256, 256).cuda()
        dummy_in_dec = torch.randn(1, 3, 256, 256).cuda()
        enc_flops, enc_params = profile(encoder, (dummy_in_enc,), verbose=False)
        dec_flops, dec_params = profile(decoder, (dummy_in_dec,), verbose=False)
        print(f"Encoder: {enc_params} Params, {enc_flops} FLOPs")
        print(f"Decoder: {dec_params} Params, {dec_flops} FLOPs")
    except:
        pass

    # # 编译模型 (可选，如果报错可以注释掉)
    # try:
    #     print("正在编译模型以进行加速 (torch.compile)...")
    #     encoder = torch.compile(encoder, mode="reduce-overhead")
    #     decoder = torch.compile(decoder, mode="reduce-overhead")
    #     print("模型编译完成。")
    # except Exception as e:
    #     print(f"torch.compile 编译失败或当前环境不支持: {e}")

    # 3. 加载断点
    if args.train_next != 0:
        model_path = os.path.join(save_path, 'model_checkpoint_%.5i.pt' % args.train_next)
        print(f"Resuming from: {model_path}")
        checkpoint = torch.load(model_path)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])

    # 4. 优化器
    params = list(encoder.parameters()) + list(decoder.parameters())
    optim_G = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    if args.train_next != 0 and 'opt' in checkpoint:
        optim_G.load_state_dict(checkpoint['opt'])


    scheduler = timm.scheduler.CosineLRScheduler(
        optimizer=optim_G,
        t_initial=args.epochs,
        lr_min=1e-6,
        warmup_t=args.warm_up_epoch,
        warmup_lr_init=args.warm_up_lr_init
    )

    criterion_L1 = L1_Charbonnier_loss().cuda()
    criterion_restrict = Restrict_Loss().cuda()
    
    # 5. 训练循环
    num_batches = min(len(DIV2K_train_cover_loader), len(DIV2K_train_secret_loader))
    # 强制检查当前 Loader 里的真实情况
    print(f"DEBUG: Cover 文件夹实际图片数: {len(DIV2K_train_cover_loader.dataset)}")
    print(f"DEBUG: Secret 文件夹实际图片数: {len(DIV2K_train_secret_loader.dataset)}")
    print(f"DEBUG: Batch Size: {DIV2K_train_cover_loader.batch_size}")
    print(f"DEBUG: 计算出的 num_batches: {num_batches}")
    for epoch in range(args.epochs):
        current_epoch = epoch + args.train_next
        scheduler.step(current_epoch)

        encoder.train()
        decoder.train()

        epoch_losses = []
        cover_iter = iter(DIV2K_train_cover_loader)
        secret_iter = iter(DIV2K_train_secret_loader)

        for i in range(num_batches):
            try:
                cover = next(cover_iter).cuda()
                secret = next(secret_iter).cuda()
            except StopIteration:
                break

            optim_G.zero_grad()

            # --- Forward (Standard FP32) ---
            # 移除了 with torch.amp.autocast('cuda'):
            input_msg = torch.cat([cover, secret], dim=1)

            # Encode
            stego = encoder(input_msg)
            if args.norm_train == 'clamp':
                stego_input = torch.clamp(stego, 0, 1)
            else:
                stego_input = stego

            # Decode
            restored_secret = decoder(stego_input)

            loss_conceal = criterion_L1(cover, stego)
            loss_reveal = criterion_L1(secret, restored_secret)
            g_ssim_loss_on_encoder = 2*kornia.losses.ssim_loss(stego, cover, window_size=5, reduction="mean")
            g_ssim_loss_on_decoder = 2*kornia.losses.ssim_loss(restored_secret, secret, window_size=5, reduction="mean")
            
            total_loss = loss_conceal + loss_reveal + criterion_restrict(stego)+ 0.02*g_ssim_loss_on_encoder + 0.01*g_ssim_loss_on_decoder
            # total_loss = loss_conceal + loss_reveal + criterion_restrict(stego)
            # --- Backward (Standard) ---
            # 移除了 scaler.scale(...)
            total_loss.backward()

            # 移除了 scaler.unscale_(optim_G)，直接进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=2.0)

            # 移除了 scaler.step(optim_G) 和 scaler.update()
            optim_G.step()

            epoch_losses.append(total_loss.item())

            # 每 10 batch 打印一次
            if i % 50 == 0:
                print(f"Epoch {current_epoch} [{i}/{num_batches}] "
                      f"Loss: {total_loss.item():.4f}", flush=True)

        # Log Epoch Loss
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {current_epoch} | Loss: {mean_loss:.6f} | LR: {optim_G.param_groups[0]['lr']:.2e}")
        writer.add_scalar("Train/Loss", mean_loss, current_epoch)

        # ==========================================
        # 6. 验证 (Validation)
        # ==========================================
        if current_epoch % args.val_freq == 0:
            encoder.eval()
            decoder.eval()

            psnr_c_list, psnr_s_list = [], []
            ssim_c_list, ssim_s_list = [], []

            with torch.no_grad():
                val_loader = zip(DIV2K_val_cover_loader, DIV2K_val_secret_loader)

                for i, (cover, secret) in enumerate(val_loader):
                    cover = cover.cuda()
                    secret = secret.cuda()

                    # 移除了 with torch.amp.autocast('cuda'):
                    input_msg = torch.cat([cover, secret], dim=1)
                    stego = encoder(input_msg)
                    stego = torch.clamp(stego, 0, 1)
                    restored = decoder(stego)
                    restored = torch.clamp(restored, 0, 1)

                    # 转 Numpy (H, W, C) 用于指标计算
                    cover_np = cover.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
                    stego_np = stego.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
                    secret_np = secret.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
                    restored_np = restored.detach().float().cpu().numpy().transpose(0, 2, 3, 1)

                    # Metrics Calculation
                    for b in range(cover_np.shape[0]):
                        # 计算 PSNR
                        p_c = safe_psnr(cover_np[b].transpose(2, 0, 1), stego_np[b].transpose(2, 0, 1))
                        p_s = safe_psnr(secret_np[b].transpose(2, 0, 1), restored_np[b].transpose(2, 0, 1))
                        psnr_c_list.append(p_c)
                        psnr_s_list.append(p_s)

                        # 计算 SSIM
                        s_c = calculate_ssim(cover_np[b], stego_np[b], data_range=1.0, channel_axis=-1)
                        s_s = calculate_ssim(secret_np[b], restored_np[b], data_range=1.0, channel_axis=-1)
                        ssim_c_list.append(s_c)
                        ssim_s_list.append(s_s)

                    # # Log Images
                    # if i == 0:
                    #     writer.add_images('Val/Stego', stego, current_epoch)
                    #     writer.add_images('Val/Secret_Rev', restored, current_epoch)

            # Average & Sanitize
            avg_psnr_c = np.mean(psnr_c_list)
            avg_psnr_s = np.mean(psnr_s_list)
            avg_ssim_c = np.mean(ssim_c_list)
            avg_ssim_s = np.mean(ssim_s_list)

            print(f"VALIDATION | Epoch {current_epoch} | "
                  f"Cover PSNR: {avg_psnr_c:.2f} SSIM: {avg_ssim_c:.4f} | "
                  f"Secret PSNR: {avg_psnr_s:.2f} SSIM: {avg_ssim_s:.4f}")

            # 写入 Tensorboard
            writer.add_scalar("Val/PSNR_Cover", avg_psnr_c, current_epoch)
            writer.add_scalar("Val/PSNR_Secret", avg_psnr_s, current_epoch)
            writer.add_scalar("Val/SSIM_Cover", avg_ssim_c, current_epoch)
            writer.add_scalar("Val/SSIM_Secret", avg_ssim_s, current_epoch)

        # ==========================================
        # 7. 保存
        # ==========================================
        if current_epoch % args.save_freq == 0:
            torch.save({
                'opt': optim_G.state_dict(),
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                # 移除了 'scaler': scaler.state_dict()
            }, os.path.join(save_path, f'model_checkpoint_{current_epoch:05d}.pt'))

    print("训练完成！")
    writer.close()


if __name__ == '__main__':
    main()