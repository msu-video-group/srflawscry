from pathlib import Path

import gdown
import torch
import torch.nn.functional as F
from torch import nn

from srflawscry.utils.device import preferred_device


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=tuple(),
        dec_blk_nums=tuple(),
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)],
                ),
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2),
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)],
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                ),
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)],
                ),
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs, strict=True):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(
            self.decoders, self.ups, encs[::-1], strict=True
        ):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetRefiner(nn.Module):
    def __init__(self, in_channels, width=64, pretrained=False):  # , device=None):
        super().__init__()

        # self.nafnet = NAFNet(img_channel=in_channels, width=width, # default nafnet \w 36 blocks
        #                     middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])

        self.nafnet = NAFNet(
            img_channel=in_channels,
            width=width,  # nafnet \w 36 blocks
            middle_blk_num=4,
            enc_blk_nums=[1, 1, 1, 25],
            dec_blk_nums=[1, 1, 1, 1],
        )

        if pretrained:
            if width != 32 and width != 64:
                raise RuntimeError(
                    f"Only 32 or 64 width supported for pretrained NAFNet, got {width}"
                )
            weights_dir = Path("weights")
            weights_dir.mkdir(parents=True, exist_ok=True)
            weights_path = weights_dir / f"nafnet_denoise{width}.pth"

            # download if missing (Google Drive ID)
            if not weights_path.exists():
                gdrive_id_w32 = "1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ"
                gdrive_id_w64 = "14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR"

                gdrive_url = f"https://drive.google.com/uc?id={gdrive_id_w32}"
                if width == 64:
                    gdrive_url = f"https://drive.google.com/uc?id={gdrive_id_w64}"
                print(f"Downloading NAFNet weights to '{weights_path}' ...")
                gdown.download(gdrive_url, str(weights_path), quiet=False)

            device = torch.device(preferred_device())
            ck = torch.load(str(weights_path), map_location=device)

            # extract state_dict from common wrapper patterns
            if isinstance(ck, dict):
                if "state_dict" in ck:
                    sd = ck["state_dict"]
                elif "model" in ck:
                    sd = ck["model"]
                elif "params" in ck:
                    sd = ck["params"]
                else:
                    sd = ck
            else:
                sd = ck

            # normalize keys (remove common prefixes)
            normalized_sd = {}
            for k, v in sd.items():
                nk = k
                nk = nk.removeprefix("module.")
                # some checkpoints use "nafnet." prefix
                nk = nk.removeprefix("nafnet.")
                normalized_sd[nk] = v

            model_sd = self.nafnet.state_dict()  # target shapes

            adapted = {}
            used_pretrained_keys = []

            for mkey, mval in model_sd.items():
                if mkey in normalized_sd:
                    pval = normalized_sd[mkey]
                    # exact-shape -> copy
                    if pval.shape == mval.shape:
                        adapted[mkey] = pval
                        used_pretrained_keys.append(mkey)
                        continue

                    # try input-channel adaptation for conv weights
                    # detect conv weight by 4D tensor (out_channels, in_channels, kH, kW)
                    if (
                        pval.ndim == 4
                        and mval.ndim == 4
                        and pval.shape[0] == mval.shape[0]
                        and pval.shape[2:] == mval.shape[2:]
                    ):
                        pin = pval.shape[1]
                        min_ch = min(pin, mval.shape[1])

                        # start with zeros and fill compatible parts
                        new = torch.zeros_like(mval, device=device)

                        # copy overlapping channels
                        new[:, :min_ch, :, :] = pval[:, :min_ch, :, :].to(device)

                        if mval.shape[1] > pin:
                            # expand: fill remaining channels with mean over pretrained input channels
                            ch_mean = pval.mean(dim=1, keepdim=True).to(
                                device
                            )  # shape (out,1,k,k)
                            repeat_times = mval.shape[1] - pin
                            new[:, pin:, :, :] = ch_mean.repeat(1, repeat_times, 1, 1)
                            adapted[mkey] = new
                            used_pretrained_keys.append(mkey + " (adapted-in-channels)")
                            continue
                        # pretrained has more channels than model expects -> already copied first min_ch
                        adapted[mkey] = new
                        used_pretrained_keys.append(mkey + " (truncated-in-channels)")
                        continue

                    # try simple broadcasting for 1D/2D params where shapes differ only by leading dimension
                    # (e.g., layernorm weights when width differs) - only copy matching sub-tensor
                    # copy intersection of dimensions if possible
                    try:
                        # compute elementwise intersection slice
                        if pval.ndim == mval.ndim:
                            slices = tuple(
                                slice(0, min(a, b))
                                for a, b in zip(pval.shape, mval.shape, strict=True)
                            )
                            new = mval.clone().to(device)
                            new[slices] = pval[
                                tuple(slice(0, s.stop) for s in slices)
                            ].to(device)
                            adapted[mkey] = new
                            used_pretrained_keys.append(mkey + " (partially-copied)")
                            continue
                    except Exception:
                        pass

                    # otherwise skip this key (shape mismatch)
                    # do not copy mismatched param
                else:
                    # key not present in pretrained - keep model's init (do nothing)
                    pass

            # load adapted parameters
            # create a new state dict merging adapted keys into model_sd
            merged = dict(model_sd)  # copy
            for k, v in adapted.items():
                merged[k] = v

            load_result = self.nafnet.load_state_dict(merged, strict=False)

            # reporting
            missing = getattr(load_result, "missing_keys", None)
            unexpected = getattr(load_result, "unexpected_keys", None)
            if missing is None and unexpected is None:
                # older torch may return None; infer from state_dict comparison
                # report keys we used and count how many model keys were not filled by exact-copy/adaptation
                used_count = len(used_pretrained_keys)
                total_model_keys = len(model_sd)
                print(
                    f"NAFNet: applied {used_count} pretrained tensors to {total_model_keys} model tensors (best-effort)."
                )
            else:
                print(
                    f"NAFNet load result: missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
                )
                if len(missing):
                    print("  example missing:", missing[:10])
                if len(unexpected):
                    print("  example unexpected:", unexpected[:10])
                print(
                    "  applied pretrained keys (examples):", used_pretrained_keys[:20]
                )

            # Done: model is initialized with any compatible pretrained params; incompatible parts remain randomly initialized.

        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)  # convert to mask

    def forward(self, image, reference, coarse_mask):

        if coarse_mask is None:
            coarse_mask = torch.empty()

        if isinstance(coarse_mask, (list, tuple)):
            coarse_mask = torch.cat(coarse_mask, dim=1)

        inp = torch.cat([image, reference, coarse_mask], dim=1)

        output = self.nafnet(inp)

        output = self.final_conv(output)
        output = torch.sigmoid(output)

        return output.squeeze(dim=1)
