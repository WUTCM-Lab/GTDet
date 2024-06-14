import torch
import torch.nn as nn
import torch.nn.functional as F
from .ASFF import ASFF

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Gridding_ASFF(nn.Module):
    def __init__(self, level, multiplier=0.5, rfb=False, vis=False, act_cfg=True):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        super(Gridding_ASFF, self).__init__()
        self.attention_map = SpatialAttention()
        self.level = level
        if level == 0:

            self.ASFF_lt = ASFF(level=0, multiplier=0.5)
            self.ASFF_lb = ASFF(level=0, multiplier=0.5)
            self.ASFF_rt = ASFF(level=0, multiplier=0.5)
            self.ASFF_rb = ASFF(level=0, multiplier=0.5)
        elif level == 1:
            self.ASFF = ASFF(level=1, multiplier=0.5)
        elif level == 2:
            self.ASFF = ASFF(level=2, multiplier=0.5)

    def forward(self, x):  # l,m,s
        x_level_0 = x[2]  # 最大特征层，1024
        x_level_1 = x[1]  # 中间特征层，512
        x_level_2 = x[0]  # 最小特征层，256
        if self.level == 0:
            attention_map = self.attention_map(x_level_0)
            max_value = attention_map.max()
            min_value = attention_map.min()
            threshold_value = min_value + 0.75 * (max_value - min_value)
            attention_map[attention_map < threshold_value] = 0
            centroid_x, centroid_y = self.get_centroid(attention_map)
            attention_patch_l = attention_map[:, :, :centroid_x, :]
            attention_patch_r = attention_map[:, :, centroid_x:, :]
            centroid_x_l, centroid_y_l = self.get_centroid(attention_patch_l)
            centroid_x_r, centroid_y_r = self.get_centroid(attention_patch_r)
            feat_patch_lt_0 = x_level_0[:, :, :centroid_x, :centroid_y_l]
            feat_patch_lb_0 = x_level_0[:, :, centroid_x:, :centroid_y_r]
            feat_patch_rt_0 = x_level_0[:, :, :centroid_x, centroid_y_l:]
            feat_patch_rb_0 = x_level_0[:, :, centroid_x:, centroid_y_r:]
            feat_patch_lt_1 = x_level_1[:, :, :2*centroid_x, :2*centroid_y_l]
            feat_patch_lb_1 = x_level_1[:, :, 2*centroid_x:, :2*centroid_y_r]
            feat_patch_rt_1 = x_level_1[:, :, :2*centroid_x, 2*centroid_y_l:]
            feat_patch_rb_1 = x_level_1[:, :, 2*centroid_x:, 2*centroid_y_r:]
            feat_patch_lt_2 = x_level_2[:, :, :4*centroid_x, :4*centroid_y_l]
            feat_patch_lb_2 = x_level_2[:, :, 4*centroid_x:, :4*centroid_y_r]
            feat_patch_rt_2 = x_level_2[:, :, :4*centroid_x, 4*centroid_y_l:]
            feat_patch_rb_2 = x_level_2[:, :, 4*centroid_x:, 4*centroid_y_r:]
            x_lt = [feat_patch_lt_2, feat_patch_lt_1, feat_patch_lt_0]
            x_lb = [feat_patch_lb_2, feat_patch_lb_1, feat_patch_lb_0]
            x_rt = [feat_patch_rt_2, feat_patch_rt_1, feat_patch_rt_0]
            x_rb = [feat_patch_rb_2, feat_patch_rb_1, feat_patch_rb_0]
        elif self.level == 1:
            pass
        elif self.level == 2:
            pass

        out_lt = self.ASFF_lt(x_lt)
        out_lb = self.ASFF_lb(x_lb)
        out_rt = self.ASFF_rt(x_rt)
        out_rb = self.ASFF_rb(x_rb)
        out_t = torch.cat((out_lt, out_rt), dim=3)
        out_b = torch.cat((out_lb, out_rb), dim=3)
        # out_t = self.feat_patchconv_t(out_t)
        # out_b = self.feat_patchconv_b(out_b)
        out = torch.cat((out_t, out_b), dim=2)
        return out

    def get_centroid(self, x):
        with torch.no_grad():
            x_2 = x.sum(2)
            x_3 = x.sum(3)
            d = 0
            for i in range(x.shape[3]):
                d = x_2[:, :,i] + d
                if d.sum() > 0.5 * x.sum():
                    break
            i = i // 2 * 2
            i = 4 if i < 4 else i
            i = x.shape[3]-4 if i > x.shape[3]-4 else i
            centroid_y = i
            d = 0
            for i in range(x.shape[2]):
                d = x_3[:, :, i] + d
                if d.sum() > 0.5 * x.sum():
                    break
            i = i // 2 * 2
            i = 4 if i < 4 else i
            i = x.shape[2]-4 if i > x.shape[2]-4 else i
            centroid_x = i

        return centroid_x, centroid_y

if __name__ == "__main__":
    asff_1 = Gridding_ASFF(level=0, multiplier=1)
    x_0 = torch.ones([2, 256, 80, 80])
    x_1 = torch.ones([2, 512, 40, 40])
    x_2 = torch.ones([2, 1024, 20, 20])
    x = [x_0, x_1, x_2]

    y = asff_1(x)
    print(y.size())