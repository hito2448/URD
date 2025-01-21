import torch
import torch.nn.functional as F
from .resnet import bn
from .de_resnet import de_wide_resnet50_2

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Student(torch.nn.Module):
    def __init__(self, img_size=256, anomaly_mode='add', pretrained=False):
        super(Student, self).__init__()
        self.img_size = img_size
        self.anomaly_mode = anomaly_mode

        self.bn = bn()

        self.decoder = de_wide_resnet50_2(pretrained=pretrained)

        self.relu = torch.nn.ReLU()

        dims = [256, 512, 1024]

        self.fuse_skip = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(dims[i + 1] * 2, dims[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(dims[i + 1]),
                torch.nn.ReLU(inplace=True)
            ) for i in range(2)
        ])
        self.conv3x3_simattn = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(dims[i + 1], dims[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(dims[i + 1]),
                torch.nn.ReLU(inplace=True)
            ) for i in range(2)
        ])

        self.conv1x1_downsample_dim = torch.nn.ModuleList([
            torch.nn.Sequential(
                # torch.nn.AvgPool2d(kernel_size=2, stride=2),
                torch.nn.Upsample(scale_factor=0.5, mode='bilinear'),
                torch.nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(dims[i + 1]),
                torch.nn.ReLU(inplace=True),
                # torch.nn.Upsample(scale_factor=0.5, mode='bilinear'),
                # torch.nn.AvgPool2d(kernel_size=2, stride=2),
            ) for i in range(2)
        ])

        self.fuse_encoder = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(dims[i + 1], dims[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(dims[i + 1]),
                torch.nn.ReLU(inplace=True)
            ) for i in range(2)
        ])

        self.apply(init_weight)

    def forward(self, x, skip=False, attn=False, noise=False, mode='test'):
        """Get the features    x: img"""
        fea = x

        if noise is True:
            rand = torch.randint(0, 10, (1,)).item()
            if rand == 0:
                std = torch.rand(1).numpy()[0] * 0.1
                x[0] = self.add_gaussian_noise(x[0], noise_std=std)

            rand = torch.randint(0, 10, (1,)).item()
            if rand == 0:
                std = torch.rand(1).numpy()[0] * 0.1
                x[1] = self.add_gaussian_noise(x[1], noise_std=std)

            rand = torch.randint(0, 10, (1,)).item()
            if rand == 0:
                std = torch.rand(1).numpy()[0] * 0.1
                x[2] = self.add_gaussian_noise(x[2], noise_std=std)

        x = self.bn(x)

        if skip is False:
            x = self.decoder(x)
        else:
            if attn is False:
                x = self.forward_decoder_gii(x, fea, mode)
            else:
                x = self.forward_decoder_wogii(x, fea, mode)

        return x

    def forward_decoder_gii(self, x, fea, mode='test'):
        feature_a = self.decoder.layer1(x)  # 512*8*8->256*16*16

        with torch.no_grad():
            sim_a = torch.unsqueeze(F.cosine_similarity(feature_a.detach(), fea[2]), dim=1)
            sim_a_a = sim_a
        f_encoder = self.conv1x1_downsample_dim[1](fea[1])
        f_encoder = self.fuse_encoder[1](fea[2] + f_encoder)
        f_b = self.fuse_skip[1](torch.cat([feature_a, self.conv3x3_simattn[1](f_encoder * sim_a_a + feature_a * (1 - sim_a_a))], 1))
        feature_b = self.decoder.layer2(f_b)

        with torch.no_grad():
            sim_b = torch.unsqueeze(F.cosine_similarity(feature_b.detach(), fea[1]), dim=1)
            # if mode == 'train':
            #     std = torch.rand(1).numpy()[0] * 0.2
            #     # std = 0.2 / (epoch//10 + 1)
            #     # std = torch.rand(1).numpy()[0] * 0.2 * (epoch / 1000 + 1)
            #     sim_b = self.add_gaussian_noise_mask(sim_b, std)
            sim_a_b = sim_b
        f_encoder = self.conv1x1_downsample_dim[0](fea[0])
        f_encoder = self.fuse_encoder[0](fea[1] + f_encoder)
        f_c = self.fuse_skip[0](torch.cat([feature_b, self.conv3x3_simattn[0](f_encoder * sim_a_b + feature_b * (1 - sim_a_b))], 1))
        feature_c = self.decoder.layer3(f_c)

        return [feature_c, feature_b, feature_a]

    def forward_decoder_wogii(self, x, fea, mode='test'):
        feature_a = self.decoder.layer1(x)  # 512*8*8->256*16*16

        f_encoder = self.conv1x1_downsample_dim[1](fea[1])
        f_encoder = self.fuse_encoder[1](fea[2] + f_encoder)
        f_b = self.fuse_skip[1](torch.cat([feature_a, f_encoder], 1))
        feature_b = self.decoder.layer2(f_b)

        # f_e = self.conv3x3_downsample_dim[0](fea[0])
        f_encoder = self.conv1x1_downsample_dim[0](fea[0])
        f_encoder = self.fuse_encoder[0](fea[1] + f_encoder)
        f_c = self.fuse_skip[0](torch.cat([feature_b, f_encoder], 1))
        feature_c = self.decoder.layer3(f_c)

        return [feature_c, feature_b, feature_a]

    def loss_distil(self, feature_s, feature_t):
        """
        Distillation between the student and teacher output
            feature_s: features of student aligned to the size of features of teacher
            feature_t: features of teacher
        """

        loss_type = torch.nn.CosineSimilarity()
        loss = 0.0
        for i in range(len(feature_s)):
            loss_i = torch.mean(1 - loss_type(
                feature_s[i].view(feature_s[i].shape[0], -1),
                feature_t[i].view(feature_t[i].shape[0], -1)))
            loss += loss_i

        return loss

    def loss_distil_p(self, feature_s, feature_t):
        """
        Distillation between the student and teacher output (pixel-wise)
            feature_s: features of student aligned to the size of features of teacher
            feature_t: features of teacher
        """
        loss_type = torch.nn.CosineSimilarity()
        loss = 0.0
        for i in range(len(feature_s)):
            cos = 1 - loss_type(feature_s[i], feature_t[i])
            loss_i = torch.mean(cos)
            loss += loss_i

        return loss

    def loss_distil_aug_normalpixels(self, feature_s, feature_t, anomaly_mask):
        """
        Distillation between the student and teacher output only in normal regions when the synthetic image is input
            feature_s: features of student aligned to the size of features of teacher
            feature_t: features of teacher
            anomaly_mask: synthetic ground truth (normal 0, anomalous 1)
        """

        loss_type = torch.nn.CosineSimilarity()
        # loss_type = nn.L1Loss()
        loss = 0.0
        for i in range(len(feature_s)):
            with torch.no_grad():
                anomaly_mask = F.interpolate(anomaly_mask, size=feature_s[i].shape[-1], mode='bilinear', align_corners=True)
                anomaly_mask = torch.where(
                    anomaly_mask < 0.5, torch.zeros_like(anomaly_mask), torch.ones_like(anomaly_mask)
                )

            cos = loss_type(feature_s[i], feature_t[i])
            cos = torch.unsqueeze(1 - cos, dim=1)
            cos = cos * (1-anomaly_mask)
            loss_i = torch.mean(cos)
            loss += loss_i

        return loss

    def loss(self, output_s, output_t, output_e, anomaly_mask, epoch=None):
        """
        Overall loss of the student network
            output_s: the features of each layer in student [output_s_normal, output_s_aug]
            output_t: the features of each layer in teacher [output_t_normal, output_t_aug]
            output_e: the features of each layer in expert [output_e_normal, output_e_aug]
        """

        output_s_normal, output_s_aug = output_s[0], output_s[1]
        output_t_normal, output_t_aug = output_t[0], output_t[1]
        output_e_normal, output_e_aug = output_e[0], output_e[1]

        loss_t_normal = self.loss_distil(output_s_normal, output_t_normal)
        loss_t_aug = self.loss_distil(output_s_aug, output_t_normal)

        loss_e_normal = self.loss_distil(output_s_normal, output_e_normal)
        loss_e_aug = self.loss_distil(output_s_aug,output_e_normal)

        loss_s_e = loss_e_normal + loss_e_aug
        loss_s_t = loss_t_normal + loss_t_aug

        loss = loss_s_e + loss_s_t

        return loss

    def cal_anomaly_map(self, feature_s, feature_t):
        """
        Calculate the anomaly map by the features of the teacher and student
        input:
            feature_s: features of student aligned to the size of features of teacher
            feature_t: features of teacher
        """

        anomaly_map_fuse = None

        l = len(feature_s)

        for i in range(l):
            anomaly_map = torch.unsqueeze(1 - F.cosine_similarity(feature_s[i], feature_t[i]), dim=1)
            anomaly_map = F.interpolate(anomaly_map, size=self.img_size, mode='bilinear', align_corners=True)

            if anomaly_map_fuse is None:
                anomaly_map_fuse = anomaly_map
                continue

            if self.anomaly_mode == 'mul':
                anomaly_map_fuse = torch.mul(anomaly_map_fuse, anomaly_map)
            else:
                anomaly_map_fuse = anomaly_map_fuse + anomaly_map


        return anomaly_map_fuse



