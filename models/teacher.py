import torch
import torch.nn.functional as F
from models.resnet import wide_resnet50_2

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Teacher(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Teacher, self).__init__()

        self.encoder = wide_resnet50_2(pretrained=pretrained)

    def forward(self, x):
        x = self.encoder(x)
        return x

    def loss_distil(self, feature_s, feature_t):
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

    def loss_distil_aug(self, feature_s, feature_t):
        loss_type = torch.nn.CosineSimilarity()
        loss = 0.0
        for i in range(len(feature_s)):
            loss_i = torch.mean(1 + loss_type(
                feature_s[i].view(feature_s[i].shape[0], -1),
                feature_t[i].view(feature_t[i].shape[0], -1)))
            loss += loss_i

        return loss

    def loss_distil_aug_p(self, feature_s, feature_t, anomaly_mask):
        cos_loss = torch.nn.CosineSimilarity()
        loss_type = torch.nn.L1Loss()
        loss = 0.0
        for i in range(len(feature_s)):
            with torch.no_grad():
                anomaly_mask = F.interpolate(anomaly_mask, size=feature_s[i].shape[-1], mode='bilinear', align_corners=True)
                anomaly_mask = torch.where(
                    anomaly_mask < 0.5, torch.zeros_like(anomaly_mask), torch.ones_like(anomaly_mask)
                )

            cos = cos_loss(
                feature_s[i],
                feature_t[i])
            cos = torch.unsqueeze(1-cos, dim=1)
            loss += loss_type(cos, anomaly_mask)

        return loss

    def loss(self, output_t, output_e, anomaly_mask, epoch=None):
        """
        Overall loss of the teacher network
            output_t: the features of each layer in teacher [output_t_normal, output_t_aug]
            output_e: the features of each layer in expert [output_e_normal, output_e_aug]
        """

        output_t_normal, output_t_aug = output_t[0], output_t[1]
        output_e_normal, output_e_aug = output_e[0], output_e[1]

        loss_t_e_normal = self.loss_distil_p(output_t_normal, output_e_normal)
        loss_t_e_aug = self.loss_distil_aug_p(output_t_aug, output_e_normal, anomaly_mask) + self.loss_distil(output_t_aug, output_e_aug)

        loss_t_e = loss_t_e_normal + loss_t_e_aug

        return loss_t_e

