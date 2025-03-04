import torch
from models.student import Student
from models.teacher import Teacher
from dataset.dataset_noise import MVTecADNoiseDataset
from dataset.mvtec_ad import MVTecADDataset
from torch.utils.data import DataLoader
import numpy as np
import random
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter
from utils.evaluation import compute_pro
from tqdm import tqdm
import math

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(student, teacher, expert, train_dataloader, optimizer_s, optimizer_t, device, log, epoch, iteration):
    for batch_idx, data in enumerate(train_dataloader, 0):
        iteration += 1

        image, augmented_image, anomaly_mask = data
        image = image.to(device)
        augmented_image = augmented_image.to(device)
        with torch.no_grad():
            anomaly_mask = anomaly_mask.to(device)

        # normal
        with torch.no_grad():
            output_e_normal = expert(image)
            output_e_aug = expert(augmented_image)

        teacher.train()
        output_t_normal = teacher(image)
        output_t_aug = teacher(augmented_image)

        output_t = [output_t_normal, output_t_aug]
        output_e = [output_e_normal, output_e_aug]
        loss_t = teacher.loss(output_t, output_e, anomaly_mask)
        optimizer_t.zero_grad()
        loss_t.backward()
        optimizer_t.step()

        teacher.eval()
        with torch.no_grad():
            output_t_normal = teacher(image)
            output_t_aug = teacher(augmented_image)
            output_t_normal_detach = [output_t_normal[0].detach(), output_t_normal[1].detach(),
                                      output_t_normal[2].detach()]
            output_t_aug_detach = [output_t_aug[0].detach(), output_t_aug[1].detach(), output_t_aug[2].detach()]

        output_s_normal = student(output_t_normal_detach, skip=True, attn=True, noise=False, mode='train')
        output_s_aug = student(output_t_aug_detach, skip=True, attn=True, noise=False, mode='train')

        output_s = [output_s_normal, output_s_aug]
        output_t = [output_t_normal, output_t_aug]
        output_e = [output_e_normal, output_e_aug]
        loss_e_s = student.loss(output_s, output_t, output_e, anomaly_mask)
        optimizer_s.zero_grad()
        loss_e_s.backward()
        optimizer_s.step()

    return iteration


def train(device, classname, data_root, log, epochs, learning_rate, batch_size, img_size, iteration_i):
    ckp_path = './checkpoints/' + classname + '.pth'
    # prepare data
    train_mean = [0.485, 0.456, 0.406]
    train_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        # transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])
    gt_train_transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])
    gt_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    train_dir = data_root + classname + '/train'
    test_dir = data_root + classname + '/test'
    gt_dir = data_root + classname + '/ground_truth'

    anomaly_source_path = './data/dtd/images/'

    full_category = [
        'carpet', 'grid', 'leather', 'tile', 'wood', 'transistor',
    ]
    mask_category = [
        'bottle', 'cable', 'capsule',
        'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'zipper'
    ]
    if classname in full_category:
        mask = False
    elif classname in mask_category:
        mask = True

    train_data = MVTecADNoiseDataset(data_dir=train_dir, anomaly_source_path=anomaly_source_path, transform=transform,
                                     gt_transform=gt_train_transform, mask=mask, classname=classname)
    test_data = MVTecADDataset(data_dir=test_dir, gt_dir=gt_dir, transform=test_transform, gt_transform=gt_transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # load models
    expert = Teacher(pretrained=True)
    teacher = Teacher(pretrained=True)
    student = Student(img_size=img_size, pretrained=False)

    expert.to(device)
    teacher.to(device)
    student.to(device)

    expert.eval()
    teacher.train()
    student.train()

    # optimizer student-lr=0.005 teacher-lr=0.0001
    optimizer_s = torch.optim.Adam(student.parameters(), lr=0.005, betas=(0.5, 0.999))
    optimizer_t = torch.optim.Adam(teacher.parameters(), lr=0.0001, betas=(0.5, 0.999))

    save_max = []
    iteration = 0
    epochs = math.ceil(iteration_i / len(train_dataloader))
    for epoch in tqdm(range(epochs)):
        student.train()
        teacher.train()
        iteration = train_one_epoch(student, teacher, expert, train_dataloader, optimizer_s, optimizer_t, device, log, epoch, iteration)
        student.eval()
        teacher.eval()
        save_max = test(student, teacher, test_dataloader, device, log, save_max, epoch, ckp_path)

        if iteration >= iteration_i:
            break


def test(student, teacher, test_dataloader, device, log, save_max, epoch, ckp_path):
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            image, gt, ad_label, ad_type = data
            img = image.to(device)
            gt = gt.to(device)

            with torch.no_grad():
                output_t = teacher(img)
                output_s = student(output_t, skip=True, attn=True)
                anomaly_map = student.cal_anomaly_map(output_s, output_t)

            anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if ad_label.item() != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        aupro = round(np.mean(aupro_list), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

        if save_max == []:
            save_max = [auroc_px, auroc_sp, aupro, ap_px, ap_sp]
        else:
            # if save_max[2] > aupro:
            #     torch.save({'teacher': teacher.state_dict(),
            #                 'student': student.state_dict()}, ckp_path)
            # save_max[0] = max(save_max[0], auroc_px)
            # save_max[1] = max(save_max[1], auroc_sp)
            # save_max[2] = max(save_max[2], aupro)
            # save_max[3] = max(save_max[3], ap_px)
            # save_max[4] = max(save_max[4], ap_sp)

            if (auroc_px + aupro) > (save_max[0] + save_max[2]):
                save_max[0] = auroc_px
                save_max[1] = auroc_sp
                save_max[2] = aupro
                save_max[3] = ap_px
                save_max[4] = ap_sp
                # torch.save({'teacher': teacher.state_dict(),
                #             'student': student.state_dict()}, ckp_path)

        print('testing epoch %d' % (epoch + 1))
        print('testing epoch %d' % (epoch + 1), file=log)
        print(
            'Accuracy on test set: auroc_px %.4f (max: %.4f), auroc_sp %.4f (max: %.4f), aupro %.4f (max: %.4f), ap_px %.4f (max: %.4f), ap_sp %.4f (max: %.4f)'
            % (
            auroc_px, save_max[0], auroc_sp, save_max[1], aupro, save_max[2], ap_px, save_max[3], ap_sp, save_max[4]))
        print(
            'Accuracy on test set: auroc_px %.4f (max: %.4f), auroc_sp %.4f (max: %.4f), aupro %.4f (max: %.4f), ap_px %.4f (max: %.4f), ap_sp %.4f (max: %.4f)'
            % (
                auroc_px, save_max[0], auroc_sp, save_max[1], aupro, save_max[2], ap_px, save_max[3], ap_sp,
                save_max[4]), file=log)

    return save_max


if __name__ == "__main__":

    setup_seed(111)

    classnames = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                  'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    log = open("./log_train.txt", 'a')

    learning_rate = 0.005
    batch_size = 16
    img_size = 256
    data_root = './data/mvtec_anomaly_detection/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(len(classnames)):
        classname = classnames[i]
        epochs_i = 3000
        if classname in ['transistor', 'cable']:
            iteration_i = 10000
        else:
            iteration_i = 5000
        print('-----------------------training on ' + classname + '[%d / %d]-----------------------' % (
            i + 1, len(classnames)))
        print('-----------------------training on ' + classname + '[%d / %d]-----------------------' % (
            i + 1, len(classnames)), file=log)
        setup_seed(111)
        train(device, classname, data_root, log, epochs_i, learning_rate, batch_size, img_size, iteration_i)
