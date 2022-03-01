import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import pdist2, sparse_minconv, lbp_graph, thin_plate_dense, FeatNet, Augmenter, farthest_point_sampling,\
    get_current_consistency_weight, update_ema_variables



### hyper-parameters
device = 'cuda'
W, H, D = 192, 192, 208
k = 10
k1 = 256
N_p_fixed = 2048
N_p_moving = 8192
#LBP
lbps_iter = 5
lbps_cost_scale = 10
lbps_alpha = -50

num_epochs = 100
batch_size = 4
init_lr = 0.01
lr_steps = []
use_amp = True

teacher_alpha = 0.98
lambda_0 = 1.
rampup_len = 20


def match(kpts_fixed, kpts_moving, kpts_fixed_feat1, kpts_moving_feat1, N_p):
    if N_p == None:
        N_p = N_p_fixed
    dist = pdist2(kpts_fixed, kpts_moving)
    ind = (-dist).topk(k1, dim=-1)[1]
    candidates = - kpts_fixed.view(1, N_p, 1, 3) + kpts_moving[:, ind.view(-1), :].view(1, N_p, k1, 3)
    candidates_cost = (kpts_fixed_feat1.view(1, N_p, 1, -1) - kpts_moving_feat1[:, ind.view(-1), :].view(1, N_p, k1, -1)).pow(2).mean(3)
    edges, edges_reverse_idx = lbp_graph(kpts_fixed, k)
    messages = torch.zeros((edges.shape[0], k1)).to(device)
    candidates_edges0 = candidates[0, edges[:, 0], :, :]
    candidates_edges1 = candidates[0, edges[:, 1], :, :]
    for _ in range(lbps_iter):
        temp_message = torch.zeros((N_p, k1)).to(device).scatter_add_(0, edges[:, 1].view(-1, 1).repeat(1, k1),
                                                                            messages)
        multi_data_cost = torch.gather(temp_message + candidates_cost.squeeze(), 0,
                                       edges[:, 0].view(-1, 1).repeat(1, k1))
        reverse_messages = torch.gather(messages, 0, edges_reverse_idx.view(-1, 1).repeat(1, k1))
        multi_data_cost -= reverse_messages
        messages = sparse_minconv(multi_data_cost, candidates_edges0, candidates_edges1, lbps_cost_scale)
    reg_candidates_cost = (temp_message + candidates_cost.view(-1, k1)).unsqueeze(0)
    kpts_fixed_disp_pred = (candidates * F.softmax(lbps_alpha * reg_candidates_cost.view(1, N_p, -1),
                                                   2).unsqueeze(3)).sum(2)
    return kpts_fixed_disp_pred


def train_baseline(setting, method, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_path = os.path.join(out_dir, 'model_fold{}_ep{}.pth')

    data = torch.load('data.pth')

    if setting == 'copd_to_l2r':
        if method == 'source':
            kptss_fixed = data['copd']['kpts_fix']
            kptss_moving = data['copd']['kpts_mov']
            kptss_fixed_disp_gt = data['copd']['gt_disp']
        elif method == 'target':
            kptss_fixed = data['l2r']['train']['kpts_fix']
            kptss_moving = data['l2r']['train']['kpts_mov']
            kptss_fixed_disp_gt = data['l2r']['train']['gt_disp']
        folds = [1,]
        test_idx_per_fold = {1: torch.arange(0)}
    elif setting == '4dct_to_copd':
        if method == 'source':
            kptss_fixed = data['4dct']['kpts_fix']
            kptss_moving = data['4dct']['kpts_mov']
            kptss_fixed_disp_gt = data['4dct']['gt_disp']
            folds = [1]
            test_idx_per_fold = {1: torch.arange(0)}
        elif method == 'target':
            kptss_fixed = data['copd']['kpts_fix']
            kptss_moving = data['copd']['kpts_mov']
            kptss_fixed_disp_gt = data['copd']['gt_disp']
            folds = [1, 2, 3, 4, 5]
            test_idx_per_fold = {1: torch.tensor([0, 1]),
                                 2: torch.tensor([2, 3]),
                                 3: torch.tensor([4, 5]),
                                 4: torch.tensor([6, 7]),
                                 5: torch.tensor([8, 9])}
        else:
            raise ValueError()

    else:
        raise ValueError()

    augmenter = Augmenter()

    for fold_nu in folds:
        idx_test = test_idx_per_fold[fold_nu]
        a_cat_b, counts = torch.cat([idx_test, torch.arange(len(kptss_fixed))]).unique(return_counts=True)
        idx_train = a_cat_b[torch.where(counts.eq(1))].tolist()

        net = FeatNet()
        net.to(device)

        optimizer = optim.Adam(net.parameters(), init_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, 0.1)

        losses = []
        for epoch in range(num_epochs):
            net.train()
            for count, i in enumerate(
                    torch.randperm(len(idx_train))[:(len(idx_train) // batch_size * batch_size)].tolist()):
                case = idx_train[i]

                if (count % batch_size == 0):
                    optimizer.zero_grad()

                # load data
                kpts_fixed = kptss_fixed[case].unsqueeze(0).to(device)
                kpts_moving = kptss_moving[case].unsqueeze(0).to(device)
                kpts_fixed_disp_gt = kptss_fixed_disp_gt[case].unsqueeze(0).to(device)

                # sample kpts for source
                _, perm = farthest_point_sampling(kpts_fixed, N_p_fixed)
                kpts_fixed = kpts_fixed[:, perm, :]
                kpts_fixed_disp_gt = kpts_fixed_disp_gt[:, perm, :]
                _, perm = farthest_point_sampling(kpts_moving, N_p_moving)
                kpts_moving = kpts_moving[:, perm, :]

                with torch.cuda.amp.autocast(enabled=use_amp):
                    # augmentations
                    kpts_fixed, kpts_moving, rot, transl, scale = augmenter(kpts_fixed, kpts_moving)

                    # forward
                    kpts_fixed_feat, kpts_moving_feat = net(kpts_fixed, kpts_moving, k)

                    # source supervision
                    kpts_fixed_disp_pred = match(kpts_fixed, kpts_moving, kpts_fixed_feat, kpts_moving_feat, N_p=N_p_fixed)
                    kpts_fixed_disp_pred = torch.bmm(kpts_fixed_disp_pred / scale, rot.transpose(1, 2))
                    loss = nn.L1Loss()(kpts_fixed_disp_pred, kpts_fixed_disp_gt)

                # loss.backward()
                scaler.scale(loss).backward()

                if (count % batch_size == batch_size - 1):
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()

                # statistics
                print(loss.item())
                losses.append(loss.item())
                torch.cuda.empty_cache()

            print(epoch + 1)

            if ((epoch + 1) % 5 == 0):
                torch.save(net.cpu().state_dict(), model_path.format(fold_nu, epoch + 1))
                net.to(device)
                np.save(os.path.join(out_dir, 'lossHist_fold' + str(fold_nu) + '.npy'), np.array(losses))

            lr_scheduler.step()


def train_ours(setting, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_path = os.path.join(out_dir, 'model_fold{}_ep{}.pth')

    data = torch.load('data.pth')

    if setting == 'copd_to_l2r':
        kptss_fixed_src = data['copd']['kpts_fix']
        kptss_moving_src = data['copd']['kpts_mov']
        kptss_fixed_src_disp_gt = data['copd']['gt_disp']
        kptss_fixed_tgt = data['l2r']['train']['kpts_fix']
        kptss_moving_tgt = data['l2r']['train']['kpts_mov']
        folds = [1,]
        test_idx_per_fold = {1: torch.arange(0)}
    elif setting == '4dct_to_copd':
        kptss_fixed_src = data['4dct']['kpts_fix']
        kptss_moving_src = data['4dct']['kpts_mov']
        kptss_fixed_src_disp_gt = data['4dct']['gt_disp']
        kptss_fixed_tgt = data['copd']['kpts_fix']
        kptss_moving_tgt = data['copd']['kpts_mov']
        folds = [1, 2, 3, 4, 5]
        test_idx_per_fold = {1: torch.tensor([0, 1]),
                             2: torch.tensor([2, 3]),
                             3: torch.tensor([4, 5]),
                             4: torch.tensor([6, 7]),
                             5: torch.tensor([8, 9])}
    else:
        raise ValueError()

    augmenter = Augmenter()

    for fold_nu in folds:
        idx_test = test_idx_per_fold[fold_nu]
        a_cat_b, counts = torch.cat([idx_test, torch.arange(len(kptss_fixed_tgt))]).unique(return_counts=True)
        idx_train_tgt = a_cat_b[torch.where(counts.eq(1))].tolist()
        idx_train_src = torch.arange(len(kptss_fixed_src)).tolist()

        list_fac = int(len(idx_train_src) / len(idx_train_tgt)) + 1
        idx_train_tgt = idx_train_tgt * list_fac

        net = FeatNet()
        net.to(device)
        teacher = FeatNet().to(device)
        for param in teacher.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(net.parameters(), init_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, 0.1)

        losses = []
        for epoch in range(num_epochs):
            np.random.shuffle(idx_train_tgt)
            net.train()
            teacher.train()
            for count, i in enumerate(
                    torch.randperm(len(idx_train_src))[:(len(idx_train_src) // batch_size * batch_size)].tolist()):
                case_src = idx_train_src[i]
                case_tgt = idx_train_tgt[i]

                if (count % batch_size == 0):
                    optimizer.zero_grad()

                # load data
                kpts_fixed_src = kptss_fixed_src[case_src].unsqueeze(0).to(device)
                kpts_moving_src = kptss_moving_src[case_src].unsqueeze(0).to(device)
                kpts_fixed_disp_gt = kptss_fixed_src_disp_gt[case_src].unsqueeze(0).to(device)
                kpts_fixed_tgt = kptss_fixed_tgt[case_tgt].unsqueeze(0).to(device)
                kpts_moving_tgt = kptss_moving_tgt[case_tgt].unsqueeze(0).to(device)

                # sample kpts for source
                _, perm = farthest_point_sampling(kpts_fixed_src, N_p_fixed)
                kpts_fixed_src = kpts_fixed_src[:, perm, :]
                kpts_fixed_disp_gt = kpts_fixed_disp_gt[:, perm, :]
                _, perm = farthest_point_sampling(kpts_moving_src, N_p_moving)
                kpts_moving_src = kpts_moving_src[:, perm, :]

                with torch.cuda.amp.autocast(enabled=use_amp):
                    # augmentations for source
                    kpts_fixed_src, kpts_moving_src, rot_src, transl_src, scale_src = augmenter(kpts_fixed_src,
                                                                                                    kpts_moving_src)
                    # sample kpts for target
                    _, perm = farthest_point_sampling(kpts_fixed_tgt, N_p_fixed)
                    kpts_fixed_tgt = kpts_fixed_tgt[:, perm, :]
                    _, perm = farthest_point_sampling(kpts_moving_tgt, N_p_moving)
                    kpts_moving_tgt = kpts_moving_tgt[:, perm, :]

                    # augment target keypoints
                    kpts_fixed_tgt_1, kpts_moving_tgt_1, rot_tgt_1, transl_tgt_1, scale_tgt_1 = augmenter(kpts_fixed_tgt, kpts_moving_tgt)
                    kpts_fixed_tgt_2, kpts_moving_tgt_2, rot_tgt_2, transl_tgt_2, scale_tgt_2 = augmenter(kpts_fixed_tgt, kpts_moving_tgt)

                    # forward source and target
                    kpts_fixed_src_feat, kpts_moving_src_feat = net(kpts_fixed_src, kpts_moving_src, k)
                    kpts_fixed_tgt_feat_1, kpts_moving_tgt_feat_1 = net(kpts_fixed_tgt_1, kpts_moving_tgt_1, k)

                    # source supervision
                    kpts_fixed_disp_pred_src = match(kpts_fixed_src, kpts_moving_src, kpts_fixed_src_feat, kpts_moving_src_feat, N_p=N_p_fixed)
                    kpts_fixed_disp_pred_src = torch.bmm(kpts_fixed_disp_pred_src / scale_src, rot_src.transpose(1, 2))
                    loss = nn.L1Loss()(kpts_fixed_disp_pred_src, kpts_fixed_disp_gt)

                    # target supervision: mean-teacher
                    # get target predictions from student and teacher
                    kpts_fixed_disp_pred_tgt_1 = match(kpts_fixed_tgt_1, kpts_moving_tgt_1,
                                                       kpts_fixed_tgt_feat_1, kpts_moving_tgt_feat_1, N_p=N_p_fixed)
                    kpts_fixed_tgt_feat_2, kpts_moving_tgt_feat_2 = teacher(kpts_fixed_tgt_2, kpts_moving_tgt_2,
                                                                            k)
                    kpts_fixed_disp_pred_tgt_2 = match(kpts_fixed_tgt_2, kpts_moving_tgt_2,
                                                       kpts_fixed_tgt_feat_2, kpts_moving_tgt_feat_2, N_p=N_p_fixed)

                    # revert augmentations
                    kpts_fixed_disp_pred_tgt_1 = torch.bmm(kpts_fixed_disp_pred_tgt_1 / scale_tgt_1,
                                                           rot_tgt_1.transpose(1, 2))
                    kpts_fixed_disp_pred_tgt_2 = torch.bmm(kpts_fixed_disp_pred_tgt_2 / scale_tgt_2,
                                                           rot_tgt_2.transpose(1, 2))

                    loss_da = nn.L1Loss()(kpts_fixed_disp_pred_tgt_1, kpts_fixed_disp_pred_tgt_2)

                    loss += get_current_consistency_weight(lambda_0, epoch + 1, rampup_len) * loss_da

                # loss.backward()
                scaler.scale(loss).backward()

                if (count % batch_size == batch_size - 1):
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    update_ema_variables(net, teacher, teacher_alpha)

                # statistics
                print(loss.item())
                losses.append(loss.item())
                torch.cuda.empty_cache()

            print(epoch + 1)

            if ((epoch + 1) % 5 == 0):
                torch.save(teacher.cpu().state_dict(), model_path.format(fold_nu, epoch+1))
                teacher.to(device)
                np.save(os.path.join(out_dir, 'lossHist_fold' + str(fold_nu) + '.npy'), np.array(losses))

            lr_scheduler.step()

def test(setting, model_path, out_dir):
    data = torch.load('data.pth')

    if setting == 'copd_to_l2r':
        kptss_fixed = data['l2r']['test']['kpts_fix']
        kptss_moving = data['l2r']['test']['kpts_mov']
        folds = [1,]
        test_idx_per_fold = {1: torch.arange(10)}
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, 'disp_{:04d}_{:04d}.npz')
    elif setting == '4dct_to_copd':
        kptss_fixed = data['copd']['kpts_fix']
        kptss_moving = data['copd']['kpts_mov']
        folds = [1, 2, 3, 4, 5]
        test_idx_per_fold = {1: torch.tensor([0, 1]),
                             2: torch.tensor([2, 3]),
                             3: torch.tensor([4, 5]),
                             4: torch.tensor([6, 7]),
                             5: torch.tensor([8, 9])}
        tre_copd = torch.zeros(2, 10)
    else:
        raise ValueError()

    for fold_nu in folds:

        # load model
        net = FeatNet()
        net.to(device)
        if 'fold' in model_path:
            net.load_state_dict(torch.load(model_path.format(fold_nu)))
        else:
            net.load_state_dict(torch.load(model_path))
        net.eval()

        idx_test = test_idx_per_fold[fold_nu].tolist()
        for case in idx_test:

            # load data
            kpts_fixed = kptss_fixed[case].unsqueeze(0).to(device)
            kpts_moving = kptss_moving[case].unsqueeze(0).to(device)

            N_p_fixed_test = kpts_fixed.shape[1]

            # extract feats
            kpts_fixed_feat1, kpts_moving_feat1 = net(kpts_fixed, kpts_moving, k)
            # match via LBP
            kpts_fixed_disp_pred = match(kpts_fixed, kpts_moving, kpts_fixed_feat1, kpts_moving_feat1, N_p=N_p_fixed_test)

            if setting == 'copd_to_l2r':
                # interpolate predictions to entire volume
                dense_flow_ = thin_plate_dense(kpts_fixed, kpts_fixed_disp_pred, (H, W, D), 4, 0.1)
                dense_flow = dense_flow_.flip(4).permute(0, 4, 1, 2, 3) * torch.tensor(
                    [H - 1, W - 1, D - 1]).cuda().view(1, 3, 1, 1, 1) / 2
                grid_sp = 2
                disp_lr = F.interpolate(dense_flow, size=(H // grid_sp, W // grid_sp, D // grid_sp), mode='trilinear',
                                        align_corners=False)
                disp_lr = np.float16(disp_lr[0].detach().cpu().numpy())
                case_id = case + 21
                np.savez(out_path.format(case_id, case_id), disp_lr)

            elif setting == '4dct_to_copd':
                dense_flow_ = thin_plate_dense(kpts_fixed, kpts_fixed_disp_pred, (H, W, D), 4, 0.1)
                disp_hr = dense_flow_.permute(0, 4, 1, 2, 3)

                copd_lms = torch.load('copd_converted_lms.pth')

                disp = F.grid_sample(disp_hr.cpu(), copd_lms['lm_copd_exh'][case].view(1, -1, 1, 1, 3),
                                     align_corners=False).squeeze().t()
                tre_before = ((copd_lms['lm_copd_exh'][case] - copd_lms['lm_copd_insp'][case]) * (
                            torch.tensor([208 / 2, 192 / 2, 192 / 2]).view(1, 3) * torch.tensor(
                        [1.25, 1., 1.75]))).pow(2).sum(1).sqrt()

                tre_after = ((copd_lms['lm_copd_exh'][case] - copd_lms['lm_copd_insp'][case] + disp) * (
                            torch.tensor([208 / 2, 192 / 2, 192 / 2]).view(1, 3) * torch.tensor(
                        [1.25, 1., 1.75]))).pow(2).sum(1).sqrt()

                print('DIRlab COPD #', case, 'TRE before (mm)', '%0.3f' % (tre_before.mean().item()), 'TRE after (mm)', '%0.3f' % (tre_after.mean().item()))

                tre_copd[0, case] = tre_before.mean().item()
                tre_copd[1, case] = tre_after.mean().item()

    if setting == '4dct_to_copd':
        print('-----------COPD-TRE-LM------------')
        print('---------initial--------------')
        print('tre per subject: ', tre_copd[0])
        print('mean tre: {} mm'.format(tre_copd[0].mean().item()))
        print('---------method--------------')
        print('tre per subject: ', tre_copd[1])
        print('mean tre: {} mm'.format(tre_copd[1].mean()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--phase",
        default="test",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--setting",
        default="4dct_to_copd",
        metavar="FILE",
        help="adaptation scenario: {4dct_to_copd, copd_to_l2r}",
        type=str,
    )
    parser.add_argument(
        "--method",
        default="4dct_to_copd",
        metavar="FILE",
        help="method to train: {source, target, ours}",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default="",
        metavar="FILE",
        help="directory to write results to",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="",
        metavar="FILE",
        help="path pretrained model",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default="0",
        metavar="FILE",
        help="gpu to train on",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.phase == 'test':
        test(setting=args.setting, model_path=args.model_path, out_dir=args.out_dir)
    elif args.phase == 'train':
        if args.method == 'ours':
            train_ours(setting=args.setting, out_dir=args.out_dir)
        elif args.method in ['source', 'target']:
            train_baseline(setting=args.setting, method=args.method, out_dir=args.out_dir)
        else:
            raise ValueError()
    else:
        raise ValueError()
