import torch
from dataset import mRiDataset
from Model.mmPose import mmWaveModel
from Model.mmPosePlus import mmPosePlus
from Config import Config, ConfigPlus
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from evaluation_metric import calculate_mpjpe, calculate_pampjpe
import numpy as np
import random
from datetime import datetime
import argparse
import torch
from mmfi_lib.mmfi import make_dataset, make_dataloader
import yaml
import pickle
from tqdm import tqdm

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


def mmPosePlus_forward(cfg, model, data):
    h0_coarse = torch.zeros(cfg.n_layers_RNN, data.shape[0], cfg.out_feature_GlobalPointNet,
                            dtype=torch.float32, device=cfg.device)
    c0_coarse = torch.zeros(cfg.n_layers_RNN, data.shape[0], cfg.out_feature_GlobalPointNet,
                            dtype=torch.float32, device=cfg.device)

    h0_agg = torch.zeros(cfg.n_layers_RNN, data.shape[0], cfg.out_feature_GCN, dtype=torch.float32,
                         device=cfg.device)
    c0_agg = torch.zeros(cfg.n_layers_RNN, data.shape[0], cfg.out_feature_GCN, dtype=torch.float32,
                         device=cfg.device)

    h0_joints = torch.zeros(cfg.n_layers_RNN, data.shape[0] * cfg.n_joints, cfg.out_feature_GCN,
                            dtype=torch.float32, device=cfg.device)
    c0_joints = torch.zeros(cfg.n_layers_RNN, data.shape[0] * cfg.n_joints, cfg.out_feature_GCN,
                            dtype=torch.float32, device=cfg.device)
    kpts_predict = model(data, h0_coarse, c0_coarse, h0_agg, c0_agg, h0_joints, c0_joints)
    return kpts_predict


def train_eval_epoch(args, dataloader, epoch, model, optimizer, criterion, file_w, phase):
    cfg = model.cfg
    if phase == 'train':
        model.train()
    else:
        model.eval()
    if phase == 'train':
        losses = 0.0
        mpjpes = 0.0
        pampjpes = 0.0
        for idx, loader in enumerate(dataloader):
            if args.dataset_name == 'mRI':
                data, kpts = loader
            elif args.dataset_name == 'mm-Fi':
                data, kpts = loader['input_mmwave'], loader['output']
                kpts = kpts.reshape(-1, 51).unsqueeze(0)
            data, kpts = data.to(cfg.device), kpts.to(cfg.device)
            data, kpts = data.float(), kpts.float()

            if args.model_name == 'mmPose':
                h0 = torch.zeros((cfg.numlayers_RNN, data.shape[0], cfg.out_feature_AnchorVoxelNet),
                                 dtype=torch.float32,
                                 device=cfg.device)
                c0 = torch.zeros((cfg.numlayers_RNN, data.shape[0], cfg.out_feature_AnchorVoxelNet),
                                 dtype=torch.float32,
                                 device=cfg.device)

                kpts_predict = model(data, h0, c0, h0, c0)
            elif args.model_name == 'mmPosePlus':
                kpts_predict = mmPosePlus_forward(cfg, model, data)

            loss = criterion(kpts_predict, kpts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kpts_predict = kpts_predict.reshape(cfg.batch_size, -1, 3)
            kpts = kpts.reshape(cfg.batch_size, -1, 3)

            mpjpe = calculate_mpjpe(kpts_predict, kpts).mean()
            pampjpe = calculate_pampjpe(kpts_predict, kpts).mean()

            losses += loss.detach().cpu().item()
            mpjpes += mpjpe.detach().cpu().item()
            pampjpes += pampjpe.detach().cpu().item()

            if idx % 10 == 0:
                s = (f"{phase} Epoch [{epoch}/{cfg.epochs}],\t "
                     f"Step [{idx}/{len(dataloader)}],\tLoss: {loss:.4f},"
                     f"\tmpjpe: {mpjpe:.4f},\tpampjpe: {pampjpe:.4f}")
                print(s)
                if cfg.write_log:
                    file_w.write(s + "\n")

        return losses / len(dataloader), mpjpes / len(dataloader), pampjpes / len(dataloader)
    else:
        outs = []
        gts = []
        with torch.no_grad():
            for idx, loader in tqdm(enumerate(dataloader)):
                if args.dataset_name == 'mRI':
                    data, kpts = loader
                elif args.dataset_name == 'mm-Fi':
                    data, kpts = loader['input_mmwave'], loader['output']
                    kpts = kpts.reshape(-1, 51).unsqueeze(0)

                data, kpts = data.to(cfg.device), kpts.to(cfg.device)
                data, kpts = data.float(), kpts.float()

                if args.model_name == 'mmPose':
                    h0 = torch.zeros(cfg.numlayers_RNN, data.shape[0],
                                     cfg.out_feature_GlobalModule).to(cfg.device)
                    c0 = torch.zeros(model.cfg.numlayers_RNN, data.shape[0],
                                     cfg.out_feature_GlobalModule).to(cfg.device)

                    out = model(data, h0, c0, h0, c0)  # [B,n_frames,C]
                elif args.model_name == 'mmPosePlus':
                    out = mmPosePlus_forward(cfg, model, data)

                gt = kpts.reshape(out.shape[0] * out.shape[1], 17, 3)
                out = out.reshape(out.shape[0] * out.shape[1], 17, 3)  # [B,n_frames,17,3]
                outs.append(out)
                gts.append(gt)

            outs = torch.concat(outs)  # [B * n_frames,17,3]
            gts = torch.concat(gts)  # [B * n_frames,17,3]
            mpjpe = calculate_mpjpe(outs, gts).detach().cpu().mean().item()
            s = (f"{phase} Epoch [{epoch}/{cfg.epochs}],\t"
                 f"mpjpe: {mpjpe:.4f},\t")
            print(s)
            if cfg.write_log:
                file_w.write(s + "\n")
            # visualize = Visualize_3D_Points(model, frame_rate=100)
            # anno_file = '/home/naver/Documents/Kien/mRI/Dataset/dataset_release/aligned_data/pose_labels/subject18_all_labels.cpl'
            # data_file = '/home/naver/Documents/Kien/mRI/Dataset/dataset_release/features/radar/subject18_featuremap.npy'
            # visualize(anno_file, data_file)
        return 0.0, mpjpe, 0.0


class mpjpe_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outs, gts):
        return calculate_mpjpe(outs, gts).mean()


def set_all_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_mmFi(args):
    dataset_root = args.dataset_root
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])

    train_loader = make_dataloader(train_dataset, is_training=False, generator=rng_generator,
                                   **config['train_loader'])
    eval_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator,
                                  **config['validation_loader'])
    return train_loader, eval_loader


def main(args, cfg: ConfigPlus):
    if args.dataset_name == 'mRI':
        train_data = mRiDataset(phase='train', cfg=cfg)
        eval_data = mRiDataset(phase='val', cfg=cfg)
        test_data = mRiDataset(phase='test', cfg=cfg)

        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
        eval_loader = DataLoader(eval_data, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    elif args.dataset_name == 'mm-Fi':
        train_loader, eval_loader = load_mmFi(args)

    criterion = nn.L1Loss()

    if args.model_name == 'mmPose':
        model = mmWaveModel(cfg).to(cfg.device)
        model.load_state_dict(
            torch.load(
                '/home/naver/Documents/Kien/mRI/mrPose/Logs/Model_mmPose/mm-Fi/2024-03-19__14:33:06/model_40.pt'))
    elif args.model_name == 'mmPosePlus':
        model = mmPosePlus(cfg).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_max)

    train_losses = []
    val_losses = []
    test_losses = []

    train_mpjes = []
    val_mpjes = []
    test_mpjes = []

    train_pampjes = []
    val_pampjes = []
    test_pampjes = []

    t = "__".join(str(datetime.now()).split(".")[0].split(" "))
    # training_log = os.path.join(cfg.log_path, f'training_log_{args.model_name}_{args.dataset_name}_{t}.txt')
    training_log_path = os.path.join(cfg.log_path, 'Model_' + args.model_name, args.dataset_name)
    if not os.path.exists(training_log_path):
        os.makedirs(training_log_path)
    training_log = os.path.join(training_log_path, f'{t}.txt')

    file_writer = open(training_log, 'a')
    print(f"********Training {args.model_name} in {args.dataset_name}********")
    file_writer.write(f"********Training {args.model_name} in {args.dataset_name}******** \n")

    save_model_path = os.path.join(cfg.save_path, 'Model_' + args.model_name, args.dataset_name, t)

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    for epoch in range(cfg.epochs):
        train_loss, mpjes_train, pampjpes_train = train_eval_epoch(args, train_loader, epoch, model, optimizer,
                                                                   criterion,
                                                                   file_writer,
                                                                   phase='train')
        line_sep = "------------------------------------------------------------------------"
        print(line_sep)
        if cfg.write_log:
            file_writer.write(line_sep + "\n")

        val_loss, mpjes_val, pampjpes_val = train_eval_epoch(args, eval_loader, epoch, model, optimizer, criterion,
                                                             file_writer,
                                                             phase='val')
        print(line_sep)
        if cfg.write_log:
            file_writer.write(line_sep + "\n")
        if args.dataset_name == 'mm-Fi':
            test_loss, mpjes_test, pampjpes_test = 0.0, 0.0, 0.0
        elif args.dataset_name == 'mRI':
            test_loss, mpjes_test, pampjpes_test = train_eval_epoch(args, test_loader, epoch, model, optimizer,
                                                                    criterion,
                                                                    file_writer,
                                                                    phase='test')
        print(line_sep)
        if cfg.write_log:
            file_writer.write(line_sep + "\n")

        loss_logs = f"Epoch [{epoch}/{cfg.epochs}],\tTrain_Loss: {train_loss:.4f},\tVal_Loss: {val_loss:.4f},\tTest_Loss: {test_loss:.4f}"
        mpjpe_logs = f"Epoch [{epoch}/{cfg.epochs}],\tTrain_mpjpe: {mpjes_train:.4f},\tVal_mpjpe: {mpjes_val:.4f},\tTest_mpjpe: {mpjes_test:.4f}"
        pampjpes_logs = (f"Epoch [{epoch}/{cfg.epochs}],\tTrain_pampjpes: {pampjpes_train:.4f},"
                         f"\tVal_pampjpes: {pampjpes_val:.4f},\tTest_pampjpes: {pampjpes_test:.4f}")

        print(loss_logs)
        print(mpjpe_logs)
        print(pampjpes_logs)
        if cfg.write_log:
            file_writer.write(loss_logs + "\n")
            file_writer.write(mpjpe_logs + "\n")
            file_writer.write(pampjpes_logs + "\n")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        train_mpjes.append(mpjes_train)
        val_mpjes.append(mpjes_val)
        test_mpjes.append(mpjes_test)

        train_pampjes.append(pampjpes_train)
        val_pampjes.append(pampjpes_val)
        test_pampjes.append(pampjpes_test)

        if epoch % cfg.save_model_frequency == 0:

            torch.save(model.state_dict(), os.path.join(save_model_path, f'model_{epoch}.pt'))
            print("-----Model Saved!!!-----")
            if cfg.write_log:
                file_writer.write("-----Model Saved!!!-----" + "\n")

        lr_scheduler.step()
        # torch.cuda.empty_cache()
        print(
            f"***************Done Epoch [{epoch}/{cfg.epochs}], lr = {optimizer.param_groups[0]['lr']}***************")
        if cfg.write_log:
            file_writer.write(
                f"***************Done Epoch [{epoch}/{cfg.epochs}], lr = {optimizer.param_groups[0]['lr']}***************" + "\n")

    metric_dir = os.path.join(save_model_path, 'Metric')
    if not os.path.exists(metric_dir):
        os.mkdir(metric_dir)

    pickle.dump(train_losses, open(os.path.join(metric_dir, 'train_losses.pkl'), "wb"))
    pickle.dump(val_losses, open(os.path.join(metric_dir, 'val_losses.pkl'), "wb"))
    pickle.dump(test_losses, open(os.path.join(metric_dir, 'test_losses.pkl'), "wb"))

    pickle.dump(train_mpjes, open(os.path.join(metric_dir, 'train_mpjes.pkl'), "wb"))
    pickle.dump(val_mpjes, open(os.path.join(metric_dir, 'val_mpjes.pkl'), "wb"))
    pickle.dump(test_mpjes, open(os.path.join(metric_dir, 'test_mpjes.pkl'), "wb"))

    pickle.dump(train_pampjes, open(os.path.join(metric_dir, 'train_pampjes.pkl'), "wb"))
    pickle.dump(val_pampjes, open(os.path.join(metric_dir, 'val_pampjes.pkl'), "wb"))
    pickle.dump(test_pampjes, open(os.path.join(metric_dir, 'test_pampjes.pkl'), "wb"))

    print(f"!!!!!!!!DONE, Best mpjpes: {np.min(test_mpjes):.4f}, epoch: {np.argmin(test_mpjes)}!!!!!!!!")
    if cfg.write_log:
        file_writer.write(
            f"!!!!!!!!DONE, Best mpjpes: {np.min(test_mpjes):.4f}, epoch: {np.argmin(test_mpjes)}!!!!!!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument('--model_name', type=str, default='mmPosePlus', help="First operand (default: 0)")
    parser.add_argument('--dataset_name', type=str, default='mm-Fi', help="First operand (default: 0)")
    parser.add_argument("--dataset_root",
                        default='/home/naver/Documents/HAR_workspace/Data/mm-Fi/MMFi_Dataset-003/MMFi_Dataset',
                        type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='/home/naver/Documents/Kien/mRI/mrPose/config.yaml', type=str,
                        help="Configuration YAML file")

    args = parser.parse_args()

    set_all_seed(42)
    if args.model_name == 'mmPose':
        cfg = Config(args)
    elif args.model_name == 'mmPosePlus':
        cfg = ConfigPlus(args)

    main(args, cfg)
