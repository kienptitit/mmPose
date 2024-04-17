import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from Config import Config
import os
import argparse

pose2label = {
    'pose_1': 0,
    'pose_2': 1,
    'pose_3': 2,
    'pose_4': 3,
    'pose_5': 4,
    'pose_6': 5,
    'pose_7': 6,
    'pose_8': 7,
    'pose_9': 8,
    'pose_10': 9,
    'free_form': 10,
    'walk': 11,
}


class mRiDataset(Dataset):
    def __init__(
            self,
            phase='train',
            cfg: Config = None
    ):
        data_split = {
            'train': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            'val': [15, 16],
            'test': [17, 18, 19, 20]
        }
        self.cfg = cfg
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        self.phase = phase
        self.split = data_split[phase]
        self.anno_dir = self.cfg.anno_dir
        self.radar_dir = self.cfg.radar_dir
        self.cfg = cfg
        self.load_data()

    def split_data(self, features, kpts):
        seq_length = self.cfg.train_seq_length if self.phase == 'train' else self.cfg.test_seq_length

        features = torch.split(features, seq_length)
        kpts = torch.split(kpts, seq_length)

        if features[-1].shape[0] != features[-2].shape[0]:
            features = features[:-1]
            kpts = kpts[:-1]

        return features, kpts

    def load_data(self):
        print(f"Loading {self.phase} data ...")
        all_data = []
        all_kps = []

        for subject_idx in self.split:
            # print(subject_idx)

            data = torch.from_numpy(
                np.load(os.path.join(self.radar_dir, "subject" + str(subject_idx) + "_featuremap.npy")))
            anno = pickle.load(
                open(os.path.join(self.anno_dir, "subject" + str(subject_idx) + "_all_labels.cpl"), "rb"))

            sos, eos = anno['radar_avail_frames']
            refined_kps = anno['refined_gt_kps']
            refined_kps = torch.from_numpy(refined_kps[sos:eos + 1])

            features, kpts = self.split_data(data, refined_kps)
            all_data.extend(features)
            all_kps.extend(kpts)

        self.data_list = all_data
        self.kps_list = all_kps
        print("Done!!!")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        pc_data = self.data_list[index]
        seq_length = self.cfg.train_seq_length if self.phase == 'train' else self.cfg.test_seq_length
        pc_data = pc_data.reshape(seq_length, -1, self.cfg.in_features_raw_data)  # 64,196,5
        kps = self.kps_list[index]
        kps = kps.permute(0, 2, 1)
        kps = kps.reshape(seq_length, -1)

        return pc_data, kps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument('--model_name', type=str, default='mmPosePlus', help="First operand (default: 0)")
    parser.add_argument('--dataset_name', type=str, default='mRI', help="First operand (default: 0)")
    parser.add_argument("--dataset_root",
                        default='/home/naver/Documents/HAR_workspace/Data/mm-Fi/MMFi_Dataset-003/MMFi_Dataset',
                        type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='/home/naver/Documents/Kien/mRI/mrPose/config.yaml', type=str,
                        help="Configuration YAML file")

    args = parser.parse_args()

    cfg = Config(args)
    anno_dir = "/home/naver/Documents/Kien/mRI/Dataset/dataset_release/aligned_data/pose_labels"
    radar_dir = "/home/naver/Documents/Kien/mRI/Dataset/dataset_release/features/radar"
    dataset = mRiDataset(phase='test', cfg=cfg)
    pc_data, kps = dataset[0]
    velocitys = []
    for i in range(len(dataset)):
        pc_data, kps = dataset[i]
        pc_data = pc_data.reshape(-1, 5)
        velocitys.extend(pc_data[:, -2].numpy().tolist())
    print(np.min(velocitys),np.max(velocitys))
