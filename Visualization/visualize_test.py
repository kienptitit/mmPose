import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from Model.mmPose import mmWaveModel
from Config import Config
from torch.utils.data import DataLoader
import yaml
from Base3DVisualization import Base3DVisualization
import argparse
from mmfi_lib.mmfi import make_dataset, make_dataloader
import numpy as np

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


class mmPoseVisualization(Base3DVisualization):
    def __init__(self, args, model, frame_rate, figure_dir, video_output_path):
        super().__init__(args, model, frame_rate, figure_dir, video_output_path)

    # @abstractmethod
    def predict(self, data_points):
        """
        :param data_points:[B,Seq_length,196,5]
        :return:
        """
        if self.args.dataset_name == 'mRI':
            data_loader = DataLoader(data_points, batch_size=2, shuffle=False)
            outs = []
            with torch.no_grad():
                for data in data_loader:
                    h0 = torch.zeros(self.model.cfg.numlayers_RNN, data.shape[0],
                                     self.model.cfg.out_feature_GlobalModule).to(self.model.cfg.device)
                    c0 = torch.zeros(self.model.cfg.numlayers_RNN, data.shape[0],
                                     self.model.cfg.out_feature_GlobalModule).to(self.model.cfg.device)

                    out = self.model(data, h0, c0, h0, c0)  # [B,n_frames,C]
                    out = out.reshape(out.shape[0] * out.shape[1], 17, 3)
                    outs.append(out)
                outs = torch.concat(outs, dim=0)
            return outs
        elif self.args.dataset_name == 'mm-Fi':
            with torch.no_grad():
                data_points = data_points.to(self.model.cfg.device)
                h0 = torch.zeros(self.model.cfg.numlayers_RNN, data_points.shape[0],
                                 self.model.cfg.out_feature_GlobalModule).to(self.model.cfg.device)
                c0 = torch.zeros(self.model.cfg.numlayers_RNN, data_points.shape[0],
                                 self.model.cfg.out_feature_GlobalModule).to(self.model.cfg.device)
                return self.model(data_points, h0, c0, h0, c0).squeeze().reshape(297, 17, 3)

    def visualize(self, anno_file, data_file):
        if self.args.dataset_name == 'mRI':
            gt_pts = self.get_gt(anno_file)
            data = np.load(data_file)

            n_frames, _, _, n_features = data.shape
            data = torch.from_numpy(data.reshape(n_frames, -1, n_features))
            inps = self.split_features(data)
            b_1, n_1 = inps[0].shape[:2]
            b_2, n_2 = inps[1].shape[:2]
            inps_to_visualize = torch.concat((inps[0].reshape(b_1 * n_1, 196, 5)[:, :, : 3],
                                              inps[1].reshape(b_2 * n_2, 196, 5)[:, :, :3]),
                                             dim=0).detach().cpu().numpy()

            if len(inps) == 2:
                out1, out2 = self.predict(inps[0]), self.predict(inps[1])
                out = torch.concat([out1, out2], dim=0)
            else:
                out = self.predict(inps)

            print("---Done Prediction---")
            out = out.detach().cpu().numpy()

            # out = out[:20]
            # gt_pts = gt_pts[:20]
            # inps_to_visualize = inps_to_visualize[:20]

            self.make_animation(out, gt_pts, inps_to_visualize)
        elif self.args.dataset_name == 'mm-Fi':
            pred = self.predict(data_file).detach().cpu().numpy()
            gt = anno_file.detach().cpu().numpy()

            self.make_animation(pred, gt, data_file.squeeze()[..., :3])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your script')

    # Add command-line arguments
    parser.add_argument('--anno_file',
                        default='/home/naver/Documents/Kien/mRI/Dataset/dataset_release/aligned_data/pose_labels/subject18_all_labels.cpl',
                        help='Description of arg1')
    parser.add_argument('--data_file',
                        default='/home/naver/Documents/Kien/mRI/Dataset/dataset_release/features/radar/subject18_featuremap.npy',
                        help='Description of arg2')
    parser.add_argument("--dataset_root",
                        default='/home/naver/Documents/HAR_workspace/Data/mm-Fi/MMFi_Dataset-003/MMFi_Dataset',
                        type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='/home/naver/Documents/Kien/mRI/mrPose/config.yaml', type=str,
                        help="Configuration YAML file")
    parser.add_argument('--dataset_name', default='mRI', help='Description of arg2')
    parser.add_argument('--idx', type=int, default=2, help='Description of arg2')

    args = parser.parse_args()

    cfg = Config(args)
    if args.dataset_name == 'mm-Fi':
        weight_path = '/home/naver/Documents/Kien/mRI/mrPose/Logs/Model_mmPose/mm-Fi/2024-03-19__15:37:19/model_30.pt'
    else:
        weight_path = '/home/naver/Documents/Kien/mRI/mrPose/Logs/Model__2024-03-11__15:50:09/model_88.pt'
    model = mmWaveModel(cfg=cfg).to(cfg.device)
    model.load_state_dict(torch.load(weight_path))

    if args.dataset_name == 'mRI':
        subject_id = os.path.basename(args.anno_file).split('.')[0].split('_')[0]
        fig_dir = os.path.join(cfg.save_fig_dir, subject_id)

        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        visualize = mmPoseVisualization(args, model, frame_rate=500, figure_dir=fig_dir,
                                        video_output_path=os.path.join(cfg.save_video, f'{subject_id}.avi'))
        visualize.visualize(args.anno_file, args.data_file)
    elif args.dataset_name == 'mm-Fi':
        fig_dir = os.path.join(cfg.save_fig_dir, str(args.idx))
        video_folder = os.path.join(cfg.save_video, 'mm_Fi')

        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        if not os.path.exists(video_folder):
            os.mkdir(video_folder)

        visualize = mmPoseVisualization(args, model, frame_rate=500, figure_dir=fig_dir,
                                        video_output_path=os.path.join(video_folder, f'{args.idx}.avi'))

        dataset_root = args.dataset_root
        with open(args.config_file, 'r') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)

        train_dataset, val_dataset = make_dataset(dataset_root, config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        eval_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator,
                                      **config['validation_loader'])
        data = None

        for idx, d in enumerate(eval_loader):
            if idx == args.idx:
                data = d
                break

        visualize.visualize(data['output'], data['input_mmwave'])
