import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle
import torch
from evaluation_metric import calculate_mpjpe
import os
from write_video import create_video
import cv2
from abc import ABC, abstractmethod
import shutil
from tqdm import tqdm


class Base3DVisualization(ABC):
    def __init__(self, args, model, frame_rate, figure_dir, video_output_path, remove_figure_dir=True):
        self.args = args
        self.model = model
        self.model.eval()
        self.seq_length = self.model.cfg.test_seq_length
        self.frame_rate = frame_rate
        self.x_min = -2
        self.x_max = 2
        self.y_min = -2
        self.y_max = 4.5
        self.z_min = 2.0
        self.z_max = 4.5
        # self.x_min = self.y_min = self.z_min = -3.5
        # self.x_max = self.y_max = self.z_max = 3.5
        self.figure_dir = figure_dir
        self.video_output_path = video_output_path
        self.remove_figure_dir = remove_figure_dir

    @abstractmethod
    def predict(self, data_points):
        """
        :param data_points:[B,Seq_length,196,5]
        :return:
        """
        # data_loader = DataLoader(data_points, batch_size=2, shuffle=False)
        # outs = []
        # with torch.no_grad():
        #     for data in data_loader:
        #         h0 = torch.zeros(self.model.cfg.numlayers_RNN, data.shape[0],
        #                          self.model.cfg.out_feature_GlobalModule).to(self.model.cfg.device)
        #         c0 = torch.zeros(self.model.cfg.numlayers_RNN, data.shape[0],
        #                          self.model.cfg.out_feature_GlobalModule).to(self.model.cfg.device)
        #
        #         out = self.model(data, h0, c0, h0, c0)  # [B,n_frames,C]
        #         out = out.reshape(out.shape[0] * out.shape[1], 17, 3)
        #         outs.append(out)
        #     outs = torch.concat(outs, dim=0)
        # return outs
        pass

    def get_gt(self, file_name):
        data = pickle.load(open(file_name, "rb"))
        sos, eos = data['radar_avail_frames']
        refined_kps = data['refined_gt_kps'][sos:eos + 1]
        return refined_kps.transpose(0, 2, 1)

    def split_features(self, features):
        features = torch.split(features, self.seq_length, dim=0)
        if features[-1].shape[0] != features[-2].shape[0]:
            return torch.stack(features[:-1]).to(self.model.cfg.device).float(), features[-1].unsqueeze(0).to(
                self.model.cfg.device).float()
        return torch.stack(features).to(self.model.cfg.device).float()

    def make_animation(self, pred_points, gt_points, inps_to_visualize):
        fig = plt.figure(figsize=(80, 40))

        # Subplot for ground truth
        ax_gt = fig.add_subplot(131, projection='3d')
        ax_gt.set_xlabel('X')
        ax_gt.set_ylabel('Y')
        ax_gt.set_zlabel('Z')
        ax_gt.set_title('Ground Truth')

        # Subplot for predictions
        ax_pred = fig.add_subplot(132, projection='3d')
        ax_pred.set_xlabel('X')
        ax_pred.set_ylabel('Y')
        ax_pred.set_zlabel('Z')
        ax_pred.set_title('Predictions')

        # Subplot for Input
        ax_input = fig.add_subplot(133, projection='3d')
        ax_input.set_xlabel('X')
        ax_input.set_ylabel('Y')
        ax_input.set_zlabel('Z')
        ax_input.set_title('Input')

        global ani
        ani = None

        if self.args.dataset_name == 'mm-Fi':
            ax_gt.view_init(elev=-90, azim=270, vertical_axis='z')
            ax_pred.view_init(elev=-90, azim=270, vertical_axis='z')

        def update(frame):
            if frame == len(pred_points):

                create_video(self.figure_dir, self.video_output_path, frame_rate=10)

                if self.remove_figure_dir:
                    shutil.rmtree(self.figure_dir)
                    print(f"Removed {self.figure_dir}")
                print("Done")
                exit()

            else:
                # ax_gt
                ax_gt.cla()
                ax_gt.set_xlabel('X')
                ax_gt.set_ylabel('Y')
                ax_gt.set_zlabel('Z')

                ax_gt.set_xlim([self.x_min, self.x_max])
                ax_gt.set_ylim([self.y_min, self.y_max])
                ax_gt.set_zlim([self.z_min, self.z_max])

                gt_point = gt_points[frame]
                x, y, z = zip(*gt_point)
                scatter_gt = ax_gt.scatter(x, y, z, c='blue', marker='o', label='GT')
                ax_gt.legend()

                # ax_pred
                ax_pred.cla()
                ax_pred.set_xlim([self.x_min, self.x_max])
                ax_pred.set_ylim([self.y_min, self.y_max])
                ax_pred.set_zlim([self.z_min, self.z_max])

                ax_pred.set_xlabel('X')
                ax_pred.set_ylabel('Y')
                ax_pred.set_zlabel('Z')

                pred_point = pred_points[frame]
                pred_point = torch.from_numpy(pred_point)
                gt_point = torch.from_numpy(gt_point)
                mpjpe = calculate_mpjpe(pred_point, gt_point)
                x2, y2, z2 = zip(*pred_point)

                ax_pred.set_title(f"Frame {frame}, mpjpe {mpjpe.item():.4f}")
                scatter_pred = ax_pred.scatter(x2, y2, z2, c='red', marker='o', label='Pred')
                ax_pred.legend()

                # ax_input
                x_min_inp = y_min_inp = z_min_inp = -4.0
                x_max_inp = y_max_inp = z_max_inp = 4.0

                ax_input.cla()
                ax_input.set_xlim([x_min_inp, x_max_inp])
                ax_input.set_ylim([y_min_inp, y_max_inp])
                ax_input.set_zlim([z_min_inp, z_max_inp])

                ax_input.set_xlabel('X')
                ax_input.set_ylabel('Y')
                ax_input.set_zlabel('Z')

                input_point = inps_to_visualize[frame]
                x3, y3, z3 = zip(*input_point)

                scatter_input = ax_input.scatter(x3, y3, z3, c='red', marker='o', label='mmWare Points')
                ax_input.legend()

                if self.args.dataset_name == 'mRI':
                    ax_gt.view_init(elev=20, azim=240)
                    ax_pred.view_init(elev=20, azim=240)

                plt.savefig(os.path.join(self.figure_dir, f"frame_{frame}.png"))

                return scatter_gt, scatter_pred, scatter_input

            # rotate_once += 1

        num_frames = len(pred_points)
        ani = animation.FuncAnimation(fig, update, frames=num_frames + 1, interval=1000 / self.frame_rate,
                                      repeat=True)
        plt.show()
        return

    @abstractmethod
    def visualize(self, anno_file, data_file):
        pass
