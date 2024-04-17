import torch
import torch.nn as nn
from Config import ConfigPlus
from torch_geometric.nn import GCNConv
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import warnings
import numpy as np
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class PointNet(nn.Module):
    def __init__(self, cfg, in_features, hidden_features_1, hidden_features_2, output_features, attention=True):
        super().__init__()

        self.cfg = cfg

        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=hidden_features_1, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(hidden_features_1)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=hidden_features_1, out_channels=hidden_features_2, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(hidden_features_2)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=hidden_features_2, out_channels=output_features, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(output_features)
        self.caf3 = nn.ReLU()

        self.attention = attention
        if attention:
            self.attn = nn.Linear(output_features, 1)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, in_mat):
        x = in_mat.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))
        x = x.transpose(1, 2)
        if self.attention:
            attn_weights = self.softmax(self.attn(x))  # Weight For Each Point In A Frames

            attn_vec = torch.sum(x * attn_weights, dim=1)
            return attn_vec
        else:
            x = torch.cat((in_mat[:, :, :self.cfg.coordinate_features], x), -1)

            return x


class CoarseSkeleton(nn.Module):
    def __init__(self, cfg: ConfigPlus):
        super().__init__()
        self.cfg = cfg
        self.global_pointNet = PointNet(cfg, in_features=cfg.out_features_BasePointNet + cfg.coordinate_features,
                                        hidden_features_1=32,
                                        hidden_features_2=48,
                                        output_features=cfg.out_feature_GlobalPointNet,
                                        attention=True)

        self.rnn = nn.LSTM(input_size=self.cfg.out_feature_GlobalPointNet,
                           hidden_size=self.cfg.out_feature_GlobalPointNet,
                           dropout=self.cfg.dropout_RNN,
                           num_layers=self.cfg.n_layers_RNN,
                           batch_first=True)

        self.ln_pose = nn.Linear(self.cfg.out_feature_GlobalPointNet,
                                 72)

        self.ln_shape = nn.Linear(self.cfg.out_feature_GlobalPointNet,
                                  10)

        self.smpl = SMPL_Layer(center_idx=0, gender='female',
                               model_root=self.cfg.smpl_path)

    def forward(self, x, h0, c0, batch_size, length_size):
        """
        :param x: [Batch_size * n_frames,N_points,24 + 4]
        :return:  [Batch_Size,N_frames,24,3]
        """
        x = self.global_pointNet(x)  #

        x = x.reshape(batch_size, length_size, self.cfg.out_feature_GlobalPointNet)  # [Batch_size,n_frames,64]
        g, (_, _) = self.rnn(x, (h0, c0))
        g = g.reshape(batch_size * length_size, -1)

        g_pose, g_shape = self.ln_pose(g), self.ln_shape(g)

        _, skeleton = self.smpl(g_pose, g_shape)

        edge = torch.tensor(self.smpl.kintree_table.astype(np.float32))[:, 1:].long().to(self.cfg.device)

        skeleton = skeleton.reshape(batch_size, length_size, self.cfg.n_joints, -1)
        g = g.reshape(batch_size, length_size, -1)

        return skeleton, edge, g


class GCN(nn.Module):
    def __init__(self, cfg: ConfigPlus):
        super().__init__()
        self.cfg = cfg
        self.gcn = nn.Sequential()
        for i in range(cfg.gcn_layers):
            self.gcn.append(GCNConv(
                in_channels=cfg.out_feature_GlobalPointNet + cfg.out_feature_JointPointNet if i == 0 else cfg.out_feature_GCN,
                out_channels=cfg.out_feature_GCN,
                node_dim=1))

    def forward(self, x, edge_index):
        """
        :param x: [batch_size,N_points,N_features]
        :param edge_index: [N_edges,2]
        :return:  [batch_size,N_points,N_features]
        """
        for i in range(self.cfg.gcn_layers):
            x = self.gcn[i](x, edge_index)
            x = F.relu(x)
        return x


class Pose_Aware_Joint(nn.Module):
    def __init__(self, cfg: ConfigPlus):
        super().__init__()
        self.cfg = cfg
        self.jointPointNet = PointNet(cfg=cfg,
                                      in_features=cfg.out_features_BasePointNet + cfg.coordinate_features + 3,
                                      hidden_features_1=32,
                                      hidden_features_2=48,
                                      output_features=cfg.out_feature_JointPointNet,
                                      attention=True)
        self.gcn = GCN(cfg)

    def index_point(self, points, index):
        c = points.shape[-1]
        index = index.unsqueeze(-1).repeat(1, 1, 1, c)
        return torch.gather(points, dim=2, index=index)

    def Joints_Grouping(self, joints, points, n_nearest=16):
        n_joints = joints.shape[1]

        xyz, points = points[..., :3], points[..., 3:]  # B,N_points,3

        distance = torch.norm(joints.unsqueeze(2) - xyz.unsqueeze(1), dim=-1)  # B,N_joints,N_points
        topk_dist = torch.argsort(distance, dim=-1)[..., :n_nearest]  # B,N_joints,n_nearest

        group_joints = joints.unsqueeze(2).repeat(1, 1, n_nearest, 1)

        xyz_expand = xyz.unsqueeze(1).repeat(1, n_joints, 1, 1)
        xyz_selected = self.index_point(xyz_expand, topk_dist)

        xyz_norm = xyz_selected - group_joints

        points = points.unsqueeze(1).repeat(1, n_joints, 1, 1)
        points_grouped = self.index_point(points, topk_dist)

        return torch.concat([group_joints, xyz_norm, points_grouped], dim=-1)

    def forward(self, points, g, skeleton, edge_index, n_nearest):
        """
        :param points: [Batch_Size,N_frames,N_points,N_features_points]
        :param g : [Batch_Size,N_frames,N_features_g]
        :param skeleton: [Batch_Size,N_frames,24,3]
        :return: skeleton features: [Batch_Size,N_frames,24,N_features]
        """
        b, n_f, n_points, _ = points.shape
        _, _, n_joints, _ = skeleton.shape

        skeleton = skeleton.reshape(b * n_f, n_joints, -1)
        points = points.reshape(b * n_f, n_points, -1)

        joints_grouped = self.Joints_Grouping(skeleton,
                                              points,
                                              n_nearest=n_nearest)  # Batch_Size * N_frames,N_joints,N_neraset,N_features_points

        joints_grouped = joints_grouped.view(b * n_f * n_joints, n_nearest, -1)

        joints_features = self.jointPointNet(joints_grouped)

        joints_features = joints_features.view(b * n_f, n_joints, -1)

        g = g.view(b * n_f, -1).unsqueeze(1).repeat(1, n_joints, 1)

        joints_features = torch.concat([joints_features, g], dim=-1)
        joints_features = joints_features.reshape(b * n_f, n_joints, -1)

        joints_features = self.gcn(joints_features, edge_index)

        return joints_features.reshape(b, n_f, n_joints, -1)


class PoseEstimator(nn.Module):
    def __init__(self, cfg: ConfigPlus):
        super().__init__()
        self.cfg = cfg
        self.jointAggregation = PointNet(cfg, in_features=cfg.out_feature_GCN,
                                         hidden_features_1=96,
                                         hidden_features_2=128,
                                         output_features=cfg.out_feature_GCN,
                                         attention=True)

        self.jagg_rnn = nn.LSTM(input_size=self.cfg.out_feature_GCN,
                                hidden_size=self.cfg.out_feature_GCN,
                                dropout=cfg.dropout_RNN,
                                num_layers=cfg.n_layers_RNN,
                                batch_first=True)

        self.jrnn = nn.LSTM(input_size=self.cfg.out_feature_GCN,
                            hidden_size=self.cfg.out_feature_GCN,
                            dropout=cfg.dropout_RNN,
                            num_layers=cfg.n_layers_RNN,
                            batch_first=True)

        self.ln = nn.Linear(
            self.cfg.out_feature_GCN + self.cfg.out_feature_GlobalPointNet +
            self.cfg.out_feature_GCN * self.cfg.n_joints,
            51)

    def forward(self, skeleton_f, g, h0_agg, c0_agg, h0_joint, c0_joint):
        """
        :param skeleton_f: [Batch_Size,N_frames,24,N_features]
        :return: [Batch_Size,N_frames,-1]
        """
        b, n_frames, n_joints, _ = skeleton_f.shape
        skeleton_agg = skeleton_f.reshape(b * n_frames, -1, self.cfg.out_feature_GCN)

        skeleton_f = skeleton_f.view(b, n_frames, -1, self.cfg.out_feature_GCN).permute(0, 2, 1, 3).reshape(-1,
                                                                                                            n_frames,
                                                                                                            self.cfg.out_feature_GCN)
        skeleton_agg = self.jointAggregation(skeleton_agg)
        skeleton_agg = skeleton_agg.reshape(b, n_frames, -1)
        skeleton_agg, (_, _) = self.jagg_rnn(skeleton_agg, (h0_agg, c0_agg))
        skeleton_agg = torch.concat([skeleton_agg, g], dim=-1)

        skeleton_f, (_, _) = self.jrnn(skeleton_f, (h0_joint, c0_joint))  # [Batch_Size * N_Joints, n_frames,-1]

        skeleton_f = skeleton_f.reshape(b, n_joints, n_frames, -1).permute(0, 2, 1, 3).reshape(b, n_frames, -1)

        return self.ln(torch.concat([skeleton_agg, skeleton_f], dim=-1))


class mmPosePlus(nn.Module):
    def __init__(self, cfg: ConfigPlus):
        super().__init__()
        self.cfg = cfg
        self.base_module = PointNet(cfg, in_features=self.cfg.in_features,
                                    hidden_features_1=8,
                                    hidden_features_2=16,
                                    output_features=self.cfg.out_features_BasePointNet,
                                    attention=False)
        self.coarse_skeleton = CoarseSkeleton(cfg)
        self.pose_aware_joint = Pose_Aware_Joint(cfg)
        self.pose_estimator = PoseEstimator(cfg)

    def forward(self, x, h0_coarse, c0_coarse, h0_agg, c0_agg, h0_joints, c0_joints,):
        """
        :param x: [Batch_size,N_frames,N_points,5]
        :return: [Batch_size,N_frames,51]
        """
        b, n_frames, n_points, _ = x.shape
        x = x.view(b * n_frames, n_points, -1)
        x = self.base_module(x)

        skeleton, edge, g = self.coarse_skeleton(x, h0_coarse, c0_coarse, b, n_frames)

        x = x.view(b, n_frames, n_points, -1)

        skeleton_feature = self.pose_aware_joint(x, g, skeleton, edge, self.cfg.n_nearest)

        pose = self.pose_estimator(skeleton_feature, g, h0_agg, c0_agg, h0_joints, c0_joints)
        return pose


if __name__ == '__main__':
    cfg = ConfigPlus()
    model = mmPosePlus(cfg).cuda()
    x = torch.rand(2, 64, 16, 5).cuda()

    h0_coarse = torch.zeros(cfg.n_layers_RNN, 2, cfg.out_feature_GlobalPointNet).cuda()
    c0_coarse = torch.zeros(cfg.n_layers_RNN, 2, cfg.out_feature_GlobalPointNet).cuda()
    h0_agg = torch.zeros(cfg.n_layers_RNN, 2, cfg.out_feature_GCN).cuda()
    c0_agg = torch.zeros(cfg.n_layers_RNN, 2, cfg.out_feature_GCN).cuda()
    h0_joints = torch.zeros(cfg.n_layers_RNN, 2 * cfg.n_joints, cfg.out_feature_GCN).cuda()
    c0_joints = torch.zeros(cfg.n_layers_RNN, 2 * cfg.n_joints, cfg.out_feature_GCN).cuda()

    pose = model(x, h0_coarse, c0_coarse, h0_agg, c0_agg, h0_joints, c0_joints)
    print(pose.shape)
