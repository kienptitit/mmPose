import torch
import torch.nn as nn
from Config import Config


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# mRI config
# def AnchorInit(x_min=-2, x_max=2, x_interval=1.6, y_min=-2, y_max=2, y_interval=1.6, z_min=-0.3, z_max=2.1,
#                z_interval=0.3):  # [z_size, y_size, x_size, npoint] => [9,3,3,3]
#     """
#     Input:
#         x,y,z min, max and sample interval
#     Return:
#         centroids: sampled controids [z_size, y_size, x_size, npoint] => [9,3,3,3]
#     """
#     x_size = round((x_max - x_min) / x_interval) + 1
#     y_size = round((y_max - y_min) / y_interval) + 1
#     z_size = round((z_max - z_min) / z_interval) + 1
#     centroids = torch.zeros((z_size, y_size, x_size, 3), dtype=torch.float32)
#     for z_no in range(z_size):
#         for y_no in range(y_size):
#             for x_no in range(x_size):
#                 lx = x_min + x_no * x_interval
#                 ly = y_min + y_no * y_interval
#                 lz = z_min + z_no * z_interval
#                 centroids[z_no, y_no, x_no, 0] = lx
#                 centroids[z_no, y_no, x_no, 1] = ly
#                 centroids[z_no, y_no, x_no, 2] = lz
#     return centroids


# mm-FI config
def AnchorInit(x_min=-2, x_max=2, x_interval=1.6, y_min=-2, y_max=4.5, y_interval=3, z_min=2.0, z_max=4.5,
               z_interval=0.3):  # [z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    Input:
        x,y,z min, max and sample interval
    Return:
        centroids: sampled controids [z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    x_size = round((x_max - x_min) / x_interval) + 1
    y_size = round((y_max - y_min) / y_interval) + 1
    z_size = round((z_max - z_min) / z_interval) + 1
    centroids = torch.zeros((z_size, y_size, x_size, 3), dtype=torch.float32)
    for z_no in range(z_size):
        for y_no in range(y_size):
            for x_no in range(x_size):
                lx = x_min + x_no * x_interval
                ly = y_min + y_no * y_interval
                lz = z_min + z_no * z_interval
                centroids[z_no, y_no, x_no, 0] = lx
                centroids[z_no, y_no, x_no, 1] = ly
                centroids[z_no, y_no, x_no, 2] = lz
    return centroids


def index_points(points, idx):
    """
    return Correspond Coordinate Of [B,N_anchor,N_Samples]
    Input:
        points: input points data, [B,N, C]
        idx: sample index data, [B,S,N_samples]
    Return:
        new_points:, indexed points data, [B, S,N_samles,C]
    """
    B, N, C = points.shape
    _, S, _ = idx.shape
    points = points.unsqueeze(1).repeat(1, S, 1, 1)
    idx = idx.unsqueeze(-1).repeat(1, 1, 1, C)
    return torch.gather(points, dim=2, index=idx)


def point_ball_set(nsample, xyz, new_xyz, cfg=None):
    """
    With Each Anchor Point, Take Its Nsample Closest Point Cloud.
    Input:
        nsample: number of points to sample
        xyz: all points, [B, N, 3]
        new_xyz: anchor points [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).view(1, 1, N).repeat(B, S, 1).to(cfg.device)

    distance = square_distance(new_xyz, xyz)  # [B,S,N]
    _, sort_index = torch.sort(distance)
    sort_index = sort_index[:, :, :nsample]
    return torch.gather(group_idx, dim=-1, index=sort_index)


def AnchorGrouping(anchors, nsample, xyz, points, cfg=None):
    """
    Input:
        anchors: [B, 9*3*3, 3], npoint=9*3*3
        nsample: number of points to sample
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D] # Point Features After PointNet
    Return:
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """

    B, N, C = xyz.shape
    _, S, _ = anchors.shape
    idx = point_ball_set(nsample, xyz, anchors, cfg)  # nsamples nearest Point Clound of each Anchor Point
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]

    grouped_anchors = anchors.view(B, S, 1, C).repeat(1, 1, nsample, 1)
    grouped_xyz_norm = grouped_xyz - grouped_anchors  # With Each Anchor Compute Its Distance To Nsample Selected Point

    grouped_points = index_points(points, idx)

    new_points = torch.cat([grouped_anchors, grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+C+D]
    return new_points


class AnchorPointNet(nn.Module):
    def __init__(self, cfg: Config = None):
        super(AnchorPointNet, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv1d(in_channels=self.cfg.out_feauture_BaseModule + self.cfg.coordinate_feature + 3,
                               out_channels=32, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=self.cfg.out_feature_AnchorPointNet, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(self.cfg.out_feature_AnchorPointNet)
        self.caf3 = nn.ReLU()

        self.attn = nn.Linear(self.cfg.out_feature_AnchorPointNet, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: (Batch_Size,N_point,n_features)
        :return:
        """
        x = x.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))  # (Batch, feature, frame_point_number)

        x = x.transpose(1, 2)

        attn_weights = self.softmax(self.attn(x))
        attn_vec = torch.sum(x * attn_weights, dim=1)
        return attn_vec, attn_weights


class AnchorVoxelNet(nn.Module):
    def __init__(self, cfg: Config = None):
        super(AnchorVoxelNet, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv3d(in_channels=self.cfg.out_feature_AnchorPointNet, out_channels=96, kernel_size=(3, 3, 3),
                               padding=(0, 0, 0))
        self.cb1 = nn.BatchNorm3d(96)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(5, 1, 1))
        self.cb2 = nn.BatchNorm3d(128)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=self.cfg.out_feature_AnchorVoxelNet, kernel_size=(3, 1, 1))
        self.cb3 = nn.BatchNorm3d(self.cfg.out_feature_AnchorVoxelNet)
        self.caf3 = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.permute(0, 4, 1, 2, 3)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.view(batch_size, self.cfg.out_feature_AnchorVoxelNet)
        return x


class AnchorRNN(nn.Module):
    def __init__(self, cfg: Config = None):
        super(AnchorRNN, self).__init__()
        self.cfg = cfg

        self.rnn = nn.LSTM(input_size=self.cfg.out_feature_AnchorVoxelNet, hidden_size=64,
                           num_layers=self.cfg.numlayers_RNN,
                           batch_first=True, dropout=self.cfg.dropout_RNN,
                           bidirectional=False)

    def forward(self, x, h0, c0):
        a_vec, (hn, cn) = self.rnn(x, (h0, c0))
        return a_vec, hn, cn


class AnchorModule(nn.Module):
    def __init__(self, cfg: Config = None):
        super(AnchorModule, self).__init__()
        self.cfg = cfg

        self.template_point = AnchorInit()

        self.z_size, self.y_size, self.x_size, _ = self.template_point.shape
        self.anchor_size = self.z_size * self.y_size * self.x_size  # Number of anchor_size
        self.apointnet = AnchorPointNet(cfg)
        self.avoxel = AnchorVoxelNet(cfg)
        self.arnn = AnchorRNN(cfg)

    def forward(self, x, g_loc, h0, c0, batch_size, length_size, feature_size):
        """
        This module aggregtate the information of all point in frames into vector.
        Seq_length = N_frames
        :param x: Point_Features [Batch_Size * N_Frames. N_points, N_features]
        :param g_loc: Predicted Displacement Of Anchor Point In Each Frame [Batch_Size,N_Frames,2]
        :param h0: Init Hidden State
        :param c0: Init Cell State
        :param batch_size: B
        :param length_size: N_frames
        :param feature_size: Point Features
        :return: local feature for each frame
        """
        g_loc = g_loc.view(batch_size * length_size, 1, 2).repeat(1, self.anchor_size, 1)  # Displacement Volume
        anchors = self.template_point.view(1, self.anchor_size, 3).repeat(batch_size * length_size, 1,
                                                                          1).to(self.cfg.device)  # Anchor Init
        anchors[:, :, :2] += g_loc  # Add Displayment

        grouped_points = AnchorGrouping(anchors, nsample=self.cfg.n_nearest_anchor_point, xyz=x[..., :3],
                                        points=x[..., 3:],
                                        cfg=self.cfg)  # [Batch_Size * Seq_Length,Anchor_Size,N_Samples,C]
        """
        x[...,:3]: Coordinate of each Point Cloud
        x[...,3:]: Another information of Point Cloud from mmWare device
        """
        grouped_points = grouped_points.view(batch_size * length_size * self.anchor_size,
                                             self.cfg.n_nearest_anchor_point, 3 + feature_size)

        voxel_points, attn_weights = self.apointnet(grouped_points)

        voxel_points = voxel_points.view(batch_size * length_size, self.z_size, self.y_size, self.x_size,
                                         self.cfg.out_feature_AnchorPointNet)
        """
        voxel_points: [Batch_Size * Length_Size , Z_Size, Y_size, X_size, Feature_Size]
        attn_weight: [Batch_Size * Length_Size * Anchor_Size,n_samples]
        """
        voxel_vec = self.avoxel(voxel_points)
        voxel_vec = voxel_vec.view(batch_size, length_size, self.cfg.out_feature_AnchorVoxelNet)
        """
        voxel_vec : [Batch_Size, Seq_length,Feature_Size]
        """
        a_vec, hn, cn = self.arnn(voxel_vec, h0, c0)
        return a_vec, attn_weights, hn, cn


class BasePointNet(nn.Module):
    def __init__(self, cfg: Config = None):
        super(BasePointNet, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv1d(in_channels=self.cfg.in_features, out_channels=8, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(8)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(16)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=self.cfg.out_feauture_BaseModule, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(self.cfg.out_feauture_BaseModule)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        """
        :param x: Input Point From Raw Data, [batch_size * n_frames,n_points,6]
        :return: a high level features of points, [batch_size,n_frames,n_points,28]
        """
        x = in_mat.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1, 2)
        x = torch.cat((in_mat[:, :, :self.cfg.coordinate_feature], x), -1)  # [:,:,:4] is X,y,z coordinate, Range Value

        return x


class GlobalPointNet(nn.Module):
    def __init__(self, cfg: Config = None):
        super(GlobalPointNet, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv1d(in_channels=self.cfg.out_feauture_BaseModule + self.cfg.coordinate_feature,
                               out_channels=32, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=self.cfg.out_feature_GlobalModule, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(self.cfg.out_feature_GlobalModule)
        self.caf3 = nn.ReLU()

        self.attn = nn.Linear(self.cfg.out_feature_GlobalModule, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: output from BaseModule [Batch_Size * Seq_length,N_points,n_features]
        :return: Global Features Of A Frame [Batch_Size * Seq_length,n_features]
        """
        x = x.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1, 2)

        attn_weights = self.softmax(self.attn(x))  # Weight For Each Point In A Frames

        attn_vec = torch.sum(x * attn_weights, dim=1)
        return attn_vec, attn_weights


class GlobalRNN(nn.Module):
    def __init__(self, cfg: Config = None):
        super(GlobalRNN, self).__init__()
        self.cfg = cfg

        self.rnn = nn.LSTM(input_size=self.cfg.out_feature_GlobalModule, hidden_size=64,
                           num_layers=self.cfg.numlayers_RNN, batch_first=True,
                           dropout=self.cfg.dropout_RNN,
                           bidirectional=False)
        self.fc1 = nn.Linear(64, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, h0, c0):
        g_vec, (hn, cn) = self.rnn(x, (h0, c0))
        g_loc = self.fc1(g_vec)
        g_loc = self.faf1(g_loc)
        g_loc = self.fc2(g_loc)
        return g_vec, g_loc, hn, cn


class GlobalModule(nn.Module):
    def __init__(self, cfg):
        super(GlobalModule, self).__init__()
        self.cfg = cfg

        self.gpointnet = GlobalPointNet(cfg=cfg)
        self.grnn = GlobalRNN(cfg=cfg)

    def forward(self, x, h0, c0, batch_size, length_size):
        x, attn_weights = self.gpointnet(x)
        x = x.view(batch_size, length_size, self.cfg.out_feature_GlobalModule)
        g_vec, g_loc, hn, cn = self.grnn(x, h0, c0)
        return g_vec, g_loc, attn_weights, hn, cn


class CombineModule(nn.Module):
    def __init__(self):
        super(CombineModule, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 9 * 6 + 3 + 10 + 1)

    def forward(self, g_vec, a_vec, batch_size, length_size):
        x = torch.cat((g_vec, a_vec), -1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)

        q = x[:, :, :9 * 6].reshape(batch_size * length_size * 9, 6).contiguous()
        tmp_x = nn.functional.normalize(q[:, :3], dim=-1)
        tmp_z = nn.functional.normalize(torch.cross(tmp_x, q[:, 3:], dim=-1), dim=-1)
        print(tmp_z[0])
        exit()
        tmp_y = torch.cross(tmp_z, tmp_x, dim=-1)

        tmp_x = tmp_x.view(batch_size, length_size, 9, 3, 1)
        tmp_y = tmp_y.view(batch_size, length_size, 9, 3, 1)
        tmp_z = tmp_z.view(batch_size, length_size, 9, 3, 1)
        q = torch.cat((tmp_x, tmp_y, tmp_z), -1)

        t = x[:, :, 9 * 6:9 * 6 + 3]
        b = x[:, :, 9 * 6 + 3:9 * 6 + 3 + 10]
        g = x[:, :, 9 * 6 + 3 + 10:]
        return q, t, b, g


class mmWaveModel(nn.Module):
    def __init__(self, cfg: Config = None):
        super(mmWaveModel, self).__init__()
        self.cfg = cfg

        self.module0 = BasePointNet(cfg=self.cfg)
        self.module1 = GlobalModule(cfg=self.cfg)
        self.module2 = AnchorModule(cfg=self.cfg)

        self.ln = nn.Linear(64 + 64, self.cfg.out_feature_mmWaveModel)
        # self.module3 = CombineModule()

    def forward(self, x, h0_g, c0_g, h0_a, c0_a):
        batch_size = x.size()[0]
        length_size = x.size()[1]
        pt_size = x.size()[2]  # N_points
        in_feature_size = x.size()[3]
        out_feature_size = self.cfg.out_feauture_BaseModule + self.cfg.coordinate_feature

        x = x.view(batch_size * length_size, pt_size, in_feature_size)
        x = self.module0(x)  # High level feature of points

        g_vec, g_loc, global_weights, hn_g, cn_g = self.module1(x, h0_g, c0_g, batch_size, length_size)
        """
        g_vec: (Batch_Size,Seq_Length,N_features)
        """
        a_vec, anchor_weights, hn_a, cn_a = self.module2(x, g_loc, h0_a, c0_a, batch_size, length_size,
                                                         out_feature_size)

        """
        a_vec: (Batch_Size,Seq_Length,N_features)
        """
        v = torch.concat([g_vec, a_vec], dim=-1)
        return self.ln(v)


if __name__ == '__main__':
    cfg = Config()
    # Anchor Module
    # data = torch.rand((7 * 13, 50, 24 + 3), dtype=torch.float32,
    #                   device='cpu')  # [Batch_Size * Seq_Length,N_Points,N_Features]
    # g_loc = torch.full((7, 13, 2), 100.0, dtype=torch.float32,
    #                    device='cpu')  # A displayment of anchor point (It's output of global module
    # h0 = torch.zeros((3, 7, cfg.out_feature_GlobalModule), dtype=torch.float32,
    #                  device='cpu')  # [N_layers,Batch_Size,N_features]
    # c0 = torch.zeros((3, 7, cfg.out_feature_GlobalModule), dtype=torch.float32, device='cpu')
    # model = AnchorModule(cfg)
    # a, w, hn, cn = model(data, g_loc, h0, c0, 7, 13, cfg.out_feauture_BaseModule + cfg.coordinate_feature)
    # exit()
    # # Anchor Module
    #
    # # BasePoint Net (Carefull In Forward Pass, need to be adapt)
    # data = torch.rand((7 * 13, 50, 6), dtype=torch.float32, device='cpu')
    # model = BasePointNet()
    # x = model(data)
    # print(x.shape)
    # # BasePoint Net (Carefull In Forward Pass, need to be adapt)
    #
    # # GlobalPoint Net
    # data = torch.rand((7 * 13, 50, 24 + 4), dtype=torch.float32, device='cpu')
    # print('\tInput:', data.shape)
    # model = GlobalPointNet()
    # x, w = model(data)
    # # Globalpoint Net
    #
    # # Global RNN
    # data = torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
    # h0 = torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    # c0 = torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    # model = GlobalRNN()
    # g, l, hn, cn = model(data, h0, c0)
    # # Global RNN
    #
    # # Combine Module
    # g = torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
    # a = torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
    # model = CombineModule()
    # q, t, b, g = model(g, a, 7, 13)
    # print('\tOutput:', q.shape, t.shape, b.shape, g.shape)
    # # Combine Module

    # Whole

    for _ in range(100):
        data = torch.rand((2, 64, 196, 5), dtype=torch.float32, device='cuda')
        inp_0 = data[0].unsqueeze(0)
        h0 = torch.zeros((3, 2, 64), dtype=torch.float32, device='cuda')
        c0 = torch.zeros((3, 2, 64), dtype=torch.float32, device='cuda')
        h1 = torch.zeros((3, 1, 64), dtype=torch.float32, device='cuda')
        c1 = torch.zeros((3, 1, 64), dtype=torch.float32, device='cuda')
        model = mmWaveModel(cfg=cfg).to('cuda')
        model.eval()
        out = model(data, h0, c0, h0, c0)
        out_ = model(inp_0, h1, c1, h1, c1)
        print((out - out_).sum())
    # Whole
