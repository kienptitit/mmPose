from datetime import datetime


class Config:
    def __init__(self, args):
        self.args = args
        self.init_model_params()
        self.init_dataset_param()
        self.init_training_param()

    def init_model_params(self):
        self.in_features = 5
        self.coordinate_feature = 3
        self.out_feauture_BaseModule = 24
        self.out_feature_GlobalModule = 64
        self.n_nearest_anchor_point = 3
        self.out_feature_AnchorPointNet = 64
        self.out_feature_AnchorVoxelNet = 64
        self.numlayers_RNN = 1
        self.dropout_RNN = 0.0
        self.out_feature_mmWaveModel = 3 * 17

    def init_dataset_param(self):
        if self.args.dataset_name == 'mRI':
            self.train_seq_length = 64
            self.test_seq_length = 64
        elif self.args.dataset_name == 'mm-Fi':
            self.train_seq_length = 16
            self.test_seq_length = 16

        self.in_features_raw_data = 5
        self.anno_dir = "/home/naver/Documents/Kien/mRI/Dataset/dataset_release/aligned_data/pose_labels"
        self.radar_dir = "/home/naver/Documents/Kien/mRI/Dataset/dataset_release/features/radar"

    def init_training_param(self):
        # date = "___".join(str(datetime.now()).split(" "))

        self.write_log = True
        if self.args.dataset_name == 'mRI':
            self.batch_size = 16
        elif self.args.dataset_name == 'mm-Fi':
            self.batch_size = 1

        self.weight_decay = 0.01
        self.T_max = 30
        self.epochs = 200
        self.save_model_frequency = 10
        self.save_path = '/home/naver/Documents/Kien/mRI/mrPose/Logs'
        self.log_path = f"/home/naver/Documents/Kien/mRI/mrPose/Training_Logs"
        self.save_fig_dir = '/home/naver/Documents/Kien/mRI/mrPose/Figure'
        self.save_video = '/home/naver/Documents/Kien/mRI/mrPose/Video_Result_All_Frames'
        self.device = 'cuda'


class ConfigPlus:
    def __init__(self, args):
        self.args = args
        self.init_model_hyper_parameters()
        self.init_default_path()
        self.init_training_param()
        self.init_dataset_param()

    def init_dataset_param(self):
        if self.args.dataset_name == 'mRI':
            self.train_seq_length = 64
            self.test_seq_length = 64
        elif self.args.dataset_name == 'mm-Fi':
            self.train_seq_length = 16
            self.test_seq_length = 16

        self.in_features_raw_data = 5
        self.anno_dir = "/home/naver/Documents/Kien/mRI/Dataset/dataset_release/aligned_data/pose_labels"
        self.radar_dir = "/home/naver/Documents/Kien/mRI/Dataset/dataset_release/features/radar"

    def init_default_path(self):
        self.smpl_path = '/home/naver/Documents/Kien/mRI/mrPose/SMPL_python_v.1.0.0-20240314T040659Z-001/SMPL_python_v.1.0.0/smpl/models'

    def init_model_hyper_parameters(self):
        self.in_features = 5
        self.out_features_BasePointNet = 24
        self.coordinate_features = 3
        self.out_feature_GlobalPointNet = 64
        self.dropout_RNN = 0.0
        self.n_layers_RNN = 1
        self.in_feature_SMPL = 10
        self.out_feature_JointPointNet = 64
        self.out_feature_GCN = 128
        self.gcn_layers = 3
        self.n_joints = 24
        self.n_nearest = 1

    def init_training_param(self):

        self.write_log = True
        if self.args.dataset_name == 'mRI':

            self.batch_size = 16
        elif self.args.dataset_name == 'mm-Fi':

            self.batch_size = 1

        self.write_log = True
        self.weight_decay = 0.01
        self.T_max = 30
        self.epochs = 200
        self.save_model_frequency = 10
        self.save_path = '/home/naver/Documents/Kien/mRI/mrPose/Logs'
        self.log_path = f"/home/naver/Documents/Kien/mRI/mrPose/Training_Logs"
        self.save_fig_dir = '/home/naver/Documents/Kien/mRI/mrPose/Figure'
        self.save_video = '/home/naver/Documents/Kien/mRI/mrPose/Video_Result_All_Frames'
        self.device = 'cuda'
