import os.path
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data_root = ''
PointOdyssey3D_root = data_root + ''

class LongTermSceneFlowDataset(Dataset):
    def __init__(self, aug_params=None, track_mode='frame_first', train_mode=False):

        self.track_mode = track_mode
        self.train_mode = train_mode

        self.augmentor = None

        self.meta_list = []

    def get_data_unit(self, index):
        return {}

    def __getitem__(self, index):
        data_invalid = True
        while data_invalid:
            index = index % len(self.meta_list)            
            du = self.get_data_unit(index)
           
            if self.augmentor is not None:
                du = self.augmentor(du)                

            outputs, data_invalid, index = self.prepare_output(du)           

        return outputs

    def __len__(self):
        return len(self.meta_list)

    def __rmul__(self, v):
        self.meta_list = v * self.meta_list
        return self

    def prepare_output(self, du):

        visibs = torch.from_numpy(du['visibs']).squeeze(-1)  # shape: (T, N_total)
        valids = torch.from_numpy(du['valids']).squeeze(-1)  # shape: (T, N_total)
        trajs_3d = torch.from_numpy(du['target_points_3d'])  # (T, N, 3)
        video_pc = torch.from_numpy(du['video_pc']).float()  # (T, N, 3)

        trajs_3d[..., :2] = -trajs_3d[..., :2]
        video_pc[..., :2] = -video_pc[..., :2]

        if 'video_pc_rgb' in du:
            video_pc_rgb = torch.from_numpy(du['video_pc_rgb']).float()   # (T, N, 6)
            

        seq_len = du['seq_len']
        track_point_num = du['track_point_3d_num']

        data_invalid = False
        index = None

        visibile_pts_first_frame_inds = (visibs[0]).nonzero(as_tuple=False)[:, 0]
        visibile_pts_inds = visibile_pts_first_frame_inds

        if self.train_mode:
            point_inds = torch.randperm(len(visibile_pts_inds))[: track_point_num]

            if len(point_inds) < track_point_num:
                data_invalid = True
                index = np.random.randint(0, len(self.meta_list))
                print('track_point_num:',track_point_num,'len(point_inds):',len(point_inds))
        else:
            point_inds = list(range(0, len(visibile_pts_inds), 1))[: track_point_num]
            
        visible_inds_sampled = visibile_pts_inds[point_inds]
        trajs_3d = trajs_3d[:, visible_inds_sampled].float()  # shape: (T, N, 3)
        visibs = visibs[:, visible_inds_sampled]  # shape: (T, N)
        valids = valids[:, visible_inds_sampled]  # shape: (T, N)

        outputs = {
            'video_pc': video_pc.permute(0,2,1),  # shape: (T, 3, N_all)
            'visibs': visibs,  # shape: (T, N)
            'valids': valids,  # shape: (T, N)
            'trajs_3d': trajs_3d,  # shape: (T, N, 3)
        }
        if 'query_points_3d' in du:
            qurey_3d = trajs_3d[0].clone()  # shape: (N, 3)
            query_points_t = torch.zeros_like(qurey_3d[:, 0:1])  # shape: (N, 1) 
            query_points_3d_t = torch.cat([query_points_t, qurey_3d], dim=-1)  # shape: (N, 4)
            outputs['query_points_3d'] = query_points_3d_t  # shape: (N, 4)
        if 'seq_name' in du: outputs['seq_name'] = du['seq_name']
        if 'intris' in du: outputs['intris'] = torch.from_numpy(du['intris'])
        if 'extris' in du: outputs['extris'] = torch.from_numpy(du['extris'])
        if self.train_mode is False and 'video_pc_rgb' in du:
            outputs['video_pc_rgb'] = video_pc_rgb.permute(0,2,1)
            outputs['path'] = du['path']

        return outputs, data_invalid, index

class PointOdyssey3D(LongTermSceneFlowDataset):
    def __init__(self, aug_params=None, root=PointOdyssey3D_root, seq_len=40, track_point_num=128, split='train', train_mode=False):
        super(PointOdyssey3D, self).__init__(aug_params, train_mode=train_mode)

        self.seq_len = seq_len
        self.track_point_num = track_point_num
        self.track_point_3d_num = track_point_num
        self.split = split
        data_root = os.path.join(root, split)
        seq_path_list = []
        for seq_path in glob(os.path.join(data_root, "*")):
            seq_path = seq_path.replace('\\', '/')
            if os.path.isdir(seq_path):
                seq_path_list.append(seq_path)
        seq_path_list = sorted(seq_path_list)

        for seq_path in seq_path_list:
            filenames = sorted(os.listdir(seq_path))
            for filename in filenames:
                if filename.endswith('.npz'):
                    full_path = os.path.join(seq_path, filename).replace('\\', '/')
                    self.meta_list.append({'npz_path': full_path})

    def get_data_unit(self, index):
        du = {}

        npz_path = self.meta_list[index]['npz_path']
        data = dict(np.load(npz_path, allow_pickle=True))
        
        du['target_points_3d'] = data['traj_seq']  # (T, N, 3)
        du['video_pc'] = data['pc_xyz_seq']  # (T, N, 3)
        du['visibs'] = data['visibs']
        du['valids'] = (data['valids'])  # (T, N)
        if self.split != 'train':
            du['video_pc_rgb'] = data['pc_xyz_rgb_seq']  # (T, num_video_pc, 6)
            du["path"] = npz_path


        du['seq_len'] = self.seq_len if self.seq_len != -1 else du['video_pc'].shape[0]
        

        track_3d = du['target_points_3d']
        query_points_3d = track_3d[0,:,:]   
        query_points_3d_t = np.zeros_like(query_points_3d[:, 0:1])  # shape: (N, 1)      
        query_points_3d = np.concatenate([query_points_3d_t, query_points_3d], axis=-1)  # shape: (N, 4)

        if self.split != 'train':
            du['query_points_3d'] = query_points_3d  # shape: (N, 4)
        du['track_point_3d_num'] = self.track_point_3d_num if self.track_point_3d_num != -1 else 1024
        return du

from contextlib import contextmanager
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

def fetch_dataloader(config):
    def prepare_data(config):

        if config.stage == 'odyssey':
            aug_params = None
            dataset = PointOdyssey3D(aug_params, seq_len=config.seq_len, train_mode=True, track_point_num=config.track_point_num)

        return dataset

    if config.is_ddp:
        with torch_distributed_zero_first(config.rank):
            dataset = prepare_data(config)
    else:
        dataset = prepare_data(config)

    batch_size_tmp = config.batch_size // config.world_size

    dataloder = DataLoader(dataset,
                           batch_size=batch_size_tmp,
                           pin_memory=True,
                           sampler=torch.utils.data.distributed.DistributedSampler(dataset) if config.is_ddp else None,
                           num_workers=8 if config.is_ddp else 0,
                           drop_last=True)

    if config.is_master:
        print('Training with %d point cloud sequences' % len(dataset))

    return dataloder


