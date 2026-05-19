import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.extractor import PointConvEncoderLight
from .block import UpdateFormer
from .corr import CorrBlock
from .embeddings import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_embedding,
)
from .losses import sequence_loss
from .util import bilinear_sample2d, smart_cat, knn_interpolate_feats

autocast = torch.cuda.amp.autocast
enable_autocast = False


def sample_pos_embed(grid_size, embed_dim, coords):
    pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size)
    pos_embed = (
        torch.from_numpy(pos_embed)
        .reshape(grid_size[0], grid_size[1], embed_dim)
        .float()
        .unsqueeze(0)
        .to(coords.device)
    )
    sampled_pos_embed = bilinear_sample2d(
        pos_embed.permute(0, 3, 1, 2), coords[:, 0, :, 0], coords[:, 0, :, 1]
    )
    return sampled_pos_embed


class TrackerIterationModule(nn.Module):

    def __init__(
        self,
        latent_dim=128,
        hidden_size=384,
        additional_dim=2,
        space_depth=6,
        time_depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        add_space_attn=True,
        enable_autocast=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.additional_dim = additional_dim
        self.enable_autocast = enable_autocast
        self.corr_levels = 3
        self.base_scales = 0.25
        self.truncate_k = 512
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.add_space_attn = add_space_attn
        self.hidden_size = hidden_size
        self.space_depth = space_depth
        self.time_depth = time_depth
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 358 + self.additional_dim)
        )
        self.fcorr_fn = CorrBlock(
            num_levels=self.corr_levels,
            base_scale=self.base_scales,
            resolution=3,
            truncate_k=self.truncate_k,
        )
        self.updateformer = nn.Sequential(
            UpdateFormer(
                space_depth=self.space_depth,
                time_depth=self.time_depth,
                input_dim=358 + self.additional_dim,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                output_dim=latent_dim + 3,
                mlp_ratio=self.mlp_ratio,
                add_space_attn=self.add_space_attn,
            )
        )
        self.norm = nn.GroupNorm(1, latent_dim)
        self.ffeat_updater = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU())
        self.vis_predictor = nn.Sequential(nn.Linear(latent_dim, 1))

    def forward(
        self,
        fmaps,
        pc_xyz,
        coords_init,
        feat_init=None,
        vis_init=None,
        track_mask=None,
        iters=4,
    ):
        B, S_init, N, D = coords_init.shape
        assert D == 3
        assert B == 1
        xyz2 = pc_xyz.permute(0, 1, 3, 2)
        B, S, __, N_all = fmaps.shape
        device = fmaps.device
        if S_init < S:
            coords = torch.cat(
                [coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            vis_init = torch.cat(
                [vis_init, vis_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
        else:
            coords = coords_init.clone()
        ffeats = feat_init.clone()
        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1)
        pose_embed = (
            self.pos_embed(coords.reshape(B * S, N, 3))
            .reshape(B, S, N, -1)
            .permute(0, 2, 1, 3)
            .reshape(B * N, S, -1)
        )
        times_embed = (
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(358 + self.additional_dim, times_[0])
            )[None]
            .repeat(B, 1, 1)
            .float()
            .to(device)
        )
        coord_predictions = []
        for __ in range(iters):
            coords = coords.detach()
            self.fcorr_fn.init_module(ffeats, fmaps, xyz2)
            fcorrs = self.fcorr_fn(coords).permute(0, 1, 3, 2)
            LRR = fcorrs.shape[3]
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 3)
            flows_cat = get_3d_embedding(flows_, 32, cat_coords=True)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)
            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0]).repeat(
                            1, vis_init.shape[1] - track_mask.shape[1], 1, 1
                        ),
                    ],
                    dim=1,
                )
            concat = (
                torch.cat([track_mask, vis_init], dim=2)
                .permute(0, 2, 1, 3)
                .reshape(B * N, S, 2)
            )
            transformer_input = torch.cat([flows_cat, fcorrs_, ffeats_, concat], dim=2)
            x = transformer_input + pose_embed + times_embed
            x = rearrange(x, "(b n) t d -> b n t d", b=B)
            with autocast(enabled=self.enable_autocast):
                delta = self.updateformer(x)
                delta = rearrange(delta, " b n t d -> (b n) t d")
                delta_coords_ = delta[:, :, :3]
                delta_feats_ = delta[:, :, 3:]
                delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
                ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)
                ffeats_ = self.ffeat_updater(self.norm(delta_feats_)) + ffeats_
                ffeats = ffeats_.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)
                coords = coords + delta_coords_.reshape(B, N, S, 3).permute(0, 2, 1, 3)
                coord_predictions.append(coords)
        with autocast(enabled=self.enable_autocast):
            vis_e = self.vis_predictor(
                ffeats.reshape(B * S * N, self.latent_dim)
            ).reshape(B, S, N)
        return (coord_predictions, vis_e.float(), feat_init)


class PCSTracker(nn.Module):

    def __init__(self, config=None):
        super(PCSTracker, self).__init__()
        self.config = config
        hidden_size = 256
        space_depth = 3
        time_depth = 3
        self.latent_dim = latent_dim = 128
        self.additional_dim = 2
        num_heads = 4
        mlp_ratio = 2.0
        add_space_attn = True
        self.S = 16
        self.enable_autocast = enable_autocast
        weightnet = 8
        self.fnet = PointConvEncoderLight(weightnet=weightnet)
        self.iter_module = TrackerIterationModule(
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            additional_dim=self.additional_dim,
            space_depth=space_depth,
            time_depth=time_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            add_space_attn=add_space_attn,
            enable_autocast=self.enable_autocast,
        )

    def forward(self, video_pc, queries, iters=4, feat_init=None, is_train=False):
        B, T, C, N_all = video_pc.shape
        B, N, _ = queries.shape
        device = queries.device
        assert B == 1
        first_positive_inds = queries[:, :, 0].long()
        __, sort_inds = torch.sort(first_positive_inds[0], dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[0][sort_inds]
        assert torch.allclose(
            first_positive_inds[0], first_positive_inds[0][sort_inds][inv_sort_inds]
        )
        coords_init_3d = queries[:, :, 1:4]
        coords_init = coords_init_3d.view(B, 1, N, 3).repeat(1, self.S, 1, 1)
        traj_xyz_e = torch.zeros((B, T, N, 3), device=device)
        ind_array = torch.arange(T, device=device)[None, :, None].repeat(B, 1, N)
        track_mask = (ind_array >= first_positive_inds[:, None, :]).unsqueeze(-1)
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10
        ind = 0
        track_mask_ = track_mask[:, :, sort_inds].clone()
        coords_init_ = coords_init[:, :, sort_inds].clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()
        prev_wind_idx = 0
        fmaps_ = None
        coord_predictions = []
        wind_inds = []
        while ind < T - self.S // 2:
            pc_seq = video_pc[:, ind : ind + self.S]
            if pc_seq.device == torch.device("cpu"):
                pc_seq = pc_seq.cuda()
            S = S_local = pc_seq.shape[1]
            if S < self.S:
                pad_pc = pc_seq[:, -1:, :, :].repeat(1, self.S - S, 1, 1)
                pc_seq = torch.cat([pc_seq, pad_pc], dim=1)
                S = pc_seq.shape[1]
            pc_seq_ = pc_seq.reshape(B * S, C, N_all)
            if fmaps_ is None:
                with autocast(enabled=self.enable_autocast):
                    color = pc_seq_
                    pc_seq_levels, fmaps_levels, _ = self.fnet(pc_seq_, color)
                    fmaps_ = fmaps_levels[-1]
                    pc_seq_down_ = pc_seq_levels[-1]
            else:
                with autocast(enabled=self.enable_autocast):
                    color = pc_seq_[self.S // 2 :]
                    pc_seq_levels, fmaps_levels, _ = self.fnet(
                        pc_seq_[self.S // 2 :], color
                    )
                    fmaps_new = fmaps_levels[-1]
                    pc_seq_down_new = pc_seq_levels[-1]
                    fmaps_ = torch.cat([fmaps_[self.S // 2 :], fmaps_new], dim=0)
                    pc_seq_down_ = torch.cat(
                        [pc_seq_down_[self.S // 2 :], pc_seq_down_new], dim=0
                    )
            N_down = pc_seq_down_.shape[-1]
            fmaps_ = fmaps_.float()
            fmaps = fmaps_.reshape(B, S, self.latent_dim, N_down)
            pc_seq_down = pc_seq_down_.reshape(B, S, C, N_down)
            curr_wind_points = torch.nonzero(first_positive_sorted_inds < ind + self.S)
            if curr_wind_points.shape[0] == 0:
                ind = ind + self.S // 2
                continue
            wind_idx = curr_wind_points[-1] + 1
            if wind_idx - prev_wind_idx > 0:
                fmaps_sample = fmaps[
                    :, first_positive_sorted_inds[prev_wind_idx:wind_idx] - ind
                ]
                pc_seq_sample = pc_seq_down[
                    :, first_positive_sorted_inds[prev_wind_idx:wind_idx] - ind
                ]
                feat_init_ = knn_interpolate_feats(
                    fmaps_sample,
                    pc_seq_sample,
                    coords_init_[:, 0, prev_wind_idx:wind_idx, :],
                )
                feat_init_ = feat_init_.unsqueeze(1).repeat(1, self.S, 1, 1)
                feat_init = smart_cat(feat_init, feat_init_, dim=2)
            if prev_wind_idx > 0:
                new_coords = coords[-1][:, self.S // 2 :]
                coords_init_[:, : self.S // 2, :prev_wind_idx] = new_coords
                coords_init_[:, self.S // 2 :, :prev_wind_idx] = new_coords[
                    :, -1
                ].repeat(1, self.S // 2, 1, 1)
                new_vis = vis[:, self.S // 2 :].unsqueeze(-1)
                vis_init_[:, : self.S // 2, :prev_wind_idx] = new_vis
                vis_init_[:, self.S // 2 :, :prev_wind_idx] = new_vis[:, -1].repeat(
                    1, self.S // 2, 1, 1
                )
            coords, vis, __ = self.iter_module(
                fmaps=fmaps,
                pc_xyz=pc_seq,
                coords_init=coords_init_[:, :, :wind_idx],
                feat_init=feat_init[:, :, :wind_idx],
                vis_init=vis_init_[:, :, :wind_idx],
                track_mask=track_mask_[:, ind : ind + self.S, :wind_idx],
                iters=iters,
            )
            if is_train:
                coord_predictions.append([coord[:, :S_local] for coord in coords])
                wind_inds.append(wind_idx)
            traj_xyz_e[:, ind : ind + self.S, :wind_idx] = coords[-1][:, :S_local]
            track_mask_[:, : ind + self.S, :wind_idx] = 0.0
            ind = ind + self.S // 2
            prev_wind_idx = wind_idx
        traj_xyz_e = traj_xyz_e[:, :, inv_sort_inds]
        train_data = (coord_predictions, wind_inds, sort_inds) if is_train else None
        return (traj_xyz_e, feat_init, train_data)

    def Loss(self, train_data, gt_list):
        coord_predictions, wind_inds, sort_inds = train_data
        trajs_g, vis_g, valids = gt_list
        trajs_g = trajs_g[:, :, sort_inds]
        vis_g = vis_g[:, :, sort_inds]
        valids = valids[:, :, sort_inds]
        vis_gts = []
        traj_gts = []
        valids_gts = []
        for i, wind_idx in enumerate(wind_inds):
            ind = i * (self.S // 2)
            vis_gts.append(vis_g[:, ind : ind + self.S, :wind_idx])
            traj_gts.append(trajs_g[:, ind : ind + self.S, :wind_idx])
            valids_gts.append(valids[:, ind : ind + self.S, :wind_idx])
        seq_loss = sequence_loss(coord_predictions, traj_gts, vis_gts, valids_gts, 0.8)
        loss = seq_loss
        with torch.no_grad():
            query_traj_pr = coord_predictions[-1][-1][:, -1, :, :]
            query_traj_gt = trajs_g[:, -1, :, :]
            epe3d = torch.norm(query_traj_pr - query_traj_gt, dim=-1)
            epe3d = epe3d.view(-1)
            pc1_sum = (epe3d < 0.1).float().sum()
            pc2_sum = (epe3d < 0.2).float().sum()
            pc4_sum = (epe3d < 0.4).float().sum()
            epe3d_sum = epe3d.sum()
            valid_sum = torch.ones_like(epe3d_sum) * epe3d.numel()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(epe3d_sum)
                dist.all_reduce(pc1_sum)
                dist.all_reduce(pc2_sum)
                dist.all_reduce(pc4_sum)
                dist.all_reduce(valid_sum)
            epe3d_mean = epe3d_sum / valid_sum
            pc1 = pc1_sum / valid_sum
            pc2 = pc2_sum / valid_sum
            pc4 = pc4_sum / valid_sum
            metric_list = [
                ["epe", epe3d_mean.item()],
                ["pc1", pc1.item()],
                ["pc2", pc2.item()],
                ["pc4", pc4.item()],
            ]
        return (loss, metric_list)

    def infer(self, model, input_list, gt_list=None, iters=4, is_train=False):
        video_pc, queries = input_list
        predictions_xyz, feat_init, train_data = model(
            video_pc, queries, iters=iters, is_train=is_train
        )
        if gt_list != None:
            loss, metric_list = self.Loss(train_data, gt_list)
            return (loss, metric_list)
        return (predictions_xyz, feat_init, train_data)

    def training_infer(self, model, step_data, device):
        video_pc = step_data["video_pc"].to(device)
        trajs = step_data["trajs_3d"].to(device)
        vis_g = step_data["visibs"].to(device).float()
        valids = step_data["valids"].to(device).float()
        B, T, C, N_all = video_pc.shape
        assert C == 3
        B, T, N, D = trajs.shape
        __, first_positive_inds = torch.max(vis_g, dim=1)
        N_rand = N // 4
        nonzero_inds = [torch.nonzero(vis_g[0, :, i]) for i in range(N)]
        rand_vis_inds = torch.cat(
            [
                nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
                for nonzero_row in nonzero_inds
            ],
            dim=1,
        )
        first_positive_inds = torch.cat(
            [rand_vis_inds[:, :N_rand], first_positive_inds[:, N_rand:]], dim=1
        )
        ind_array_ = torch.arange(T, device=device)
        ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
        assert torch.allclose(
            vis_g[ind_array_ == first_positive_inds[:, None, :]], torch.ones_like(vis_g)
        )
        assert torch.allclose(
            vis_g[ind_array_ == rand_vis_inds[:, None, :]], torch.ones_like(vis_g)
        )
        gather = torch.gather(
            trajs, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, 3)
        )
        xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)
        queries = torch.cat([first_positive_inds[:, :, None], xys], dim=2)
        if hasattr(model, "module"):
            model = model.module
        loss, metric_list = model.infer(
            model,
            input_list=[video_pc, queries],
            gt_list=[trajs, vis_g, valids],
            iters=4,
            is_train=True,
        )
        return (loss, metric_list)
