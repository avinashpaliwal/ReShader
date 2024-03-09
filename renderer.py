# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import time
import imageio
import cv2
from PIL import ImageFile

import config
import math
import torchvision
from torchvision.transforms import Pad
import torch.utils.data.distributed
from tqdm import tqdm
from utils import *
from model_3dm import SpaceTimeModel
from core.utils import *
from core.renderer import ImgRenderer
from core.inpainter import Inpainter
from model import UNet, FCN
from posenc import get_embedder
ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_boundary_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    dilation = cv2.dilate(closing, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0.)
    return dilation


def get_input_data(args, ds_factor=1):
    to_tensor = torchvision.transforms.ToTensor()
    input_dir = args.input_dir
    img_file = (sorted(glob.glob(os.path.join(input_dir, 'input.png'))) + \
                sorted(glob.glob(os.path.join(input_dir, 'input.jpg'))))[0]
    src_img = imageio.imread(img_file) / 255.
    h1, w1 = src_img.shape[:2]

    disparity = (np.abs(np.load(os.path.join(input_dir, 'disp.npy'))))
    disparity = np.maximum(disparity, 1e-3)
    src_depth = (65535/disparity) / args.dscl

    fy = w1 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    fx = fy

    intrinsic = np.array([[fx, 0, w1 // 2],
                          [0, fy, h1 // 2],
                          [0, 0, 1]])

    pose = np.eye(4)
    return {
        'src_img1': to_tensor(src_img).float()[None],
        'src_depth1': to_tensor(src_depth).float()[None],
        'intrinsic1': torch.from_numpy(intrinsic).float()[None],
        'tgt_intrinsic': torch.from_numpy(intrinsic).float()[None],
        'pose': torch.from_numpy(pose).float()[None],
        'scale_shift1': torch.tensor([1., 0.]).float()[None],
        'src_rgb_file1': [img_file],
        'multi_view': [False]
    }

def reshading(reshader, dir_model, rgbd, xyz):

    xyz = xyz[None] * torch.tensor([-1, 1, 1]).cuda()
    pos_feat = dir_model(xyz)
    pred = reshader(rgbd, pos_feat)
    pred = torch.clamp(pred, 0, 1)

    return pred

def render(args):
    to_tensor = torchvision.transforms.ToTensor()
    device = "cuda:{}".format(args.local_rank)

    print('========================= Reshading Init...=========================')
    runpath = "runs/"

    reshader = UNet(14, 3).cuda()
    dir_model = FCN().cuda()
    reshader.load_state_dict(torch.load(f'{runpath}{args.model_dir}/model{args.ckpt}.pth'))
    dir_model.load_state_dict(torch.load(f'{runpath}{args.model_dir}/dir_model{args.ckpt}.pth'))
    embed_fn = get_embedder(args.pos_enc_freq)


    print('=========================run 3D Moments...=========================')

    data = get_input_data(args)
    rgb_file1 = data['src_rgb_file1'][0]
    frame_id1 = os.path.basename(rgb_file1).split('.')[0]
    scene_id = rgb_file1.split('/')[-3]

    video_out_folder = os.path.join(args.input_dir, 'out')
    os.makedirs(video_out_folder, exist_ok=True)

    im_h, im_w = data['src_img1'].shape[2:]
    pad_h, pad_w = (32 * math.ceil(im_h / 32) - im_h), (32 * math.ceil(im_w / 32) - im_w)
    padder = Pad(padding=(0, 0, pad_w, pad_h))

    model = SpaceTimeModel(args)
    if model.start_step == 0:
        raise Exception('no pretrained model found! please check the model path.')

    inpainter = Inpainter(args)
    renderer = ImgRenderer(args, model, None, inpainter, device)

    model.switch_to_eval()
    with torch.no_grad():
        renderer.process_data_single(data)

        pts1, rgb1, feat1, mask, side_ids = \
            renderer.render_rgbda_layers_from_one_view(return_pts=True)

        num_frames = 60#[60, 60, 60, 90]
        video_paths = ['circle']#['up-down', 'zoom-in', 'side', 'circle']
        Ts = [
            # define_camera_path(num_frames[0], 0., -0.08, 0., path_type='double-straight-line', return_t_only=True),
            # define_camera_path(num_frames[1], 0., 0., -0.24, path_type='straight-line', return_t_only=True),
            # define_camera_path(num_frames[2], -0.09, 0, -0, path_type='double-straight-line', return_t_only=True),
            # define_camera_path(num_frames, -0.15, -0.15, -0.15, path_type='circle', return_t_only=True),
            # define_camera_path(num_frames, -0.14, -0.14, -0.14, path_type='circle', return_t_only=True),
            define_camera_path(num_frames, -0.14, -0.14, -0.14, path_type='circle', return_t_only=True),
        ]
        crop = 32
        ##### the max value of the relative coordinates (above) should not exceed 0.3 (max used in training)
        
        ref_input = data['src_img1']
        for j, T in enumerate(Ts):
            print(video_paths[j])
            T = torch.from_numpy(T).float().to(renderer.device)
            time_steps = np.linspace(0, 1, num_frames)
            frames = []
            reshaded_frames = []

            for i, t_step in tqdm(enumerate(time_steps), total=len(time_steps),
                                  desc='generating video of {} camera trajectory'.format(video_paths[j])):
                
                ######### RESHADING INSERT#############
                disparity = 1 / data['src_depth1']
                # disparity is scaled by 4 during training (should not exceed 0.25 (max during training))
                disparity = (disparity / 4)
                disparity = torch.Tensor(embed_fn(disparity[0])[0]).permute(2, 0, 1)[None]

                rgbd = padder(torch.cat((ref_input, disparity), 1).cuda())
                reshaded = reshading(reshader, dir_model, rgbd, T[i])[:, :, :im_h, :im_w]
                reshaded_frames.append((255. * reshaded.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8))

                data['src_img1'] = reshaded

                renderer.process_data_single(data)

                pts1, rgb1, feat1, mask, side_ids = \
                    renderer.render_rgbda_layers_from_one_view(return_pts=True)
                #################################

                pred_img, _, meta = renderer.render_pcd_single(pts1, rgb1,
                                                        feat1, mask, side_ids,
                                                        t=T[i], R=None, time=0)
                frame = (255. * pred_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8)
                # mask out fuzzy image boundaries due to no outpainting
                img_boundary_mask = (meta['acc'] > 0.5).detach().cpu().squeeze().numpy().astype(np.uint8)
                img_boundary_mask_cleaned = process_boundary_mask(img_boundary_mask)
                frame = frame * img_boundary_mask_cleaned[..., None]
                frame = frame[crop:-crop, crop:-crop]
                frames.append(frame)
            
            video_out_file = os.path.join(video_out_folder, '{}_{}-{}.mp4'.format(
                video_paths[j], scene_id, frame_id1))
            imageio.mimwrite(video_out_file, frames, fps=25, quality=8)
            imageio.mimwrite(f"{video_out_folder}/out_{args.model_dir}_model{args.ckpt}.mp4", reshaded_frames, fps=25, quality=8)


        print('output videos have been saved in {}.'.format(video_out_folder))

if __name__ == '__main__':
    args = config.config_parser()
    
    render(args)