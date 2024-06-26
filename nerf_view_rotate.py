import os
import numpy as np
import torch
import math
import cv2, os, numpy as np
from run_nerf import create_nerf, render, to8b
from load_blender import pose_spherical
from PIL import Image
class NeRFSynthesizer:


    def __init__(self, args):

        self.args = args
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
        self.render_kwargs_test = render_kwargs_test
        self.render_kwargs_test.update({
            'near': 2.0,
            'far': 6.0
        }) 

        self.H, self.W, self.focal = 400, 400, 549.99998201859785

        # 相机内参矩阵
        K = np.array([
            [self.focal,     0,     0.5 * self.W],
            [0,         self.focal, 0.5 * self.H],
            [0,              0,                1]
        ])
        self.hwf = [400, 400, K]
        self.ckpt_dir = './logs/blender_paper_lego'
        self.iter = 200000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'

        self._load_model()
    

    def _load_model(self):
        ckpt_path = os.path.join(self.ckpt_dir, f'{self.iter:06d}.tar')
        with open(ckpt_path, 'rb') as f:
            ckpt = torch.load(f, map_location=torch.device(self.device))

        # 粗网络权重
        self.render_kwargs_test['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
        # 精网络权重
        self.render_kwargs_test['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])


    def get_image_by_spherical(self, theta, phi, radius): #由视角得到变换矩阵再得到图像
        """
        theta: -180 -- +180
        phi: 固定值 -30
        radius: 固定值 4
        """
        torch.set_grad_enabled(False)
        render_pose = pose_spherical(theta, phi, radius)  #生成相机坐标系到世界坐标系的变换矩阵
        rgb = self.get_image_by_pose(render_pose)  #生成图像
        torch.set_grad_enabled(True)
        return rgb


    def get_image_by_pose(self, render_pose):  #由变换矩阵得到图像
        """
        render_pose是camera to world坐标转换矩阵（Transformation Matrix）
        """
        # 生成图片
        torch.set_grad_enabled(False)
        H, W, K = self.hwf
        render_pose = render_pose.to(self.device)
        rgb, disp, acc, extras = render(H, W, K, chunk=512, c2w=render_pose[:3, :4], **self.render_kwargs_test)
        torch.set_grad_enabled(True)
        rgb = rgb.cpu().numpy()
        rgb = to8b(rgb)
        return rgb


    def pose_spherical(self, theta, phi, radius): #由视角得到变换矩阵
        """
        theta: -180 -- +180
        phi: 固定值 -30
        radius: 固定值 4
        """
        # 计算相机坐标系到世界坐标系的变换矩阵
        c2w = pose_spherical(theta, phi, radius)
        return c2w


    def spherical_pose(self, render_pose):  #由变换矩阵得到视角
        """
        render_pose是camera to world坐标转换矩阵（Transformation Matrix）
        """
        render_pose = render_pose.to(self.device)
        inv_mat = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])).inverse()
        render_pose = inv_mat @ render_pose
        R = render_pose[:3, :3]
        # 计算x轴旋转参数phi
        phi = -math.atan2(-R[2, 1], R[2, 2]) / math.pi * 180
        # 计算y轴旋转参数theta
        theta = math.atan2(R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)) / math.pi * 180
        # 计算Radius
        inv_c2w = torch.inverse(render_pose)
        radius = inv_c2w[:3, 3].norm().item()
        return theta, phi, radius



def visualize_and_save(rgb, fname):
    import matplotlib.pyplot as plt
    plt.imshow(rgb)
    plt.imsave(fname, rgb)
    plt.show()


if __name__ == '__main__':
  
    from opts import config_parser
    parser = config_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    synthesizer = NeRFSynthesizer(args)

    # # 极坐标系转换为相机坐标系
    render_pose = synthesizer.pose_spherical(45,-45, 4)   #视角到变换矩阵
    print(render_pose)

    rgb1 = synthesizer.get_image_by_pose(render_pose)  # 变换矩阵到图像
    visualize_and_save(rgb1, "origin11.jpg")
    # # 检验相机坐标系转换为极坐标系
    theta, phi, radius = synthesizer.spherical_pose(render_pose)  #变换矩阵到视角
    #render_pose = synthesizer.pose_spherical(theta +30, phi, radius)


    rgb2 = synthesizer.get_image_by_spherical(theta+50, phi-50, radius) #视角到图像

    visualize_and_save(rgb2, "rotate22.jpg")
    print(111)
    #render_pose = synthesizer.pose_spherical(theta, phi, radius)
    #print(render_pose)
    #visualize_and_save(render_pose, "rotate.jpg")
    # 从极坐标系生成图片
    #rotate = synthesizer.get_image_by_spherical(theta, phi,4)    #视角到图像
    #visualize_and_save(rotate, "rotate.jpg")


    #
    # # 从相机坐标系生成图片
    # render_pose = torch.tensor([
    #     [
    #         0.5655173063278198,
    #         -0.2501452565193176,
    #         0.7858863472938538,
    #         3.1680095195770264
    #     ],
    #     [
    #         0.8247364163398743,
    #         0.17152325809001923,
    #         -0.5388780236244202,
    #         -2.1722869873046875
    #     ],
    #     [
    #         0.0,
    #         0.9528940916061401,
    #         0.30330324172973633,
    #         1.2226545810699463
    #     ],
    #     [
    #         0.0,
    #         0.0,
    #         0.0,
    #         1.0
    #     ]
    #         ])
    # rgb2 = synthesizer.get_image_by_pose(render_pose)  #变换矩阵到图像
    #
    # # render_pose -> polar coords   相机坐标系转换为极坐标系
    # theta, phi, radius = synthesizer.spherical_pose(render_pose)  #变换矩阵到视角
    # print(theta,phi,radius)
    # print(123)
    # render_pose = synthesizer.pose_spherical(theta + 5, phi, radius)
    # print(render_pose)
    # visualize_and_save(origin, "rotate.jpg")
    #
    # visualize_and_save(rgb2, "origin1.jpg")
    # # rbg3 = synthesizer.get_image_by_spherical(theta + 5, phi, radius)
    # # rbg3 = synthesizer.get_image_by_spherical(theta - 5, phi, radius)
    # # rbg3 = synthesizer.get_image_by_spherical(theta, phi + 5, radius)
    # rgb3 = synthesizer.get_image_by_spherical(theta+1 , phi, radius)
    # print(theta+10, phi, radius)
    # visualize_and_save(rgb3, "rotated2.jpg")



    # # matplotlib visualize


