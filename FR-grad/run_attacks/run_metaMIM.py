import sys
sys.path.extend(['E:\\PythonCode\\FR_GRAD'])
import numpy as np
import torch
import os
import argparse
# from meta_attacks.MetaMIM import MetaMIM
from SM_attacks.smmim import MetaMIM
from run_attacks.utils import save_images
from networks.get_model import getmodel
from dataset import LOADER_DICT
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--device', help='device id', type=str, default='cuda')
parser.add_argument('--dataset', help='dataset', type=str, default='lfw', choices=['lfw', 'ytf', 'cfp'])
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
parser.add_argument('--goal', help='dodging or impersonate', type=str, default='dodging', choices=['dodging', 'impersonate'])
parser.add_argument('--eps', help='epsilon', type=float, default=6)
parser.add_argument('--mu', help='mu', type=float, default=0.9)
parser.add_argument('--iters', help='iters', type=int, default=20)
parser.add_argument('--distance', help='l2 or linf', type=str, default='linf', choices=['linf', 'l2'])
parser.add_argument('--seed', help='random seed', type=int, default=1234)
parser.add_argument('--batch_size', help='batch_size', type=int, default=20)
parser.add_argument('--output', help='output dir', type=str, default='../output/mim_lfw_d_n_n')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
def main():
    models = ['MobilenetV2-stride1', 'ShuffleNet_V1_GDConv',
              'ArcFace', 'ShuffleNet_V2_GDConv-stride1',
              'ResNet50', 'IR50-CosFace', 'IR50-Am',
              'IR50-SphereFace', 'Mobilenet-stride1']
    # models = ['MobilenetV2-stride1', 'ShuffleNet_V1_GDConv',
    #           'ArcFace']
    model, img_shape = getmodel(args.model, device=args.device)
    attack = MetaMIM(model=None,
                 goal=args.goal,
                 eps=args.eps,
                 iters=args.iters,
                 distance_metric=args.distance,
                 mu=args.mu)

    datadir = os.path.join('../data/lfw', '{}-{}x{}'.format(args.dataset, img_shape[0], img_shape[1]))
    loader = LOADER_DICT[args.dataset](datadir, args.goal, args.batch_size, model)

    os.makedirs(args.output, exist_ok=True)
    cnt = 0
    outputs = []
    model_s = []
    for index, element in enumerate(models):
        # 根据索引得到模型
        model, _ = getmodel(models[index], device='cuda')
        model_s.append(model)
    for xs, ys, ys_feat, pairs in tqdm(loader, total=len(loader)):
        x_adv = attack.attack(src=xs, dict=ys,models=model_s)
        for i in range(len(pairs)):
            img = x_adv[i].detach().cpu().numpy().transpose((1, 2, 0))
            cnt += 1
            npy_path = os.path.join(args.output, str(cnt) + '.npy')

            outputs.append([npy_path, pairs[i][0], pairs[i][1]])
            np.save(npy_path, img)
            original_image = xs[i].cpu().numpy().transpose((1, 2, 0))
            save_images(img, original_image, str(cnt) + '.png', args.output)
    with open(os.path.join(args.output, 'annotation.txt'), 'w') as f:
        for pair in outputs:
            f.write('{} {} {}\n'.format(pair[0], pair[1], pair[2]))

if __name__ == '__main__':
    main()
