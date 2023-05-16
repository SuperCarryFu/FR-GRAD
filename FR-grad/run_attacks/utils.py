import os
from skimage.io import imread, imsave
import numpy as np
import torch
from tqdm import tqdm
from networks.config import threshold_lfw, threshold_ytf, threshold_cfp

threshold_dict = {
    'lfw': threshold_lfw,
    'ytf': threshold_ytf,
    'cfp': threshold_cfp
}
def save_images(image, original_image, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image = np.clip(image, 0, 255).astype(np.uint8)
    imsave(os.path.join(output_dir, filename), image.astype(np.uint8))

def cosdistance(x, y, offset=1e-5):
    x = x / torch.sqrt(torch.sum(x**2)) + offset
    y = y / torch.sqrt(torch.sum(y**2)) + offset
    return torch.sum(x * y)

def L2distance(x, y):
    return torch.sqrt(torch.sum((x - y)**2))

def run(loader, Attacker, args):
    os.makedirs(args.output, exist_ok=True)
    cnt = 0
    outputs = []
    for xs, ys, ys_feat, pairs in tqdm(loader, total=len(loader)):
        x_adv = Attacker.attack(images=xs, img_ft=ys_feat)
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

