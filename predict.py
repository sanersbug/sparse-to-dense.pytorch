import os
import time
import numpy as np
import h5py
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True
import dataloaders.transforms as transforms
import matplotlib.pyplot as plt
cmap = plt.cm.viridis
from PIL import Image
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo

iheight, iwidth = 480, 640 # raw image size
output_size = (228, 304)
def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def test_transform(rgb, depth):
    depth_np = depth
    transform = transforms.Compose([
        transforms.Resize(240.0 / iheight),
        transforms.CenterCrop(output_size),
    ])
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    depth_np = transform(depth_np)

    return rgb_np, depth_np


def create_sparse_depth(rgb, depth):
    sparsifier = UniformSampling(num_samples=0.0, max_depth=1.0)
    mask_keep = sparsifier.dense_to_sparse(rgb, depth)
    sparse_depth = np.zeros(depth.shape)
    sparse_depth[mask_keep] = depth[mask_keep]
    return sparse_depth

if __name__ == "__main__":
    model_path = 'model_best.pth'
    h5_path = './data/nyudepthv2/val/official/00002.h5'
    save_pic = 'official_00002.png'
    to_tensor = transforms.ToTensor()
    checkpoint = torch.load(model_path)
    args = checkpoint['args']
    # print(args)
    start_epoch = checkpoint['epoch'] + 1
    best_result = checkpoint['best_result']
    model = checkpoint['model']
    print("=> loaded best model (epoch {})".format(checkpoint['epoch']))

    #从h5导入数据
    rgb0, depth0 = h5_loader(h5_path)

    #转稀疏，如果本来提供的就是稀疏矩深度，可以注释到，把下面的打开
    depth0 = create_sparse_depth(rgb0, depth0)
    rgb, sparse_depth = test_transform(rgb0, depth0)

    # rgb, sparse_depth = np.asfarray(rgb0, dtype='float') / 255, depth0

    rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
    rgbd_in = to_tensor(rgbd)
    rgbd_in = rgbd_in.unsqueeze(0)

    input = rgbd_in.cuda()
    torch.cuda.synchronize()
    # compute output
    end = time.time()
    with torch.no_grad():
        pred = model(input)
    torch.cuda.synchronize()
    gpu_time = time.time() - end

    depth_pred_cpu = np.squeeze(pred.data.cpu().numpy())

    #拉伸转为彩色查看
    d_min = np.min(depth_pred_cpu)
    d_max = np.max(depth_pred_cpu)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img = Image.fromarray(depth_pred_col.astype('uint8'))
    img.save(save_pic)

    #存出来原图看看
    rgb_test = Image.fromarray(rgb0.astype('uint8'))
    rgb_test.save('2.png')