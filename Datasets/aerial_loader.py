"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
import re
import numpy as np
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Utils.utils import write_file, depth_read


class Random_Sampler():
    "Class to downsample input lidar points"

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, depth):
        mask_keep = depth > 0
        n_keep = np.count_nonzero(mask_keep)

        if n_keep == 0:
            return mask_keep
        else:
            depth_sampled = np.zeros(depth.shape)
            prob = float(self.num_samples) / n_keep
            mask_keep =  np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)
            depth_sampled[mask_keep] = depth[mask_keep]
            return depth_sampled


class AerialPreprocessing(object):
    def __init__(self, dataset_path, input_type='depth', side_selection=''):
        self.train_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.val_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.test_files = {'img': [], 'lidar_in': []}
        self.dataset_path = dataset_path
        self.depth_keyword = 'proj_depth'
        self.rgb_keyword = 'kitti_raw'
        # self.use_rgb = input_type == 'rgb'
        self.use_rgb = True

    def get_paths(self):
        # train and validation dirs
        train_img_paths, train_gt_paths, train_sdepth_paths = self.get_train_test_paths(self.dataset_path, 'train')
        val_img_paths, val_gt_paths, val_sdepth_paths = self.get_train_test_paths(self.dataset_path, 'val')
        self.train_paths['lidar_in'] = train_sdepth_paths
        self.train_paths['gt'] = train_gt_paths
        self.val_paths['lidar_in'] = val_sdepth_paths
        self.val_paths['gt'] = val_gt_paths
        if self.use_rgb:
            self.train_paths['img'] = train_img_paths
            self.val_paths['img'] = val_img_paths

    def get_train_test_paths(self, root_dir, mode):
        data_dir = os.path.join(root_dir, mode)
        all_videos = [f for f in os.listdir(data_dir)]
        img_paths, gt_paths, sdepth_paths, masks = list(), list(), list(), list()
        for name in all_videos:
            path_video = os.path.join(data_dir, name)
            files = [f for f in os.listdir(path_video) if 'image' in f]
            indices = [int(f.split('.')[0][5:]) for f in files]
            indices = sorted(indices)
            for id_in_data in indices:
                img_paths.append(os.path.join(path_video, 'image%d.png' % id_in_data))
                sdepth_paths.append(os.path.join(path_video, 'sparse%d.png' % id_in_data))
                if mode != 'test':
                    gt_paths.append(os.path.join(path_video, 'depth%d.png' % id_in_data))

        return img_paths, gt_paths, sdepth_paths


    def downsample(self, lidar_data, destination, num_samples=500):
        # Define sampler
        sampler = Random_Sampler(num_samples)

        for i, lidar_set_path in tqdm.tqdm(enumerate(lidar_data)):
            # Read in lidar data
            name = os.path.splitext(os.path.basename(lidar_set_path))[0]
            sparse_depth = Image.open(lidar_set_path)


            # Convert to numpy array
            sparse_depth = np.array(sparse_depth, dtype=int)
            assert(np.max(sparse_depth) > 255)

            # Downsample per collumn
            sparse_depth = sampler.sample(sparse_depth)

            # Convert to img
            sparse_depth_img = Image.fromarray(sparse_depth.astype(np.uint32))

            # Save
            folder = os.path.join(*str.split(lidar_set_path, os.path.sep)[7:12])
            os.makedirs(os.path.join(destination, os.path.join(folder)), exist_ok=True)
            sparse_depth_img.save(os.path.join(destination, os.path.join(folder, name)) + '.png')

    def convert_png_to_rgb(self, rgb_images, destination):
        for i, img_set_path in tqdm.tqdm(enumerate(rgb_images)):
            name = os.path.splitext(os.path.basename(img_set_path))[0]
            im = Image.open(img_set_path)
            rgb_im = im.convert('RGB')
            folder = os.path.join(*str.split(img_set_path, os.path.sep)[8:12])
            os.makedirs(os.path.join(destination, os.path.join(folder)), exist_ok=True)
            rgb_im.save(os.path.join(destination, os.path.join(folder, name)) + '.jpg')
            # rgb_im.save(os.path.join(destination, name) + '.jpg')

    def get_selected_paths(self, selection):
        files = []
        for file in sorted(os.listdir(os.path.join(self.dataset_path, selection))):
            files.append(os.path.join(self.dataset_path, os.path.join(selection, file)))
        return files

    def prepare_dataset(self, has_test_set=False):
        self.get_paths()
        if has_test_set:
            if self.use_rgb:
                self.test_files['img'], _, self.test_files['lidar_in'] = self.get_train_test_paths(self.dataset_path, 'test')
            else:
                _, _, self.test_files['lidar_in'] = self.get_train_test_paths(self.dataset_path, 'test')
        print("#training sparse depth inputs: ", len(self.train_paths['lidar_in']))
        print("#training image inputs: ", len(self.train_paths['img']))
        print("#training ground truths: ", len(self.train_paths['gt']))
        print("#validation sparse depth inputs: ", len(self.val_paths['lidar_in']))
        print("#validation image inputs: ", len(self.val_paths['img']))
        print("#validation ground truths: ", len(self.val_paths['gt']))
        if has_test_set and self.use_rgb:
            print(len(self.test_files['lidar_in']))
            print(len(self.test_files['img']))

    def compute_mean_std(self):
        nums = np.array([])
        means = np.array([])
        stds = np.array([])
        max_lst = np.array([])
        for i, raw_img_path in tqdm.tqdm(enumerate(self.train_paths['lidar_in'])):
            raw_img = Image.open(raw_img_path)
            raw_np = depth_read(raw_img)
            vec = raw_np[raw_np >= 0]
            # vec = vec/84.0
            means = np.append(means, np.mean(vec))
            stds = np.append(stds, np.std(vec))
            nums = np.append(nums, len(vec))
            max_lst = np.append(max_lst, np.max(vec))
        mean = np.dot(nums, means)/np.sum(nums)
        std = np.sqrt((np.dot(nums, stds**2) + np.dot(nums, (means-mean)**2))/np.sum(nums))
        return mean, std, max_lst


if __name__ == '__main__':

    # Imports
    import tqdm
    from PIL import Image
    import os
    import argparse
    from Utils.utils import str2bool

    # arguments
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument("--png2img", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--calc_params", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--num_samples', default=0, type=int, help='number of samples')
    parser.add_argument('--datapath', default='/usr/data/tmp/Depth_Completion/data')
    parser.add_argument('--dest', default='/usr/data/tmp/')
    args = parser.parse_args()

    dataset = AerialPreprocessing(args.datapath, input_type='rgb')
    dataset.prepare_dataset()
    if args.png2img:
        os.makedirs(os.path.join(args.dest, 'Rgb'), exist_ok=True)
        destination_train = os.path.join(args.dest, 'Rgb/train')
        destination_valid = os.path.join(args.dest, 'Rgb/val')
        dataset.convert_png_to_rgb(dataset.train_paths['img'], destination_train)
        dataset.convert_png_to_rgb(dataset.val_paths['img'], destination_valid)
    if args.calc_params:
        import matplotlib.pyplot as plt
        params = dataset.compute_mean_std()
        mu_std = params[0:2]
        max_lst = params[-1]
        print('Means and std equals {} and {}'.format(*mu_std))
        plt.hist(max_lst, bins='auto')
        plt.title('Histogram for max depth')
        plt.show()
        # mean, std = 14.969576188369581, 11.149000139428104
        # Normalized
        # mean, std = 0.17820924033773314, 0.1327261921360489
    if args.num_samples != 0:
        print("Making downsampled dataset")
        os.makedirs(os.path.join(args.dest), exist_ok=True)
        destination_train = os.path.join(args.dest, 'train')
        destination_valid = os.path.join(args.dest, 'val')
        dataset.downsample(dataset.train_paths['lidar_in'], destination_train, args.num_samples)
        dataset.downsample(dataset.val_paths['lidar_in'], destination_valid, args.num_samples)
