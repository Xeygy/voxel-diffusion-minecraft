from PIL import Image
import os
import argparse

# for grid generation from a directory of pngs

def grid_multi(dir_path):
    '''
    expects pngs of the form xxx_n.png, where
    - xxx is the generator name, and 
    - n is the sample number
    saves images xxx.png for each generator name
    '''
    paths = [dir_path + '/' + f for f in os.listdir(dir_path) if f.endswith('.png') and '_' in f]
    dict_paths = {}
    for path in paths:
        fname = path.split('/')[-1]
        key = fname.split('_')[0].split('.')[0]
        if key not in dict_paths:
            dict_paths[key] = [path]
        else:
            dict_paths[key].append(path)

    ims = []
    keys = list(dict_paths.keys())
    keys.sort()
    for key in keys:
        im_list = dict_paths[key]
        n = len(im_list)
        sqrtn = int(n**0.5)

        # make a grid of n images
        im = Image.open(im_list[0])
        width, height = im.size
        grid = Image.new('RGB', (width * sqrtn, height * sqrtn))
        for i, im_path in enumerate(im_list):
            im = Image.open(im_path)
            grid.paste(im, (width * (i % sqrtn), height * (i // sqrtn)))
        
        grid.save(f'{dir_path}/{key}.png')

if __name__ == '__main__':
    # add arg for directory
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='directory containing pngs')
    args = parser.parse_args()
    grid_multi(args.dir)
    
    