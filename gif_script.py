from PIL import Image
import os
import argparse

def to_gif(paths, name='output'):
    im_list = []
    for path in paths:
        img = Image.open(path)
        im_list.append(img)
    im_list[0].save(f'{name}.gif', 
                    save_all=True, 
                    append_images=im_list[1:] + [im_list[-1]] * 10, 
                    loop=0, duration=500)

def to_gif_imlist(im_list, name='output', frame_duration=500, resize_width=None):
    if resize_width is not None:
        width, height = im_list[0].size
        new_width  = resize_width
        new_height = new_width * height / width 

        for i, im in enumerate(im_list):
            im.thumbnail((new_width, new_height), Image.LANCZOS)
            im_list[i] = im
    im_list[0].save(f'{name}.gif', 
                    save_all=True, 
                    append_images=im_list[1:] + [im_list[-1]] * 10, 
                    loop=0, duration=frame_duration)

def giffed_single(dir_path):
    paths = [dir_path + f for f in os.listdir(dir_path) if f.endswith('.png')]
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    to_gif(paths, "monocube")
    
def giffed_multi(dir_path, gif_name='multicube'):
    '''
    expects pngs of the form
    xxxx_n_s.png
    where n is the epoch number
    and s is the sample number
    '''
    paths = [dir_path + '/' + f for f in os.listdir(dir_path) if f.endswith('.png') and '_' in f]
    dict_paths = {}
    for path in paths:
        fname = path.split('/')[-1]
        key = int(fname.split('_')[-2].split('.')[0])
        if key not in dict_paths:
            dict_paths[key] = [path]
        else:
            dict_paths[key].append(path)

    ims = []
    keys = list(dict_paths.keys())
    keys.sort()
    for key in keys:
        im_list = dict_paths[key]
        im_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        n = len(im_list)
        sqrtn = int(n**0.5)

        # make a grid of n images
        im = Image.open(im_list[0])
        width, height = im.size
        grid = Image.new('RGB', (width * sqrtn, height * sqrtn))
        for i, im_path in enumerate(im_list):
            im = Image.open(im_path)
            grid.paste(im, (width * (i % sqrtn), height * (i // sqrtn)))
        ims.append(grid)
    to_gif_imlist(ims, gif_name, 200)
    
if __name__ == '__main__':
    # add arg for directory
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='directory containing pngs')
    parser.add_argument('name', type=str, help='name of gif', )
    args = parser.parse_args()
    giffed_multi(args.dir, args.name)
    
    