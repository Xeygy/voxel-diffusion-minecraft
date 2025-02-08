from PIL import Image
import os

def to_gif(paths, name='output'):
    im_list = []
    for path in paths:
        img = Image.open(path)
        im_list.append(img)
    im_list[0].save(f'{name}.gif', 
                    save_all=True, 
                    append_images=im_list[1:] + [im_list[-1]] * 10, 
                    loop=0, duration=500)

if __name__ == '__main__':
    dir_path = "qeval20250208_104617/"
    
    paths = [dir_path + f for f in os.listdir(dir_path) if f.endswith('.png')]
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    to_gif(paths, "monocube")