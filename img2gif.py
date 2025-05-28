from PIL import Image
import os

def img2gif(path):
    def extract_number(filename):
        return int(''.join(filter(str.isdigit, filename)))

    img_list = os.listdir(path)
    img_list = sorted(img_list, key=extract_number)
    img_list = [os.path.join(path, x) for x in img_list]
    images = [Image.open(x) for x in img_list]
    
    im = images[0]
    im.save('Lab1_Vanilla_GAN/result.gif', save_all=True, append_images=images[1:], loop=0xff, duration=200)

img2gif('Lab1_Vanilla_GAN/output')
