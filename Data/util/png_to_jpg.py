from PIL import Image
import os

def png2jpg ( path ):
    #change path to the image folder
    for img in os.listdir(path):
        if img.endswith(".png"):
            im = Image.open(path + "{}".format(img))
            rgb_im = im.convert("RGB")
            img_name = img.split(".")[0]
            rgb_im.save(path + "{}.jpg".format(img_name))

    for img in os.listdir(path):
        if img.endswith(".png"):
            os.remove(path+"{}".format(img))

if __name__ == "__main__":
    path = "/home/amc/yolo/yolo_all/"
    png2jpg(path)