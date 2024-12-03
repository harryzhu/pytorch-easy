# coding=utf-8
import os
import shutil

img_dir = "data/img_dst_canny"
img4video_dir = "data/img4video"



def main():
    shutil.rmtree(img4video_dir)
    os.mkdir(img4video_dir)
    for root,dirs, files in os.walk(img_dir):
        for img in files:
            if img[-6:] != "_0.png":
                continue
            img_path = "/".join([root, img])
            ren_path = "/".join([img4video_dir, img.replace("_0.png",".png")])
            print(img_path)
            shutil.copy(img_path,ren_path)

if __name__ == "__main__":
    main()
