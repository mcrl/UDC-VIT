import cv2
import numpy as np
import os

def yuv2png(yuv_filename, output_dir, fname, width, height, skip_frames, vflip, hflip):

    file_size = os.path.getsize(yuv_filename)
    print(yuv_filename, ': file_size :', file_size)
    n_frames = file_size // (width*height*3 // 2)
    print(yuv_filename, ': n_frames :', n_frames)
    f = open(yuv_filename, 'rb')
    for i in range(n_frames):
        if i < skip_frames:
            yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
            print('skip this frame', i)
        else:
            yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            if vflip:
                bgr = cv2.flip(bgr, 0)
            if hflip:
                bgr = cv2.flip(bgr, 1)
            output_filename = os.path.join(output_dir, f'{fname}_f{i-skip_frames:05d}.png')
            print(output_filename)
            cv2.imwrite(output_filename, bgr)
    f.close()

def run_yuv2png(input_dir, output_dir, skip_frames):
    
    width, height = 1920, 1080
    fname_0, fname_1 = 'cam_0', 'cam_1'
    output_dir_0 = os.path.join(output_dir, fname_0)
    output_dir_1 = os.path.join(output_dir, fname_1)
    if not os.path.exists(output_dir_0):
        os.makedirs(output_dir_0)
    if not os.path.exists(output_dir_1):
        os.makedirs(output_dir_1)

    yuv_filename_0 = os.path.join(input_dir, fname_0+'.yuv')
    yuv2png(yuv_filename_0, output_dir_0, fname_0, width, height, skip_frames, True, True)
    
    yuv_filename_1 = os.path.join(input_dir, fname_1+'.yuv')
    yuv2png(yuv_filename_1, output_dir_1, fname_1, width, height, skip_frames, True, False)


if __name__ == "__main__":
    num=str(1)
    run_yuv2png("yuv/"+num+"/", "png/"+num+"_all/", 0)
