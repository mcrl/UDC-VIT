import rawpy
import imageio
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *
import numpy as np
import os
import sys
from tqdm import tqdm


PNG = './png'
DNG = './dng'

H = 1080
W = 1920
BPP = 16

def raw2dng(pathImg, width, height, bpp):
    data = np.fromfile(pathImg, dtype=np.uint16)
    data = data.reshape(1920,1080)

    t = DNGTags()
    t.set(Tag.ImageWidth, width)
    t.set(Tag.ImageLength, height)
    t.set(Tag.CFAPattern, CFAPattern.RGGB)
    t.set(Tag.BitsPerSample, bpp)
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.BlackLevel, (4096 >> (16 - bpp)))
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
    t.set(Tag.AsShotNeutral, [[1,1]])

    # convert format
    r = RAW2DNG()
    r.options(t, path="", compress=False)
    file_name = os.path.join(DNG,os.path.splitext(os.path.basename(pathImg))[0])
    data = r.convert(data, filename=file_name)
    
    return data
  
def dng2png(path):
  with rawpy.imread(path) as raw:
    rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,use_camera_wb=True,use_auto_wb=True)
  file_name = os.path.join(PNG,os.path.splitext(os.path.basename(path))[0]+'.png')
  imageio.imsave(file_name, rgb)


if __name__ == '__main__':
  rawImages = os.listdir(sys.argv[1])
  print("[info] Directory for raw images : {}".format(sys.argv[1]))
  print("[info] Directory for DNG images : {}".format(DNG))
  print("[info] Directory for PNG images : {}".format(PNG))
  print("[info] Total Images : {}".format(len(rawImages)))
  
  print("processing...")
  for img in tqdm(rawImages):
    dngName = raw2dng(os.path.join(sys.argv[1],img),W,H,BPP)
    dng2png(dngName)

