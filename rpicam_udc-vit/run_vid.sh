# sudo meson install -C build

mpiexec -np 2  ./rpicam-apps/build/apps/rpicam-vid  --output cam --output_format .yuv  --width 1920 --height 1080 --framerate 60 --frames 210 --codec yuv420 --info-text "#%frame (%fps fps) exp %exp / Focus measure: %focus / focus state: %afstate" 
