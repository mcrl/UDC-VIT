# The controller using Raspberry Pi5

This program is programmed for Raspberry Pi5 that has two four-lane MIPI interface connections for high bandwidth to utilize two cameras. The program synchronizes the cameras by utilizing MPI barriers which ensure less than 0.5 fps (8 msec) difference between paired frames from two cameras. The codes are mainly from https://github.com/raspberrypi/rpicam-apps.



## Environment

+ Raspberry Pi5
+ libcamera version: v0.2.0+120-eb00c13d
+ OS version: Debian 12



## How to Build

Please refer to the official raspberrypi documentation (https://www.raspberrypi.com/documentation/computers/camera_software.html#building-rpicam-apps-without-building-libcamera) to build rpicam apps without re-building libcamera.




## Run following command to execute the program

```bash
sh run_vid.sh
```



## Result

The result will be saved under ```rpicam_udc-vix/result```. Once you execute the program, the program creates two videos in yuv format and two logs that contain information such as fps, exposure, focus measure and focus state of each frames of the videos.




## License and Acknowledgement

This work is licensed under Raspberry Pi Ltd License. The codes are mainly from following repositories.
For more information, refer to [original work](https://github.com/raspberrypi/documentation/blob/develop/documentation/asciidoc/computers/camera/rpicam_apps_intro.adoc).
