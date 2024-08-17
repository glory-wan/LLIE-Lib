from LibLlie.utils import ConvertFormat
from LibLlie.troditionAlgorithm.utils import script_ta

if __name__ == '__main__':
    path_to_img = r"D:\BaiduSyncdisk\code\myweb\mypage\static\images\0.jpg"  # it can be input local image, url, or bytes stream

    img = script_ta(
        path_to_img,
        algorithm='he',
        color_space='rgb',

        # following parameters are alternative
        showimg=True,
        saveimg=False,
        # name='rgb_he',
        # width=800,
        # height=600,
        # format='jpg',
        # directory=results_path,
        # clipLimit=,   # default = 2.0
        # gridSize=,    # default = 8
        # iteration=    # default = 2
    )
    """
        Quickly start:
        
        supported algorithm:
            he: he | clahe | rclahe
        
        supported color_space:
            cs: rgb | hls | hsv | lab | yuv 
            
    """
