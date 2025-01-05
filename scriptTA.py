from LibLlie.troditionAlgorithm.utils import script_ta

if __name__ == '__main__':
    path_to_img = r"assets/input.jpg"  # it can be input local image, url, or bytes stream

    img = script_ta(
        path_to_img,
        algorithm='clahe',
        color_space='rgb',
        directory=r"C:\Users\13011\Desktop\save",
        # following parameters are alternative
        showimg=True,
        saveimg=True,
        # name='rgb_he',
        # width=800,
        # height=600,
        # format='jpg',
        # directory=results_path,
        # clipLimit=,   # default = 2.0
        # gridSize=,    # default = 8
        # iteration=    # default = 2
        # dcpSize=15,     # default = 15
    )
    """
        Quickly start:
        
        supported algorithm:
            he: he | clahe | rclahe | DCP
        
        supported color_space:
            cs: rgb | hls | hsv | lab | yuv 
            
    """
