from LLIELib.troditionalAlgorithm.utils import TAProcessor

if __name__ == '__main__':
    path_to_img = r"assets/input.jpg"  # it can be input local image, url, or bytes stream

    TA = TAProcessor(
        algorithm='he',
        color_space='rgb',
        # following parameters are alternative
        showimg=True,
        saveimg=False,
        # name_Suffix='rgb_he',
        # width=800,
        # height=600,
        # format='jpg',
        # directory=results_path,
        # clipLimit=,   # default = 2.0
        # gridSize=,    # default = 8
        # iteration=    # default = 2
        # dcpSize=15,     # default = 15
        )

    img = TA.processor(path_to_img)
    """
        Quickly start:
        
        supported algorithm:
            he: he | clahe | rclahe | DCP
        
        supported color_space:
            cs: rgb | hls | hsv | lab | yuv 
            
    """
