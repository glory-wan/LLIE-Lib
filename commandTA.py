from LibLlie.troditionAlgorithm.utils import command_ta

if __name__ == '__main__':
    command_ta()

    """
        Quickly start:
        
        necessary parameters:
        --img: path to img
        --method: name of selected algorithm 
        --cs: name of selected color space
        
        example:
        ( if your root directory is `LibLlie`, run following command )
        python example/commandTA.py --img path/to/img --method he --cs hsv --name he_hsv --display True
        
        
        Refer to LibLlie.troditionAlgorithm.config for details of other parameters 
        
    """
