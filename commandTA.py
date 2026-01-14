from LLIELib.troditionalAlgorithm.utils import TAProcessor

if __name__ == '__main__':
    TA = TAProcessor()

    img = TA.command()

    """
        Quickly start:
        
        necessary parameters:
        --img: path to img
        --algorithm: name of selected algorithm 
        --cs: name of selected color space
        
        example:
        ( if your root directory is `LLIE-Lib`, run following command )
        python commandTA.py --img path/to/img --method he --cs hsv --name he_hsv --showing True
        
        
        Refer to LibLlie.troditionAlgorithm.config for details of other parameters 
        
    """
