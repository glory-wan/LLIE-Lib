import numpy as np


def process_pipeline(method, cs, algorithms, color_space, process, pipeline, DCP):
    """
    Description:
        Pass in the selected algorithm and color space, as well as the image's pipeline and processor.
        This function can automatically complete the enhancement of the image
        according to the passed parameters, and return the enhanced image.

    :param method: the selected algorithm
    :param cs: the selected color space
    :param algorithms: a dictionary of algorithm names
    :param color_space: a dictionary of color space
    :param process: the processor of pipeline
    :param pipeline: the value of pipeline needed to be processed
    :param DCP: a class of DarkChannel, see details in LibLlie/troditionAlgorithm/methods/DarkChannel.py

    :return: enhanced image
    """
    if method in algorithms:
        selected_algorithm = algorithms[method]
        if cs in color_space:
            if cs != 'rgb':
                process.pipeline = np.array(color_space[cs]())
                process.pipeline = selected_algorithm()
                pipeline.pipeline = process.pipeline
                img = pipeline.merge_pipeline(cs=cs)
            else:
                process.pipeline_tuple = np.array(color_space[cs]())
                for i in range(3):
                    process.pipeline = process.pipeline_tuple[i]
                    process.pipeline_tuple[i] = selected_algorithm()
                pipeline.pipeline = process.pipeline_tuple
                img = pipeline.merge_pipeline(cs=cs)
        else:
            raise ValueError('Incorrect color space name was input !')
    elif method == 'DCP':
        img = DCP.run(pipeline.img)
    else:
        raise ValueError('Incorrect algorithm name was input !')

    return img
