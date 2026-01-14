import cv2

from .AbsMethod import AbsMethod


class HeImage(AbsMethod):
    def __init__(self, param=None, pipeline=None, clipLimit=None, gridSize=None, iteration=None):
        super().__init__(param, pipeline)
        self._clahe = None
        self.clipLimit = clipLimit
        self.gridSize = gridSize
        self.iteration = iteration

    def check_way(self, r_clahe=False):
        if self.param is not None:
            try:
                self.clipLimit = self.param.clipLimit
            except AttributeError:
                print("No 'clipLimit' was input, it was set to default=2.0")
            try:
                self.gridSize = self.param.gridSize
            except AttributeError:
                print("No 'gridSize' was input, it was set to default=8")
            if r_clahe:
                try:
                    self.iteration = self.param.iteration
                except AttributeError:
                    print("No 'iteration' was input, it was set to default=2")
        else:
            if self.clipLimit is None:
                self.clipLimit = 2.0
                print("No 'clipLimit' was input, it was set to default=2.0")
            if self.gridSize is None:
                self.gridSize = 8
                print("No 'gridSize' was input, it was set to default=8")
            if r_clahe:
                if self.iteration is None:
                    self.iteration = 2
                    print("No 'iteration' was input, it was set to default=2")

        self.create_clahe(self.clipLimit, self.gridSize)

    def create_clahe(self, clip_limit, grid_size):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                      tileGridSize=(grid_size, grid_size))

    def he(self):
        return cv2.equalizeHist(self.pipeline)

    def clahe(self):
        self.check_way()
        return self._clahe.apply(self.pipeline)

    def recursive_clahe(self):
        self.check_way(r_clahe=True)
        recursive_clahe_pipline = self._clahe.apply(self.pipeline)
        if self.iteration > 0:
            self.clipLimit += 1
            self.iteration -= 1
            self._clahe = cv2.createCLAHE(clipLimit=self.clipLimit,
                                          tileGridSize=(self.gridSize, self.gridSize))
            recursive_clahe_pipline = self.recursive_clahe()
        return recursive_clahe_pipline
