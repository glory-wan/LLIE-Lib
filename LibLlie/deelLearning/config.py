from LibLlie.deelLearning.model.SCI import *
from LibLlie.deelLearning.model.Zero_DCE import *

models = {
    'SCI-easy': Finetunemodel,
    'SCI-medium': Finetunemodel,
    'SCI-difficult': Finetunemodel,
    'Zero-DCE': enhance_net_nopool,
}

