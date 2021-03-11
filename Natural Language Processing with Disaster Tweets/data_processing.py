import torch
import torch.nn as nn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    #print(len(batch),len(batch[0]), len(vocab))
    """
    Called after numericalisation but before vectorisation.
    """

    return batch

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

# def convertLabel(datasetLabel):# 演示用的regression
#     """
#     Labels (product ratings) from the dataset are provided to you as
#     floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
#     You may wish to train with these as they are, or you you may wish
#     to convert them to another representation in this function.
#     Consider regression vs classification.
#     """
#     #print("datasetLabel",datasetLabel)
#     #datasetLabel = torch.clamp(torch.round(datasetLabel),1,5)
#     return datasetLabel

# def convertNetOutput(netOutput):# 与上面对应，转换模型的输出的格式，收敛到1-5之间而且必须是整数，round()，clamp 限定范围
#     """
#     Your model will be assessed on the predictions it makes, which
#     must be in the same format as the dataset labels.  The predictions
#     must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
#     If your network outputs a different representation or any float
#     values other than the five mentioned, convert the output here.
#     """
#     #print("netOutput",netOutput)
#     #netOutput = torch.clamp(torch.round(netOutput),1,5)
#     #netOutput = torch.round(netOutput)
#     return netOutput