import pandas as pd
import numpy as np
import torch
import pickle
from torchmetrics.classification import MulticlassConfusionMatrix


predict_1 = pd.read_csv('./output/0_2000_prediction.csv', header=0, index_col='Unnamed: 0').to_numpy()
predict_2 = pd.read_csv('./output/2000_4000_prediction.csv', header=0, index_col='Unnamed: 0').to_numpy()

predict = np.concatenate((predict_1, predict_2), axis=1)

preds = torch.from_numpy(predict[0])
target = torch.from_numpy(predict[1])
metric = MulticlassConfusionMatrix(num_classes=5)
print(metric(preds, target))

with open('./output/encoder_600.pkl', 'rb') as fp:
    encoder = pickle.load(fp)
print(encoder.inverse_transform([0, 1, 2, 3, 4]))
