import pandas as pd

train_path = "train.csv"
pred_path = "models_pred.csv"
output_path = "train_clean.csv"

pred_csv = pd.read_csv(pred_path)
train_csv = pd.read_csv(train_path)

# get the pred acc for this line
def get_acc(pred_csv, index):
    label = pred_csv.loc[index, 'label']
    acc = 0
    if pred_csv.loc[index, 'pred_0'] == label:
        acc += 0.2
    if pred_csv.loc[index, 'pred_1'] == label:
        acc += 0.2
    if pred_csv.loc[index, 'pred_2'] == label:
        acc += 0.2
    if pred_csv.loc[index, 'pred_3'] == label:
        acc += 0.2
    if pred_csv.loc[index, 'pred_4'] == label:
        acc += 0.2
    return round(acc,1)

pred_acc_record = {0:0, 0.2:0, 0.4:0, 0.6:0, 0.8:0, 1:0}
delete_index = []
for index in range(len(pred_csv)):
    acc = get_acc(pred_csv, index)
    pred_acc_record[acc] += 1
    # remove noise data
    if acc <= 0.2:
        delete_index.append(index)

train_csv = train_csv.drop(delete_index)
train_csv = train_csv.reset_index(drop=True)
print(pred_acc_record)
print((pred_acc_record[0] + pred_acc_record[0.2])/len(pred_csv))

print(len(pred_csv),len(train_csv))

train_csv.to_csv(output_path, index=False, sep=',')

