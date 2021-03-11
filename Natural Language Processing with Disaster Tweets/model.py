import torch
import torch.nn as nn

class network(nn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self, input_size=50, hidden_size=50, num_layers=2):
        super(network, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.4, bidirectional = False, batch_first = True)
        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(True),
            nn.Linear(2 * hidden_size, 1),
        )

    def forward(self, input):  # LSTM,GRU模型,batch normal---Bidirectionel语序正反各来一次
        # LSTM-全连接-预测，基本操作，应该一层或者两层
        # dropout 0,4-0,6
        # input:32,xx,dim
        #input = input.permute(1, 0, 2)
        #print(input.shape)
        lstm_out, (h_n, c_n) = self.lstm(input)
        # print(lstm_out.shape)
        # lstm_out = lstm_out[:,-1,:]
        # print(h_n.shape)
        lstm_get = torch.cat((h_n[0], h_n[1]), 1)
        # print(lstm_get.shape)

        linear_in = lstm_get.view(lstm_get.shape[0], -1)
        output = self.classifier(linear_in)
        # 去除无用的维度
        output = output.squeeze(-1)
        output = torch.sigmoid(output)

        return output


def get_model(dim, from_old_model, model_path):
    net = network(input_size=dim,hidden_size=dim)
    if from_old_model:
        net.load_state_dict(torch.load(model_path))
    return net