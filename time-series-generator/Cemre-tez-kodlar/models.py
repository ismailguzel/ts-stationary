
import torch
import torch.nn as nn

#CNN Model
class ConvModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ConvModel, self).__init__()

        #input layer
        layers = []
        layers.append(nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=5, padding=2))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_size))

        #hidden layers
        for n in range(num_layers-1):
            layers.append(nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, padding=2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))

        #out FC layer
        layers.append(nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=1, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv1d(in_channels=hidden_size*2, out_channels=4, kernel_size=1, padding=0))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x.unsqueeze(1))
        return x


#BiLSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
           nn.Conv1d(in_channels=hidden_size*2, out_channels=512, kernel_size=1),
           nn.ReLU(),
           nn.Conv1d(in_channels=512, out_channels=4, kernel_size=1),
       )

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # *2 for bidirectional
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x.unsqueeze(-1), (h_0, c_0))
        out = out.permute(0, 2, 1)
        out = self.linear(out)  # Get the output of the last time step
        return out

#CNN-BiLSTM Parallel Model
class ConvLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(ConvLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.device = device
        # Convolutional Layers
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, padding=2),
        )

        # LSTM Layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # Final fully connected layers
        self.linear = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size*3, out_channels=hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_size, out_channels=4, kernel_size=1),
        )

    def forward(self, x):

        # Apply LSTM path
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # *2 for bidirectional
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x.unsqueeze(-1), (h_0, c_0))
        lstm_features = out.permute(0, 2, 1)
        

        # Apply convolutional path
        conv_features = self.conv(x.unsqueeze(1))
        

        # Concatenate the features from both models
        concatenated_features = torch.cat((conv_features, lstm_features), dim=1)
        # print(f'lstm_features: {lstm_features.shape}')
        # print(f'conv_features: {conv_features.shape}')
        # print(f'concat_features: {concatenated_features.shape}')
        
        # Apply the final fully connected layers
        output = self.linear(concatenated_features)

        return output