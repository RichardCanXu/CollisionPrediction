import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):

        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden = nn.Linear(6,9)
        self.nonlinear_activation = nn.Sigmoid()
        self.hidden_to_hidden = nn.Linear(9,3)
        self.hidden_to_output = nn.Linear(3,1)

    def forward(self, input):

        hidden = self.input_to_hidden(input)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.hidden_to_hidden(hidden)
        hidden = self.nonlinear_activation(hidden)
        output = self.hidden_to_output(hidden)
        return output


    def evaluate(self, model, test_loader, loss_function):

        loss = 0
        count = 0
        for idx, sample in enumerate(test_loader):
            input_, target = sample['input'], sample['label']
            loss = loss + loss_function(model.forward(torch.Tensor(input_)), torch.Tensor([target]))
            count += 1
        return float(loss / count)

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
