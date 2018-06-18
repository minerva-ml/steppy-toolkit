import torch


def persist_torch_model(model, path):
    model.eval()
    if torch.cuda.is_available():
        model.cpu()
        torch.save(model.state_dict(), path)
        model.cuda()
    else:
        torch.save(model.state_dict(), path)
    model.train()


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
