import torch


def save_torch_model(model, path):
    model.eval()
    if torch.cuda.is_available():
        model.cpu()
        torch.save(model.state_dict(), path)
        model.cuda()
    else:
        torch.save(model.state_dict(), path)
    model.train()
