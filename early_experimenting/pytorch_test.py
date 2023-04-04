import torch
#
print(f"{torch.__version__=}")
print(f"{torch.__file__=}")
print(f"{torch.cuda.device_count()=}")
print(f"{torch.cuda.is_available()=}")
print(f"{torch.version.cuda=}")
#
device = torch.device("cuda")
data = torch.tensor([[1, 2],[3, 4]]).to(device)
print(data)
