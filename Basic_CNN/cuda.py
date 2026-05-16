import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print("GPU", i)
    print("name:", torch.cuda.get_device_name(i))
    print("properties:", torch.cuda.get_device_properties(i))