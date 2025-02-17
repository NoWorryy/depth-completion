import torch

args = torch.tensor([[0,1,0,3,0,5],
                     [0,1,1,3,1,0],
                     [0,1,3,3,3,5],
                     [0,1,5,3,5,5]])

args = args.unsqueeze(0).unsqueeze(0)   # (b,1,4,6)
print(args.shape)
img = torch.tensor([[1.,2.,3.],
                    [4.,5.,6.]])

img = img.unsqueeze(0).unsqueeze(0)    # (b,1,2,3)
img = img.view(1,1,1,-1)  # (b,1,1,6)
print(img.shape)
img = img.repeat(1,1,4,1)
print(img.shape)

x = torch.gather(img, dim=-1, index=args)   # (b,1,4,6)
x = torch.mean(x, dim=-2)   # (b,1,6)
x = x.view(1,1,2,3)
print(x.shape)
print(x)
