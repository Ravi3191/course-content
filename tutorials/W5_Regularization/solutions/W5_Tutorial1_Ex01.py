def calculate_frobenius_norm(model):
    """
        Input: Pytorch newtork
        Output: Frobenious Norm of all the tensors of the model
    """
    norm = 0.0

    for name,param in model.named_parameters():
        norm += torch.norm(param).data         
    return norm

net = nn.Linear(10,1)
print(f'Frobenius Norm of Single Linear Layer: {calculate_frobenius_norm(net)}')