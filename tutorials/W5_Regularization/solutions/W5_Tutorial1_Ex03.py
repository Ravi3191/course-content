def early_stopping_main(model,train_loader,val_loader,test_data):

    """
        Inputs: 
            Model: Pytorch model
            Loaders: Pytorch Train and Validation loaders

        The function trains the model and terminates the training based on the early stopping criterion.
    """

    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    patience = 5
    wait = 0

    best_acc  = 0.0
    best_epoch = 0

    val_acc_list, train_acc_list = [], []
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train_acc = test(model,device,train_loader, 'Train')
        val_acc = test(model,device,val_loader, 'Val')
        if (val_acc > best_acc):
            best_model = copy.deepcopy(model)
            best_acc = val_acc
            best_epoch = epoch
            wait = 1
        else:
            if (wait > patience):
                print('early stopped on epoch:',epoch)
                break
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    return val_acc_list, train_acc_list, model, best_epoch