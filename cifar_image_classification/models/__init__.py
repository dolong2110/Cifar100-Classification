def get_model(args):
    if args.model == 'resnet18':
        from resnet import resnet18
        model = resnet18()
    elif args.model == 'resnet34':
        from resnet import resnet34
        model = resnet34()
    elif args.model == 'resnet50':
        from resnet import resnet50
        model = resnet50()
    elif args.model == 'resnet101':
        from resnet import resnet101
        model = resnet101()
    elif args.model == 'resnet152':
        from resnet import resnet152
        model = resnet152()
    elif args.model == 'mobilenet':
        from mobilenet import mobilenet
        model = mobilenet()
    else:
        raise 'Model {} not available'.format(args.model)

    if args.gpu:  # use_gpu
        model = model.cuda()

    return model