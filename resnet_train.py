import torch.backends.cudnn as cudnn
from models import model

if __name__ == '__main__':
    cudnn.enabled = True


    n_epochs = 20
    gpu = 0
    loss_func = 

def train_bel(n_epochs=20, gpu=0, code='u', code_bits=200, n_bits=10, loss_func='bce'):

    # load model and pretrained weights
    model = models.ResNet(block=torchvision.models.resnet.Bottleneck,
                          layers=[3, 4, 6, 3],
                          n_bits=n_bits,
                          code='u')

    state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    model.load_state_dict(state_dict)

    # load dataset & apply tranforms
    transformations = transforms.Compose([
                  transforms.Scale(240),
                  transforms.RandomCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pose_dataset = datasets.Pose_300W_LP_random_ds_bcd('datasets/300W_LP/',
                                                       "datasets/300W_LP/overall_filelist_clean.txt",
                                                       transformations,
                                                       code_bits=code_bits,
                                                       code=code)

    test_pose_dataset = datasets.AFLW2000_bcd("datasets/AFLW2000/",
                                              "datasets/AFLW2000/overall_filelist_clean.txt",
                                              transformations,
                                              code_bits=code_bits,
                                              code=code)

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    helper.train(model,train_loader, test_loader, pose_dataset, args.output_string, gpu, args.arch,args.lr,20,batch_size,195,args.num_bits,args.code_bits,args.code,loss_func,di,dis)

def convert_belu():
    encode = (encode.sign()+1)/2
    number=torch.sum(encode,dim=1)
    number=number.cpu()
    return number

def train(model, train_loader, test_loader, pose_dataset, output_string, gpu, arch, lr, num_epochs, batch_size, val_bound, num_bits, code_bits, code, loss_func, di, dis):
    model.cuda(gpu)

    criterion = nn.BCEWithLogitsLoss(reduction="sum").cuda(gpu)
    crit_CE = nn.CrossEntropyLoss(reduction="sum").cuda(gpu)
    crit_MSE = nn.MSELoss(reduction="mean").cuda(gpu)
    crit_L1 = nn.L1Loss(reduction="mean").cuda(gpu)

    optimizer = torch.optim.Adam([
        {'params': get_ignored_params(model, arch), 'lr': 0},
        {'params': get_non_ignored_params(model, arch), 'lr': lr},
        {'params': get_fc_params(model, arch), 'lr': lr * 5}
    ], lr=lr)

    di = torch.transpose(di, 0, 1).cuda()
    dis = torch.transpose(dis, 0, 1).cuda()
    for epoch in range(num_epochs):
        model.train()
        if epoch == 10:
            lr /= 10
            optimizer = torch.optim.Adam([
                {'params': get_ignored_params(model, arch), 'lr': 0},
                {'params': get_non_ignored_params(model, arch), 'lr': lr},
                {'params': get_fc_params(model, arch), 'lr': lr * 5}
            ], lr=lr)

        total, train_loss, test_loss = 0.0, 0.0, 0.0
        trpy_error = [0.0, 0.0, 0.0]

        for i, (images, labels, cont_labels, name, trpy_error) in tqdm(enumerate(train_loader)):
            tyaw, tpitch, troll = tyaw.cuda(gpu), tpitch.cuda(gpu), troll.cuda(gpu)
            images = Variable(images).cuda(gpu)
            yaw, pitch, roll, rangles = model(images)
            angles = torch.cat((yaw, pitch, roll), dim=1)
            bout = torch.cat((tyaw, tpitch, troll), dim=1)

            if loss
                print("Epoch %d Complete: Avg. Training Loss: %s" % (epoch, train_loss/i))
                f.write("Epoch %d Complete: Avg. Training Loss: %s \n" % (epoch, train_loss/i))
                
                if epoch%val_bound==0 and epoch!=0:
                    mae, test_loss = evaluate(model,test_loader,criterion,gpu,convert_func,arch,code_bits,code,di,dis)
                    if mae < best_MAE:
                        print("Found better model %.4f -> %.4f, saving to best_snapshot" % (best_MAE,mae))
                        f.write("Found better model %.4f -> %.4f, saving to best_snapshot \n" % (best_MAE,mae))
                        best_MAE = mae
                        torch.save(model.state_dict(),os.environ['TMPDIR']+"/output/snapshots/best_snapshot")
                        print("Model saved as %s/best_snapshot" % (os.environ['TMPDIR']+"/output/snapshots/"))
                        f.write("Model saved as %s/best_snapshot \n" % (os.environ['TMPDIR']+"/output/snapshots/"))
                        
                    print("Validation MAE: %.4f, Avg. Test Loss: %s" % (mae, test_loss))
                    f.write("Validation MAE: %.4f, Avg. Test Loss: %s \n" % (mae, test_loss))
                    
                if epoch%5==0:
                    print("Snapshotting model")
                    f.write("Snapshotting model \n")
                    torch.save(model.state_dict(),os.environ['TMPDIR']+"/output/snapshots/epoch_%d" % (epoch))
                    print("Model saved as %s/epoch_%d" % (os.environ['TMPDIR']+"/output/snapshots/",epoch))
                    f.write("Model saved as %s/epoch_%d \n" % (os.environ['TMPDIR']+"/output/snapshots/",epoch))
                    print("Ready to train network.")
            f.close()
            return model
