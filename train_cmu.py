import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from dataset_cmu_mupots import create_datasets
from model import create_model


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def loss(output, target):
    loss = torch.mean( torch.norm(output.reshape(-1,3) - target.reshape(-1,3), 2, 1) )
    return loss


def train_one_epoch(config, model, train_loader, optimizer, scheduler, loss_fnc, sctx_loss_fnc):
    running_loss = 0.
    running_ctx_acc = torch.zeros(1).to(config.device)[0]

    for i, x_orig in enumerate(iter(train_loader)):

        x = x_orig.copy()

        optimizer.zero_grad()

        outputs = model(x, True)

        loss = (loss_fnc(outputs["z0"].reshape(*x["out_keypoints0"].shape).float(), x["out_keypoints0"].float()) +
                loss_fnc(outputs["z1"].reshape(*x["out_keypoints1"].shape).float(), x["out_keypoints1"].float())) / 2.

        running_loss += loss.item()

        if config.use_ctx_loss:
            loss += sctx_loss_fnc(outputs["s01"].float(), x["sctx01"].reshape(-1).long()) * (0.01/3)
            loss += sctx_loss_fnc(outputs["s12"].float(), x["sctx12"].reshape(-1).long()) * (0.01/3)
            loss += sctx_loss_fnc(outputs["s02"].float(), x["sctx02"].reshape(-1).long()) * (0.01/3)
            running_ctx_acc += torch.sum(torch.argmax(outputs["s01"].float(), dim=1) == x["sctx01"].reshape(-1).long()) / outputs["s01"].shape[0]
            running_ctx_acc += torch.sum(torch.argmax(outputs["s12"].float(), dim=1) == x["sctx12"].reshape(-1).long()) / outputs["s12"].shape[0]
            running_ctx_acc += torch.sum(torch.argmax(outputs["s02"].float(), dim=1) == x["sctx02"].reshape(-1).long()) / outputs["s02"].shape[0]

        loss.backward()

        optimizer.step()

    avg_loss = running_loss / (i+1)
    avg_ctx_acc = running_ctx_acc / (i+1) / 3

    scheduler.step()

    return avg_loss, avg_ctx_acc


def train(config):

    EPOCHS = 500
    train_loader, _, _ = create_datasets(config=config)

    model = create_model(config, "3")
    print("#Param:", sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 200, 400], gamma=0.1)

    loss_fnc = loss
    sctx_loss_fnc = nn.CrossEntropyLoss()

    for epoch_number in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss, avg_ctx_acc = train_one_epoch(config, model, train_loader, optimizer, scheduler, loss_fnc, sctx_loss_fnc)
        model.train(False)

        print('LOSS train: {}, ctx_acc: {}% - lr: {}'.format(avg_loss, torch.round(avg_ctx_acc*100), get_lr(optimizer)))

        epoch_number += 1

    return model


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_ctx_loss", action="store_true")
    parser.add_argument("--backward_movement", action="store_true")
    parser.add_argument("--reversed_order", action="store_true")
    parser.add_argument("--random_scale", action="store_true")
    parser.add_argument("--random_rotate_x", action="store_true")
    parser.add_argument("--random_rotate_y", action="store_true")
    parser.add_argument("--random_rotate_z", action="store_true")
    parser.add_argument("--random_reposition", action="store_true")
    parser.add_argument("--use_full_augmentation", action="store_true")
    parser.add_argument("--use_dct", action="store_true")
    parser.add_argument("--dct_n", type=int, default=40)
    parser.add_argument("--out_model_name", type=str)
    args = parser.parse_args()

    #num_workers = 10
    #device = torch.device(args.device)

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)


    class Aug:
        backward_movement = True
        reversed_order = True
        random_scale = True
        random_rotate_x = True
        random_rotate_y = True
        random_rotate_z = True
        random_reposition = True

    class Config:
        input_len = 15
        output_len = 45 # 15
        device = "cpu"
        num_kps = 15
        use_ctx_loss = True
        dct_n = 40
        augment = Aug()

        def __str__(self):
            res = vars(self).copy()
            res["augment"] = str(vars(self.augment))
            return str(res)

    config = Config()

    config.device = args.device
    config.use_ctx_loss = args.use_ctx_loss
    config.use_dct = args.use_dct
    config.dct_n = args.dct_n
    config.augment.backward_movement = args.backward_movement
    config.augment.reversed_order = args.reversed_order
    config.augment.random_scale = args.random_scale
    config.augment.random_rotate_x = args.random_rotate_x
    config.augment.random_rotate_y = args.random_rotate_y
    config.augment.random_rotate_z = args.random_rotate_z
    config.augment.random_reposition = args.random_reposition

    if args.use_full_augmentation:
        config.augment.backward_movement = config.augment.reversed_order = config.augment.random_scale = config.augment.random_rotate_x = config.augment.random_rotate_y = config.augment.random_rotate_z = config.augment.random_reposition = True

    print("Config:", config)

    model = train(config)

    if args.out_model_name:
        model_path = './models/{}.pt'.format(args.out_model_name)
        torch.save(model.state_dict(), model_path)
