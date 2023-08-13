import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from dataset_3dpw import create_datasets


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

        loss = (loss_fnc(outputs["z0"][:, :config.output_len].reshape(*x["out_keypoints0"].shape).float(), x["out_keypoints0"].float()) +
                loss_fnc(outputs["z1"][:, :config.output_len].reshape(*x["out_keypoints1"].shape).float(), x["out_keypoints1"].float())) / 2.

        running_loss += loss.item()

        if config.use_ctx_loss:
            loss += sctx_loss_fnc(outputs["s0"].float(), x["sctx"].reshape(-1).long()) * 0.01
            running_ctx_acc += torch.sum(torch.argmax(outputs["s0"].float(), dim=1) == x["sctx"].reshape(-1).long()) / outputs["s0"].shape[0]

        loss.backward()

        optimizer.step()

    avg_loss = running_loss / (i+1)
    avg_ctx_acc = running_ctx_acc / (i+1)

    scheduler.step()

    return avg_loss, avg_ctx_acc


def calc_ds_loss(config, model, ds_loader, loss_fnc):
    running_vloss = 0.0

    for i, vx in enumerate(iter(ds_loader)):

        voutputs = model(vx, False)
        vloss = (loss_fnc(voutputs["z0"][:, :config.output_len].reshape(*vx["out_keypoints0"].shape).float(), vx["out_keypoints0"].float()) +
                 loss_fnc(voutputs["z1"][:, :config.output_len].reshape(*vx["out_keypoints1"].shape).float(), vx["out_keypoints1"].float())) / 2.

        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    return avg_vloss


def train(config):

    EPOCHS = 500

    if config.ablation:
        train_loader, valid_loader, test_loader, ctx_test_loader = create_datasets(config=config, test_name="somofvalid")
    else:
        train_loader, valid_loader, test_loader, ctx_test_loader = create_datasets(config=config)

    if config.independent_ctx:
        from independent_model import create_model
    else:
        from model import create_model
        
    model = create_model(config)
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

        avg_vloss = calc_ds_loss(config, model, valid_loader, loss_fnc)

        print('LOSS train: {}, valid: {}, ctx_acc: {}%  - lr: {}'.format(avg_loss, avg_vloss, torch.round(avg_ctx_acc*100), get_lr(optimizer)))

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
    parser.add_argument("--out_model_name", type=str)
    parser.add_argument("--dct_n", type=int, default=30)
    parser.add_argument("--independent_ctx", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    #num_workers = 10
    device = torch.device(args.device)

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

    class DSConfig:
        train_ds = ""
        valid_ds = ""
        test_ds = ""

    class Config:
        input_len = 16
        output_len = 14
        device = "cpu"
        num_kps = 13
        use_ctx_loss = True
        use_dct = True
        dct_n = 30
        augment = Aug()

        def __str__(self):
            res = vars(self).copy()
            res["augment"] = str(vars(self.augment))
            return str(res)

    config = Config()

    config.ablation = args.ablation
    config.independent_ctx = args.independent_ctx
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
        import os
        if not os.path.exists("./models"):
            os.makedirs("./models", exist_ok=True)

        model_path = './models/{}.pt'.format(args.out_model_name)
        torch.save(model.state_dict(), model_path)
