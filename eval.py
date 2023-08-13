import numpy as np
import torch
from metrics import VIM, keypoint_mpjpe
from torchmetrics.classification.accuracy import Accuracy


def eval_vim(config, ds_loader, model, EVAL_FRAMES=[2, 4, 8, 10, 14]):
    OUTPUT_LEN = config.output_len
    y_pred = []
    y_test = []
    running_test_aux_acc = 0.

    for i, inp in enumerate(iter(ds_loader)):

        out = model(inp, False)

        p0 = out["z0"][:, :OUTPUT_LEN].reshape(out["z0"].shape[0], 1, OUTPUT_LEN, config.num_kps, 3).float().detach().cpu().numpy()
        p1 = out["z1"][:, :OUTPUT_LEN].reshape(out["z1"].shape[0], 1, OUTPUT_LEN, config.num_kps, 3).float().detach().cpu().numpy()

        y_pred.extend(np.concatenate((p0, p1), axis=1))
        y_test.extend(np.concatenate((inp["out_keypoints0"].float().detach().cpu().numpy().reshape(inp["out_keypoints0"].shape[0], 1, *inp["out_keypoints0"].shape[1:]),
                                      inp["out_keypoints1"].float().detach().cpu().numpy().reshape(inp["out_keypoints1"].shape[0], 1, *inp["out_keypoints1"].shape[1:])
                                      ), axis=1))

        running_test_aux_acc += torch.sum(torch.argmax(out["s0"].float(), dim=1) == inp["sctx"].reshape(-1).long()) / out["s0"].shape[0]

    test_aux_acc = round( (running_test_aux_acc.detach().cpu().numpy() / (i+1) ) *100)

    y_pred = np.array(y_pred)

    return [round(np.mean( [(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_test)] ) * 100, 1) for LEN in EVAL_FRAMES], test_aux_acc


def eval_mpjpe(config, ds_loader, model, EVAL_FRAMES=[2, 4, 8, 10, 14]):
    OUTPUT_LEN = config.output_len
    y_pred = []
    y_test = []
    running_test_aux_acc = 0.

    for i, inp in enumerate(iter(ds_loader)):

        out = model(inp, False)

        p0 = out["z0"][:, :OUTPUT_LEN].reshape(out["z0"].shape[0], 1, OUTPUT_LEN, config.num_kps, 3).float().detach().cpu().numpy()
        p1 = out["z1"][:, :OUTPUT_LEN].reshape(out["z1"].shape[0], 1, OUTPUT_LEN, config.num_kps, 3).float().detach().cpu().numpy()

        y_pred.extend(np.concatenate((p0, p1), axis=1))
        y_test.extend(np.concatenate((inp["out_keypoints0"].float().detach().cpu().numpy().reshape(inp["out_keypoints0"].shape[0], 1, *inp["out_keypoints0"].shape[1:]),
                                      inp["out_keypoints1"].float().detach().cpu().numpy().reshape(inp["out_keypoints1"].shape[0], 1, *inp["out_keypoints1"].shape[1:])
                                      ), axis=1))

        running_test_aux_acc += torch.sum(torch.argmax(out["s0"].float(), dim=1) == inp["sctx"].reshape(-1).long()) / out["s0"].shape[0]

    test_aux_acc = round( (running_test_aux_acc.detach().cpu().numpy() / (i+1) ) *100, 1)

    y_pred = np.array(y_pred)

    return [round(np.mean( [(keypoint_mpjpe(pred[0][LEN-1:LEN], gt[0][LEN-1:LEN]) + keypoint_mpjpe(pred[1][LEN-1:LEN], gt[1][LEN-1:LEN])) / 2. for pred, gt in zip(y_pred, y_test)] ) * 100, 1) for LEN in EVAL_FRAMES], test_aux_acc


def eval_ctx(config, ds_loader, model, print_all=False):

    ctx_gt, ctx_pred = [], []
    for i, inp in enumerate(iter(ds_loader)):
        out = model(inp, False)
        ctx_pred.append( torch.argmax(out["s0"].float(), dim=1) )#.detach().cpu().numpy() )
        ctx_gt.append( inp["sctx"].reshape(-1).long() )#.detach().cpu().numpy() )

    ctx_pred = torch.cat(ctx_pred, dim=0).cpu()
    ctx_gt = torch.cat(ctx_gt, dim=0).cpu()

    metric = Accuracy(task="multiclass", num_classes=3, average="macro")
    mca = Accuracy(task="multiclass", num_classes=3, average=None)
    ctx_test_aux_acc = round(metric(ctx_pred, ctx_gt).detach().cpu().numpy()*100, 1)
    ctx_test_aux_acc_sep = np.round(mca(ctx_pred, ctx_gt).detach().numpy()*100, 1).astype(np.float32)

    if print_all:
        from torchmetrics.classification.f_beta import F1Score
        from torchmetrics.classification.precision_recall import Precision, Recall

        f1a = F1Score(task="multiclass", num_classes=3, average="macro")
        f1 = F1Score(task="multiclass", num_classes=3, average=None)
        print("F1: ", f1(ctx_pred, ctx_gt), " - ", f1a(ctx_pred, ctx_gt))

        precisiona = Precision(task="multiclass", average='macro', num_classes=3)
        precision = Precision(task="multiclass", average=None, num_classes=3)
        print("Precision: ", precision(ctx_pred, ctx_gt), " - ", precisiona(ctx_pred, ctx_gt))

        recalla = Recall(task="multiclass", average='macro', num_classes=3)
        recall = Recall(task="multiclass", average=None, num_classes=3)
        print("Recall: ", recall(ctx_pred, ctx_gt), " - ", recalla(ctx_pred, ctx_gt))

    return ctx_test_aux_acc, ctx_test_aux_acc_sep


def forecast_cmu_mupots(config, ds, dataset_name, model):
    x_test = []
    y_pred = []
    y_test = []

    for inp in iter(ds):

        input_seq = torch.cat((inp["keypoints0"][:, None, :, :, :], 
                               inp["keypoints1"][:, None, :, :, :], 
                               inp["keypoints2"][:, None, :, :, :]), dim=1).cpu().detach().clone().numpy()
        
        output_seq = torch.cat((inp["out_keypoints0"][:, None, :, :, :], 
                                inp["out_keypoints1"][:, None, :, :, :], 
                                inp["out_keypoints2"][:, None, :, :, :]), dim=1).cpu().detach().clone().numpy()

        out = model(inp, False)

        results = torch.cat((out["z0"][:, None, :, :], out["z1"][:, None, :, :], out["z2"][:, None, :, :]), dim=1).cpu().detach().numpy()

        x_test.extend(input_seq)
        y_test.extend(output_seq)
        y_pred.extend(results)

    y_pred = np.array(y_pred).reshape(-1, 3, 45, config.num_kps, 3)
    y_test = np.array(y_test)
    x_test = np.array(x_test)

    return x_test, y_test, y_pred


def eval_cmu_mupots(config, ds, dataset_name, model):
    loss_list1=[]
    loss_list2=[]
    loss_list3=[]

    n_joints = 15
    
    for inp in iter(ds):

        out = model(inp, False)

        results = torch.cat((out["z0"], out["z1"], out["z2"]), dim=0).cpu().detach()
        output_seq = torch.cat((out["out_keypoints0"], out["out_keypoints1"], out["out_keypoints2"]), dim=0).cpu().detach()

        prediction_1=results[:, :15,:].view(results.shape[0],-1,n_joints,3)
        prediction_2=results[:, :30,:].view(results.shape[0],-1,n_joints,3)
        prediction_3=results[:, :45,:].view(results.shape[0],-1,n_joints,3)
        
        gt_1=output_seq[:, :15,:].view(results.shape[0],-1,n_joints,3)
        gt_2=output_seq[:, :30,:].view(results.shape[0],-1,n_joints,3)
        gt_3=output_seq[:, :45,:].view(results.shape[0],-1,n_joints,3)

        scale = 0.1*1.8/3 # scale back from mix_mocap.py scaling

        loss1=torch.sqrt(((prediction_1/scale - gt_1/scale) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        loss2=torch.sqrt(((prediction_2/scale - gt_2/scale) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        loss3=torch.sqrt(((prediction_3/scale - gt_3/scale) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()

        loss_list1.append(np.mean(loss1))
        loss_list2.append(np.mean(loss2))
        loss_list3.append(np.mean(loss3))

    print("{0} MPJPE: avg 1s: {1:.2f}, 2s: {2:.2f}, 3s: {3:.2f} - {4:.2f}".format(dataset_name, np.mean(loss_list1), np.mean(loss_list2), np.mean(loss_list3), np.mean([np.mean(loss_list1), np.mean(loss_list2), np.mean(loss_list3)])))
    

def eval_mw_mpjpe_cmu_mupots(config, ds, dataset_name, model):

    x_test, y_test, y_pred = forecast_cmu_mupots(config, ds, dataset_name, model)

    pred = np.concatenate((x_test, y_pred), axis=2).astype(np.float32)
    target = np.concatenate((x_test, y_test), axis=2).astype(np.float32)
    
    """
    Movement Weighted MPJPE MW-MPJPE
    @pred expectes connected input and predicted sequences
    @target expectes connected input and output sequences
    """
    mvm = target[:, :, :, 0:1] - target[:, :, :1, 0:1]

    fixed = target - mvm
    mvm = (fixed[:, :, 1:] - fixed[:, :, :-1])

    pred, target, mvm = pred[:, :, 15:], target[:, :, 15:], mvm[:, :, 14:]

    results = np.concatenate((pred[:, 0], pred[:, 1], pred[:, 2]), axis=0)
    output_seq = np.concatenate((target[:, 0], target[:, 1], target[:, 2]), axis=0)
    mvm = np.concatenate((mvm[:, 0], mvm[:, 1], mvm[:, 2]), axis=0)

    mvm_1 = mvm[:, :15, :]
    mvm_2 = mvm[:, :30, :]
    mvm_3 = mvm[:, :45, :]

    prediction_1=results[:, :15, :]
    prediction_2=results[:, :30, :]
    prediction_3=results[:, :45, :]

    gt_1=output_seq[:, :15, :]
    gt_2=output_seq[:, :30, :]
    gt_3=output_seq[:, :45, :]

    scale = 0.1*1.8/3 # scale back from mix_mocap.py scaling

    loss1=np.sqrt(((prediction_1/scale - gt_1/scale) ** 2).sum(axis=-1)).mean(axis=-1).mean(axis=-1).tolist() * np.sum(np.abs(mvm_1).reshape(mvm_1.shape[0], -1), axis=1)
    loss2=np.sqrt(((prediction_2/scale - gt_2/scale) ** 2).sum(axis=-1)).mean(axis=-1).mean(axis=-1).tolist() * np.sum(np.abs(mvm_2).reshape(mvm_2.shape[0], -1), axis=1)
    loss3=np.sqrt(((prediction_3/scale - gt_3/scale) ** 2).sum(axis=-1)).mean(axis=-1).mean(axis=-1).tolist() * np.sum(np.abs(mvm_3).reshape(mvm_3.shape[0], -1), axis=1)

    print("{0} MW-MPJPE: avg 1s: {1:.2f}, 2s: {2:.2f}, 3s: {3:.2f} - {4:.2f}".format(dataset_name, np.mean(loss1), np.mean(loss2), np.mean(loss3), np.mean([np.mean(loss1), np.mean(loss2), np.mean(loss3)])))


    
def test_3dpw(config, args):
    from dataset_3dpw import create_datasets

    if config.independent_ctx:
        from independent_model import create_model
    else:
        from model import create_model
    
    if args.ablation:
        _, _, test_loader, ctx_test_loader = create_datasets(config=config, test_name="somofvalid")
    else:
        _, _, test_loader, ctx_test_loader = create_datasets(config=config)

    model = create_model(config)
    print("Loading model from:", args.out_model_name)
    model.load_state_dict(torch.load(args.out_model_name))
    model.eval()

    test_vim, test_aux_acc = eval_vim(config, test_loader, model)
    ctx_test_aux_acc, ctx_test_aux_acc_sep = eval_ctx(config, ctx_test_loader, model, True)

    print("[Test] VIM: {0} - {1}, CtxAcc: {2}, PerClassCtxAcc: {3}".format(test_vim, round(np.mean(test_vim), 2), ctx_test_aux_acc, ctx_test_aux_acc_sep))

    test_mpjpe, _ = eval_mpjpe(config, test_loader, model)
    print("[Test] MPJPE: {0} - {1}".format(test_mpjpe, round(np.mean(test_mpjpe), 2)))


def test_cmu_mupots(config, args):
    from dataset_cmu_mupots import create_datasets
    from model import create_model

    _, cmu_test_loader, mupots_test_loader = create_datasets(config=config)

    model = create_model(config, persons="3")
    print("Loading model from:", args.out_model_name)
    model.load_state_dict(torch.load(args.out_model_name))
    model.eval()

    eval_cmu_mupots(config, cmu_test_loader, "mocap", model)
    eval_cmu_mupots(config, mupots_test_loader, "mupots", model)

    eval_mw_mpjpe_cmu_mupots(config, cmu_test_loader, "mocap", model)
    eval_mw_mpjpe_cmu_mupots(config, mupots_test_loader, "mupots", model)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="3dpw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_dct", action="store_true")
    parser.add_argument("--dct_n", type=int, default=30)
    parser.add_argument("--out_model_name", type=str)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--independent_ctx", action="store_true")
    args = parser.parse_args()

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    class Config:
        input_len = 16
        output_len = 14
        device = "cpu"
        use_ctx_loss = True
        num_kps = 13
        dct_n = 30
        use_dct = True

        def __str__(self):
            res = vars(self).copy()
            return str(res)

    config = Config()

    config.device = args.device
    config.use_dct = args.use_dct
    config.dct_n = args.dct_n
    config.independent_ctx = args.independent_ctx
    args.out_model_name = "./models/{}.pt".format(args.out_model_name)

    print("Config:", config)

    if args.dataset == "3dpw":
        test_3dpw(config, args)
    else:
        config.input_len = 15
        config.output_len = 45
        config.num_kps = 15
        test_cmu_mupots(config, args)


