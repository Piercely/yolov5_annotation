import argparse
import logging
import os
import random
import shutil
import time
from pathlib import Path
from warnings import warn

import math
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging, init_seeds)
from utils.google_utils import attempt_download
from utils.torch_utils import ModelEMA, select_device, intersect_dicts

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    logger.info(f'Hyperparameters {hyp}')
    # 获取记录训练日志的路径
    """
    训练日志包括：权重、tensorboard文件、超参数hyp、设置的训练参数opt(也就是epochs,batch_size等),result.txt
    result.txt包括: 占GPU内存、训练集的box loss, objectness loss, classification loss, 总loss, 
    targets的数量, 输入图片分辨率, 准确率TP/(TP+FP),召回率TP/P ; 
    验证集(测试集)的mAP50, mAP@0.5:0.95, box loss, objectness loss, classification loss.
    还会保存batch<3（前三个batch）的ground truth
    """
    # 如果设置进化算法则不会传入tb_writer(则为None)，设置一个evolve文件夹作为日志目录
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
    # 设置保存权重的路径
    wdir = log_dir / 'weights'  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    # 设置保存results的路径
    results_file = str(log_dir / 'results.txt')
    # 获取轮次、批次、总批次(涉及到分布式训练)、权重、进程序号(主要用于分布式训练)
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Save run settings
    # 保存hyp和opt
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    # torch_distributed_zero_first同步所有进程
    # check_dataset检查数据集，如果没找到数据集则下载数据集(仅适用于项目中自带的yaml文件数据集)
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(2 + rank) # 设置随机种子
    # 加载数据配置信息
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    # 获取训练集、测试集图片路径
    train_path = data_dict['train']
    test_path = data_dict['val']
    # 获取类别数量和类别名字
    # 如果设置了opt.single_cls则为一类
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    # 如果采用预训练
    if pretrained:
        # 加载模型，从google云盘中自动下载模型
        # 但通常会下载失败，建议提前下载下来放进weights目录
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        # 加载模型及参数
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        """
        这里模型创建，可通过opt.cfg，也可通过ckpt['model'].yaml
        这里的区别在于是否是resume，resume时会将opt.cfg设为空，
        则按照ckpt['model'].yaml创建模型；
        这也影响着下面是否除去anchor的key(也就是不加载anchor)，如果resume则不加载anchor
        主要是因为保存的模型会保存anchors，有时候用户自定义了anchor之后，再resume，则原来基于coco数据集的anchor就会覆盖自己设定的anchor，
        参考https://github.com/ultralytics/yolov5/issues/459
        所以下面设置了intersect_dicts，该函数就是忽略掉exclude 
        """
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        # 显示加载预训练权重的的键值对和创建模型的键值对
        # 如果pretrained为True，则会少加载两个键值对(anchors,anchor_grid)
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        # 创建模型，ch为输入图片通道
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create

    # Freeze
    """
    冻结模型层,设置冻结层名字即可,
    具体可以查看https://github.com/ultralytics/yolov5/issues/679
    其实上面只是给一个freeze的示例：
    You can add any parameters you want to this list, with full or partial names, 
    to freeze them before training starts. 
    This code freezes all weights, leaving only biases with active gradients:
    """
    freeze = ['', ]  # parameter names to freeze (full or partial)
    if any(freeze):
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False # 不进行梯度计算

    # Optimizer
    """
    nbs为标称的batch_size; 
    比如默认的话上面设置的opt.batch_size为16,nbs为64，
    则模型梯度累积了64/16=4(accumulate)次之后
    再更新一次模型，变相的扩大了batch_size
    """
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置权重衰减系数
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    # 将模型分成三组(weight, bias, 其它所有参数)进行优化
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    # 选用优化器，并设置pg0组的优化方式
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 设置weight的优化方式
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # 设置biases的优化方式
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # 打印优化信息
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # 设置学习率衰减，这里为余弦退火方式进行衰减
    # 就是根据以下公式lf,epoch和超参数hyp['lrf']进行衰减
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume 断点续训
    # 初始化开始训练的epoch和最好的结果
    # best_fitness是以[0.0, 0.0, 0.1, 0.9]为系数并乘以[精确度, 召回率, mAP@0.5, mAP@0.5:0.95]再求和所得
    # 根据best_fitness来保存best.pt
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        # 加载训练结果result.txt
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        """
        如果resume,则备份权重
        尽管目前resume能够近似100%成功起作用,参照:https://github.com/ultralytics/yolov5/pull/756
        但为了防止resume时出现其他问题导致把之前的权重覆盖了,在这里进行备份,参照:https://github.com/ultralytics/yolov5/pull/765
        """
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
            shutil.copytree(wdir, wdir.parent / f'weights_backup_epoch{start_epoch - 1}')  # save previous weights
        """
        如果新设置epochs小于加载的epoch，
        则视新设置的epochs为需要再训练的轮次数而不再是总的轮次数
        """
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    # 获取模型总步长和模型输入图片分辨率
    gs = int(max(model.stride))  # grid size (max stride)
    # 检查输入图片分辨率确保能够整除总步长gs
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    # 分布式训练,参照:https://github.com/ultralytics/yolov5/issues/475
    # DataParallel模式,仅支持单机多卡
    # rank为进程编号, 如果设置为rank=-1并且有多块gpu，则使用DataParallel模式
    # rank=-1且gpu数量=1时,不会进行分布式
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    # 使用跨卡同步BN
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Exponential moving average
    # 为模型创建EMA指数滑动平均,如果GPU进程数大于1,则不创建
    # EMA（指数移动平均）:一种给予近期数据更高权重的平均方法
    # 用EMA方法对模型的参数做平均，以提高测试指标并增加模型鲁棒(减少模型权重抖动)
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    # 如果rank不等于-1,则使用DistributedDataParallel模式
    # local_rank为gpu编号,rank为进程,例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU。
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Trainloader
    # 创建训练集dataloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)
    
    # 获取标签中最大的类别值，并与类别数作比较，如果小于类别数则表示有问题
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        # 更新EMA模型的updates参数,保持EMA的平滑性
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        # 创建测试集dataloader
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,
                                       hyp=hyp, augment=False, cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers)[0]  # testloader

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            plot_labels(labels, save_dir=log_dir)
            if tb_writer:
                # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
                tb_writer.add_histogram('classes', c, 0)

            # Anchors
            """
            计算默认锚框anchor与数据集标签框的宽高比值
            标签的高h宽w与anchor的高h_a宽w_a的比值, 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
            如果bpr小于98%，则根据k-mean算法聚类新的锚框
            """
            # 计算最佳anchor
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Model parameters
    # 根据自己数据集的类别数设置分类损失的系数
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    # 设置类别数，超参数
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    """
    设置iou的值在objectness loss中做标签的系数, 使用代码如下
    tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)
    这里model.gr=1，也就是说完全使用标签框与预测框的iou值来作为该预测框的objectness标签
    """
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # 从训练样本标签得到类别权重（和类别中的目标数--即类别频率--成反比）
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    # 获取类别的名字
    model.names = names

    # Start training
    t0 = time.time()
    # 获取热身训练的迭代次数
    nw = max(round(hyp['warmup_epochs'] * nb), 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    # 初始化mAP和results
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    """
    设置学习率衰减所进行到的轮次，
    目的是打断训练后，--resume接着训练也能正常的衔接之前的训练进行学习率衰减
    """
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 通过torch1.6以上自带的api设置混合精度训练
    scaler = amp.GradScaler(enabled=cuda) # 在训练最开始之前实例化一个GradScaler对象
    """
    打印训练和测试输入图片分辨率
    加载图片时调用的cpu进程数
    从哪个epoch开始训练
    """
    logger.info('Image sizes %g train, %g test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, log_dir, epochs))
    # 训练
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            """
            如果设置进行图片采样策略，
            则根据前面初始化的图片采样权重model.class_weights以及maps配合每张图片包含的类别数
            通过random.choices生成图片索引indices从而进行采样
            """
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # 得到图片采样权重
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP；如果是DDP模式,则广播采样策略
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                # 广播索引到其他group
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            # DDP模式下打乱数据, ddp.sampler的随机采样数据是基于epoch+seed作为随机种子，
            # 每次epoch不同，随机种子就不同
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        if rank in [-1, 0]:
            # tqdm 创建进度条，方便训练时信息的展示
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad() # 梯度清零
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # 计算迭代的次数iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            """
            热身训练(前nw次迭代)
            在前nw次迭代中，根据以下方式选取accumulate和学习率
            """
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    """
                    bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                    其他的参数学习率从0增加到lr*lf(epoch).
                    lf为上面设置的余弦退火的衰减函数
                    """
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    # 动量momentum也从0.9慢慢变到hyp['momentum'](default=0.937)
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            # 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            # 混合精度训练
            with amp.autocast(enabled=cuda): # 开启autocast的context managers语义(model + loss)
                pred = model(imgs)  # forward；前向传播
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失         
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size
                if rank != -1:
                    # 平均不同gpu之间的梯度
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward() # 反向传播；Scales loss. 为了梯度放大.

            # Optimize
            if ni % accumulate == 0: # 模型反向传播accumulate次之后再根据累积的梯度更新一次参数
                # scaler.step() 首先把梯度的值unscale回来.
                # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                scaler.step(optimizer)  # optimizer.step 进行参数更新
                # 准备着，看是否要增大scaler
                scaler.update()
                optimizer.zero_grad() # 梯度清零
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                # 进度条显示以上信息
                pbar.set_description(s)

                # Plot
                # 将前三次迭代batch的标签框在图片上画出来并保存
                if ni < 3:
                    f = str(log_dir / f'train_batch{ni}.jpg')  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    # if tb_writer and result is not None:
                    # tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        # 进行学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step() # 对lr进行调整

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema:
                # 更新EMA的属性
                # 添加include的属性
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            # 判断该epoch是否为最后一轮
            final_epoch = epoch + 1 == epochs
            # 对测试集进行测试，计算mAP等指标
            # 测试时使用的是EMA模型
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=log_dir,
                                                 plots=epoch == 0 or final_epoch)  # plot first and last
            # Write
            # 将指标写入result.txt
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n') # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            # 如果设置opt.bucket, 上传results.txt到谷歌云盘
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Tensorboard
            # 添加指标，损失等信息以便tensorboard显示
            if tb_writer:
                tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
                for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            # 更新best_fitness
            fi = fitness(np.array(results).reshape(1, -1)) # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            """
            保存&加载带checkpoint的模型用于inference或resuming training；
            保存模型，还保存了epoch，results，optimizer等信息，
            optimizer将不会在最后一轮完成后保存
            model保存的是EMA的模型
            """
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f: # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        """
        模型训练完后，strip_optimizer函数将optimizer从ckpt中去除；
        并且对模型进行model.half(), 将Float32的模型->Float16，
        可以减少模型大小，提高inference速度
        """
        n = opt.name if opt.name.isnumeric() else ''
        fresults, flast, fbest = log_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'
        for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                if str(f2).endswith('.pt'):  # is *.pt
                    strip_optimizer(f2)  # strip optimizer；保存为半精度
                    os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None  # upload
        # Finish
        # 可视化results.txt文件
        if not opt.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    # 释放显存
    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser() # 建立参数解析对象parser
    """
    opt参数解析：
    cfg: 模型配置文件，网络结构
    data: 数据集配置文件，数据集路径，类名等
    hyp: 超参数文件
    epochs: 训练总轮次
    batch-size: 批次大小
    img-size: 输入图片分辨率大小
    rect: 是否采用矩形训练，默认False
    resume: 断点续训(从上次打断训练的结果接着训练)
    nosave: 不保存模型，默认False
    notest: 不进行test，默认False
    noautoanchor: 不自动调整anchor，默认False
    evolve: 是否进行超参数进化，默认False
    bucket: 谷歌云盘bucket，一般不会用到
    cache-images: 是否提前缓存图片到内存，以加快训练速度，默认False
    weights: 加载的权重文件
    name: 训练结果文件名，如果设置：results.txt（默认） to results_name.txt
    device: 训练的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
    multi-scale: 是否进行多尺度训练，默认False
    single-cls: 数据集是否只有一个类别，默认False
    adam: 是否使用adam优化器
    sync-bn: 是否使用跨卡同步BN,在DDP模式使用
    local_rank: gpu编号
    logdir: 存放日志的目录
    workers: dataloader的最大worker数量
    """
    # 添加属性：给xx实例增加一个aa属性，如 xx.add_argument("aa")    
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--name', default='', help='renames experiment folder exp{N} to exp{N}_{name} if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    # 采用parser对象的parse_args函数获取解析的参数
    opt = parser.parse_args()

    # Set DDP variables
    """
    设置DDP模式的参数
    world_size:表示全局进程个数
    global_rank:进程编号
    """
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()  # 检查你的代码版本是否为最新的(不适用于windows系统)

    # Resume
    if opt.resume:  # resume an interrupted run 断点续训
        # 如果resume是str,则表示传入的是模型的路径地址
        # get_latest_run()函数获取runs文件夹中最近的last.pt
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        # opt参数也全部替换
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        # opt.cfg设置为'' 对应着train函数里面的操作(加载权重时是否加载权重里的anchor)
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)

    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        # 检查配置文件信息
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # 扩展image_size为[image_size, image_size]一个是训练size，一个是测试size
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        # 根据opt.logdir生成目录
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp0 /exp1 ...

    # DDP mode
    # 选择设备
    device = select_device(opt.device, batch_size=opt.batch_size)

    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank) # 根据gpu编号选择设备
        # 初始化进程组
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        # 将总批次按照进程数分配给各个gpu
        opt.batch_size = opt.total_batch_size // opt.world_size

    # 打印opt参数信息
    logger.info(opt)
    # Hyperparameters 加载超参数列表
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    # Train
    logger.info(opt)
    # 如果不进行超参数进化，则直接调用train()函数，开始训练
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            # 创建tensorboard
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.logdir}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0

        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参数进化列表,括号里分别为(突变规模, 最小值,最大值)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.logdir) / 'evolve' / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        # 默认进化300代
        """
        这里的进化算法是：根据之前训练时的hyp来确定一个base hyp再进行突变；
        具体方法：
        通过之前每次进化得到的results来确定之前每个hyp的权重
        有了每个hyp和每个hyp的权重之后有两种进化方式；
        1.single: 根据每个hyp的权重随机选择一个之前的hyp作为base hyp，random.choices(range(n), weights=w)
        2.weighted: 根据每个hyp的权重对之前所有的hyp进行融合获得一个base hyp，(x * w.reshape(n, 1)).sum(0) / w.sum()
        evolve.txt会记录每次进化之后的results+hyp
        每次进化时，hyp会根据之前的results进行从大到小的排序；
        再根据fitness函数计算之前每次进化得到的hyp的权重
        再确定哪一种进化方式，从而进行进化
        """

        for _ in range(300):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                # 选择进化方式
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                # 加载evolve.txt
                x = np.loadtxt('evolve.txt', ndmin=2)
                # 选取至多前5次进化的结果
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                # 根据results计算hyp的权重
                w = fitness(x) - fitness(x).min()  # weights
                # 根据不同进化方式获得base hyp
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # random selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                # 超参数进化
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                # 获取突变初始值
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                # 设置突变
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # 将突变添加到base hyp上
                # [i+7]是因为x中前七个数字为results的指标(P, R, mAP, F1, test_losses=(box, obj, cls))，之后才是超参数hyp
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            # 修剪hyp在规定范围里
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            # 训练
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            """
            写入results和对应的hyp到evolve.txt
            evolve.txt文件每一行为一次进化的结果
            一行中前七个数字为(P, R, mAP, F1, test_losses=(box, obj, cls))，之后为hyp
            保存hyp到yaml文件
            """
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
