import argparse # python的命令行解析的模块，内置于python，不需要安装
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class, check_dataset, check_file, check_img_size, compute_loss, non_max_suppression, scale_coords,
    xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class, set_logging)
from utils.torch_utils import select_device, time_synchronized


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images;保存在测试时第一个batch的图片上画出标签框和预测框的图片路径
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True):
    # Initialize/load model and set device
    # 判断是否在训练时调用test，如果是则获取训练时的设备
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
         # 选择设备
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels
        
        # Remove previous
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)  # delete dir
        os.makedirs(save_dir)  # make new dir

        if save_txt:
            out = save_dir / 'autolabels'
            if os.path.exists(out):
                shutil.rmtree(out)  # delete dir
            os.makedirs(out)  # make new dir
            
        # Load model； 加载模型
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # 检查输入图片分辨率是否能被32整除
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    # 如果设备不是cpu并且gpu数目为1，则将模型由Float32转为Float16，提高前向传播的速度
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half() # GPU上使用FP16推理

    # Configure
    # eval()时，框架会自动把BN和DropOut固定住，用训练好的值; 不启用 BatchNormalization 和 Dropout
    model.eval()
    # 加载数据配置信息
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict 字典
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 设置iou阈值，从0.5~0.95，每间隔0.05取一次
    # iouv iou值得列表[0.5, 0.55, 0.6,..., 0.95]
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    # iou个数=10
    niou = iouv.numel()

    # Dataloader
    if not training:
        # 创建一个全0数组测试一下前向传播是否正常运行
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        # 获取图片路径
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        # 创建dataloader
        # 注意这里rect参数为True，yolov5的测试评估是基于矩形推理的
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]

    # 初始化测试的图片数量
    seen = 0
    # 获取类别的名字
    names = model.names if hasattr(model, 'names') else model.module.names
    """
    获取coco数据集的类别索引
    coco数据集有80个类别(索引范围应该为0~79)，但是其索引却属于1~90
    coco80_to_coco91_class()就是为了与上述索引对应起来，返回一个范围在1~90的索引数组
    """
    coco91class = coco80_to_coco91_class()
    # 设置tqdm进度条的显示信息
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化指标，时间
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    # 初始化测试集的损失
    loss = torch.zeros(3, device=device)
    # 初始化json文件的字典，统计信息，ap
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        # 图片也由Float32->Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            
            """
            time_synchronized()函数里面进行了torch.cuda.synchronize()，再返回的time.time()
            torch.cuda.synchronize()等待gpu上完成所有的工作，这样测试时间会更准确 
            """
            t = time_synchronized()
            # 前向传播；inf_out为预测结果, train_out训练结果
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            # t0累计前向传播的时间
            t0 += time_synchronized() - t

            # Compute loss
            # 如果是在训练时进行的test，则通过训练结果计算并返回测试集的box, obj, cls损失
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls
            # Run NMS
            # t1累计后处理NMS的时间
            t = time_synchronized()
            
            """
            non_max_suppression进行非极大值抑制;
            conf_thres为置信度阈值，iou_thres为iou阈值，merge为是否合并框
            """
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t

        # Statistics per image
        # 为每一张图片做统计, 写入预测信息到txt文件, 生成json文件字典, 统计tp等
        for si, pred in enumerate(output):
            # 获取第si张图片的标签信息, 包括class,x,y,w,h
            # targets[:, 0]为标签属于哪一张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            # 获取标签类别
            tcls = labels[:, 0].tolist() if nl else []  # target class
            # 统计测试图片数量
            seen += 1

            # 如果预测为空，则添加空的信息到stats里
            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # 保存预测结果为txt文件
            if save_txt:
                # 获得对应图片的高和宽
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                x = pred.clone()
                # 将预测框的坐标调整到基于其原本宽高的坐标
                x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in x:
                    # xyxy格式->xywh, 并对坐标进行归一化处理
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, conf, *xywh) if save_conf else (cls, *xywh)  # label format
                    # 保存预测类别和坐标到txt文件
                    with open(str(out / Path(paths[si]).stem) + '.txt', 'a') as f:
                        f.write(('%g ' * len(line) + '\n') % line)
                        
            # Clip boxes to image bounds
            # 修正预测坐标到图片内部
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            # 保存coco格式的json文件字典
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                # coco格式json文件大概包含信息如上
                # 获取图片id
                image_id = Path(paths[si]).stem
                # 获取框坐标信息
                box = pred[:, :4].clone()  # xyxy
                # 将框调整为基于原图大小的
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                # 转换为xywh格式
                box = xyxy2xywh(box)  # xywh
                
                """
                注意：之前所说的xyxy格式为左上角右下角的坐标，xywh是中心点坐标和宽高
                而coco的json格式中的框坐标格式为xywh,此处的xy为左上角坐标
                也就是coco的json格式的坐标格式为：左上角坐标+宽高
                所以下面一行代码就是将：中心点坐标->左上角
                """
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                
                """
                image_id：图片id, 即属于哪张图
                category_id: 类别, coco91class()从索引0~79映射到索引0~90
                bbox：框的坐标
                score：置信度得分
                """
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            # 初始化预测评定，niou为iou阈值的个数
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                # detected用来存放已检测到的目标
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                # 获得xyxy格式的框并乘以wh
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                # 对图片中的每个类单独处理
                for cls in torch.unique(tcls_tensor):
                    # 标签框该类别的索引
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # 预测框该类别的索引
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # box_iou计算预测框与标签框的iou值，max(1)选出最大的ious值,i为对应的索引
                        
                        """
                        pred shape[N, 4]
                        tbox shape[M, 4]
                        box_iou shape[N, M]
                        ious shape[N, 1]
                        i shape[N, 1], i里的值属于0~M
                        """
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            # 获得检测到的目标
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                # 添加d到detected
                                detected.append(d)
                                # iouv为以0.05为步长，0.5到0.95的列表
                                # 获得不同iou阈值下的true positive
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            # 每张图片的结果统计到stats里
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        # 画出第1个batch的图片的ground truth和预测框并保存
        if plots and batch_i < 1:
            f = save_dir / f'test_batch{batch_i}_gt.jpg'  # filename
            plot_images(img, targets, paths, str(f), names)  # ground truth
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, str(f), names)  # predictions
    # Compute statistics
    # 将stats列表的信息拼接到一起
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        # 根据上面得到的TP等信息计算指标
        # 精准度=TP/TP+FP，召回率=TP/P，map，f1分数，类别ap
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # nt是一个列表，测试集每个类别有多少个目标框
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    # 打印指标结果
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # 细节展示每一个类别的指标
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    # 打印前向传播耗费的时间、nms的时间、总时间
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    # 采用之前保存的json格式预测结果，通过cocoapi评估指标
    # 需要注意的是 测试集的标签也需要转成coco的json格式
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        file = save_dir / f"detections_val2017_{w}_results.json"  # predicted annotations file
        print('\nCOCO mAP with pycocotools... saving %s...' % file)
        with open(file, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # 获取图片id
            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            # 获取并初始化测试集标签的json文件
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            # 初始化预测框的文件
            cocoDt = cocoGt.loadRes(str(file))  # initialize COCO pred api
            # 创建评估器
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            # 评估
            cocoEval.evaluate()
            cocoEval.accumulate()
            # 展示结果
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    # 返回测试指标结果
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    # 建立参数解析对象parser
    parser = argparse.ArgumentParser(prog='test.py')
    """
    opt参数详解
    ==========================================
    weights:测试的模型权重文件
    data:数据集配置文件，数据集路径，类名等
    batch-size:前向传播时的批次, 默认32
    img-size:输入图片分辨率大小, 默认640
    conf-thres:筛选框的时候的置信度阈值, 默认0.001
    iou-thres:进行NMS的时候的IOU阈值, 默认0.65
    save-json:是否按照coco的json格式保存预测框，并且使用cocoapi做评估(需要同样coco的json格式的标签), 默认False
    task:设置测试形式, 默认val, 具体可看下面代码解析注释
    device:测试的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
    single-cls:数据集是否只有一个类别，默认False
    augment:测试时是否使用TTA(Test Time Augmentation), 默认False
    verbose:是否打印出每个类别的mAP, 默认False
    save-txt:是否以txt文件的形式保存模型预测的框坐标, 默认False
    """
    # 添加属性：给xx实例增加一个aa属性，如 xx.add_argument("aa")
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='runs/test', help='directory to save results')
    # 采用parser对象的parse_args函数获取解析的参数
    opt = parser.parse_args()
    # 设置参数save_json
    opt.save_json |= opt.data.endswith('coco.yaml')
    # check_file检查文件是否存在
    opt.data = check_file(opt.data)  # check file
    print(opt)

    # task = ['val', 'test']时就正常测试验证集、测试集
    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_dir=Path(opt.save_dir),
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             )

        print('Results saved to %s' % opt.save_dir)
        
    # task == 'study'时，就评估yolov5系列和yolov3-spp各个模型在各个尺度下的指标并可视化
    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # utils.general.plot_study_txt(f, x)  # plot
