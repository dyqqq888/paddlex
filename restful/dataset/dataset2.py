from paddlex_restful.restful.utils import (set_folder_status, get_folder_status, DatasetStatus,
                     TaskStatus, is_available, DownloadStatus,
                     PretrainedModelStatus, ProjectType)
from threading import Thread
import random
from paddlex_restful.restful.dataset.utils import copy_directory, get_label_count
import traceback
import shutil
import psutil
import pickle
import os
import os.path as osp
import time
import json
import base64
import cv2
from paddlex_restful.restful import workspace_pb2 as w
import requests
from ..utils import ShareData
SD = ShareData()
def init(dirname, logger):
    #初始化工作空间
    from ..workspace import init_workspace
    from ..system import get_system_info
    SD.workspace = w.Workspace(path=dirname)
    init_workspace(SD.workspace, dirname, logger)
    SD.workspace_dir = dirname
    get_system_info(SD.machine_info)

def get_dataset_status(data, workspace):
    """获取数据集当前状态

    Args:
        data为dict, key包括
        'did':数据集id
    """
    from .operate import get_dataset_status
    dataset_id = data['did']
    assert dataset_id in workspace.datasets, "数据集ID'{}'不存在.".format(dataset_id)
    dataset_type = workspace.datasets[dataset_id].type
    dataset_path = workspace.datasets[dataset_id].path
    dataset_name = workspace.datasets[dataset_id].name
    dataset_desc = workspace.datasets[dataset_id].desc
    dataset_create_time = workspace.datasets[dataset_id].create_time
    status, message = get_dataset_status(dataset_id, dataset_type,
                                         dataset_path)
    dataset_pids = list()
    for key in workspace.projects:
        if dataset_id == workspace.projects[key].did:
            dataset_pids.append(workspace.projects[key].id)

    attr = {
        "type": dataset_type,
        "id": dataset_id,
        "name": dataset_name,
        "path": dataset_path,
        "desc": dataset_desc,
        "create_time": dataset_create_time,
        "pids": dataset_pids
    }
    return {
        'status': 1,
        'id': dataset_id,
        'dataset_status': status.value,
        'message': message,
        'attr': attr
    }
def create_dataset(data, workspace,monitored_processes,load_demo_proc_dict):
    
    
    """
    1、创建dataset  create_dataset
    """
    modelName=data['modelName']
    create_time = time.time()
    time_array = time.localtime(create_time)   #格式化为本地时间
    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)  #
    id = workspace.max_dataset_id + 1
    if id < 10000:
        did = 'D%04d' % id
    else:
        did = 'D{}'.format(id)
    dname=data['from_folder']
    assert not did in workspace.datasets, "【数据集创建】ID'{}'已经被占用.".format(did)
    
    path = osp.join(workspace.path, 'datasets', did)
    if osp.exists(path):
        if not osp.isdir(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    os.makedirs(path)
    set_folder_status(path, DatasetStatus.XEMPTY)
    workspace.max_dataset_id = id
    ds = w.Dataset(
        id=did,
        name=data['from_folder'],
        desc="test",
        type='detection',
        create_time=create_time,
        path=path)
    workspace.datasets[did].CopyFrom(ds)
    #return {'status': 1, 'did': did}
    
    """
    2、导入 import_dataset
    """
    dataset_id = did
    assert dataset_id in workspace.datasets, "数据集ID'{}'不存在.".format(dataset_id)
    dataset_type = workspace.datasets[dataset_id].type
    dataset_path = workspace.datasets[dataset_id].path
    valid_dataset_type = [
        'classification', 'detection', 'segmentation', 'instance_segmentation',
        'remote_segmentation'
    ]
    assert dataset_type in valid_dataset_type, "无法识别的数据类型{}".format(
        dataset_type)

    from .operate import import_dataset
    #data['desc']代表endpoint
    #data['name']代表from_folder
    process = import_dataset(dataset_id, dataset_type, dataset_path,
                             data['minio_url'],data['folder'],data['from_folder'])
                             
    monitored_processes.put(process.pid)
    if 'demo' in data:
        prj_type = getattr(ProjectType, dataset_type)  #返回ProjectType对象的dataset_type属性值
        if prj_type not in load_demo_proc_dict:
            load_demo_proc_dict[prj_type] = []
        load_demo_proc_dict[prj_type].append(process)
    #return {'status': 1}

    """
    3、将数据集切分为训练集、验证集和测试集   split_dataset  img_base64
    """
    from .operate import split_dataset
    from .operate import get_dataset_details
    #dataset_id = did
    assert dataset_id in workspace.datasets, "数据集ID'{}'不存在.".format(dataset_id)
    dataset_type = workspace.datasets[dataset_id].type
    dataset_path = workspace.datasets[dataset_id].path
    val_split = 0.2
    test_split = 0.1
    split_dataset(dataset_id, dataset_type, dataset_path, val_split,
                  test_split, data['minio_url'],data['folder'])
    #from .dataset import img_base64
    #data_path={'path':osp.join(dataset_path,'JPEGImages'),'did':dataset_id}
    #img_base64(data_path,workspace)
    #return {'status': 1}
    """
    4、创建project create_project
    """
    create_time = time.time()
    time_array = time.localtime(create_time)
    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    id = workspace.max_project_id + 1
    workspace.max_project_id = id
    if id < 10000:
        pid = 'P%04d' % id
    else:
        pid = 'P{}'.format(id)
    assert not pid in workspace.projects, "【项目创建】ID'{}'已经被占用.".format(id)
    if 'path' not in data:
        path = osp.join(workspace.path, 'projects', pid)
    if not osp.exists(path):
        os.makedirs(path)
    pj = w.Project(
        id=pid,
        name=data['from_folder'],
        desc=data['from_folder'],
        type='detection',
        create_time=create_time,
        path=path)
    workspace.projects[pid].CopyFrom(pj)

    with open(os.path.join(path, 'info.pb'), 'wb') as f:
        f.write(pj.SerializeToString())

    """
    5、创建任务  create_task
    """
    from ..project.train.params import ClsParams, DetParams, SegParams
    create_time = time.time()
    time_array = time.localtime(create_time)
    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    id = workspace.max_task_id + 1
    workspace.max_task_id = id
    if id < 10000:
        tid = 'T%04d' % id
    else:
        tid = 'T{}'.format(id)
    #pid =p_id
    assert pid in workspace.projects, "【任务创建】项目ID'{}'不存在.".format(pid)
    assert not tid in workspace.tasks, "【任务创建】任务ID'{}'已经被占用.".format(id)
    #did = workspace.projects[pid].did
    #assert did in workspace.datasets, "【任务创建】数据集ID'{}'不存在".format(did)
    path = osp.join(workspace.projects[pid].path, tid)
    if not osp.exists(path):
        os.makedirs(path)
    set_folder_status(path, TaskStatus.XINIT)

    data['task_type'] = 'detection'
    #data['task_type'] = workspace.projects[pid].type
    data['dataset_path'] = workspace.datasets[did].path
    data['pretrain_weights_download_save_dir'] = osp.join(workspace.path,
                                                          'pretrain')
    #参数
    data['train']='{"cuda_visible_devices": "0", "batch_size": 1, "save_interval_epochs": 1, "pretrain_weights": "COCO", "model": "FasterRCNN", "num_epochs": 1, "learning_rate": 0.00125, "lr_decay_epochs": [8, 11], "train_num": 21, "resume_checkpoint": null, "sensitivities_path": null, "eval_metric_loss": null, "image_shape": [800, 1333], "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225], "horizontal_flip_prob": 0.5, "brightness_range": 0.5, "brightness_prob": 0.0, "contrast_range": 0.5, "contrast_prob": 0.0, "saturation_range": 0.5, "saturation_prob": 0.0, "hue_range": 18.0, "hue_prob": 0.0, "horizontal_flip": true, "brightness": true, "contrast": true, "saturation": true, "hue": true, "warmup_steps": 50, "warmup_start_lr": 0.00083333, "use_mixup": true, "mixup_alpha": 1.5, "mixup_beta": 1.5, "expand_prob": 0.5, "expand_image": true, "crop_image": true, "backbone": "HRNet_W18", "with_fpn": true, "random_shape": true, "random_shape_sizes": [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]}'
    params_json = json.loads(data['train'])
        #if (data['task_type'] == 'classification'):
            #params_init = ClsParams()
        #if (data['task_type'] == 'detection' or
                #data['task_type'] == 'instance_segmentation'):
    params_init = DetParams()
        #if (data['task_type'] == 'segmentation' or
                #data['task_type'] == 'remote_segmentation'):
            #params_init = SegParams()
    params_init.load_from_dict(params_json)
    data['train'] = params_init
    parent_id = ''
    """
    if 'parent_id' in data:
        data['tid'] = data['parent_id']
        parent_id = data['parent_id']
        assert data['parent_id'] in workspace.tasks, "【任务创建】裁剪任务创建失败".format(
            data['parent_id'])
        r = get_task_params(data, workspace)
        train_params = r['train']
        data['train'] = train_params
    """
    desc = ""
    if 'desc' in data:
        desc = data['desc']
    with open(osp.join(path, 'params.pkl'), 'wb') as f:
        pickle.dump(data, f)
    task = w.Task(
        id=tid,
        pid=pid,
        path=path,
        create_time=create_time,
        parent_id=parent_id,
        desc=desc)
    workspace.tasks[tid].CopyFrom(task)

    with open(os.path.join(path, 'info.pb'), 'wb') as f:
        f.write(task.SerializeToString())

    """
    启动训练任务
    """
    from ..project.operate import train_model
    #tid = tid
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    
    if 'eval_metric_loss' in data and \
        data['eval_metric_loss'] is not None:
        # 裁剪任务
        parent_id = workspace.tasks[tid].parent_id
        assert parent_id != "", "任务{}不是裁剪训练任务".format(tid)
        parent_path = workspace.tasks[parent_id].path
        sensitivities_path = osp.join(parent_path, 'prune',
                                      'sensitivities.data')
        eval_metric_loss = data['eval_metric_loss']
        parent_best_model_path = osp.join(parent_path, 'output', 'best_model')
        params_conf_file = osp.join(path, 'params.pkl')
        with open(params_conf_file, 'rb') as f:
            params = pickle.load(f)
        params['train'].sensitivities_path = sensitivities_path
        params['train'].eval_metric_loss = eval_metric_loss
        params['train'].pretrain_weights = parent_best_model_path
        with open(params_conf_file, 'wb') as f:
            pickle.dump(params, f)
    
    p = train_model(path)
    monitored_processes.put(p.pid)
    n_tries_t = 30
    n_try_t = 0
    wait_sec_t = 60
    train_path =osp.join(path,'XTRAINDONE')   
    while n_try_t<n_tries_t:

        if osp.exists(train_path):
            """
            评估
            """
            data={'tid': tid}
            url="http://127.0.0.1:8066/project/task/evaluate"
            r=requests.post(url,params= data)
            print(r)
            if r.status_code==200:
                
                ret=requests.get(url,params= data)
                if ret.status_code==200:
                    res = json.loads(ret.text)
            break       
        else:
            time.sleep(wait_sec_t)
            n_try_t+=1
            continue
        

    """导出部署模型

    Args:
        data为dict，key包括
        'tid'任务id, 'save_dir'导出模型保存路径
    """ 
    from ..project.operate import export_noquant_model, export_quant_model
    #tid = data['tid']
    #save_dir = data['save_dir']
    save_dir=osp.join('Workspace/projects',pid,tid,'export_model')
    #epoch = data['epoch'] if 'epoch' in data else None
    epoch= None
    #quant = data['quant'] if 'quant' in data else False
    quant = False
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    if quant:
        p = export_quant_model(path, save_dir, epoch)
    else:
        p = export_noquant_model(path, save_dir, epoch)
    monitored_processes.put(p.pid)
    n_tries_m = 10
    n_try_m = 0
    wait_sec_m = 5 
    while n_try_m<n_tries_m:
        from ..project.operate import get_export_status
        task_path = workspace.tasks[tid].path
        status, message = get_export_status(task_path)
        if status == TaskStatus.XEXPORTED:
            #'name': tid+'_export_model'                                              
            data={'pid': pid, 'tid': tid,'name': modelName, 'type': 'exported', 'source_path': '', 'path': save_dir, 'exported_type': 0, 'eval_results': {}}
            url="http://127.0.0.1:8066/model"
            r=requests.post(url,params= data)
            rr=json.loads(r.text)
            return {'status': 1, 'emid': rr['emid'],'createdDateTime':rr['createdDateTime'],'modelName':rr['modelName'],'averageAccuracy':res['averageAccuracy'],'result':res['result']}
        else:
            time.sleep(wait_sec_m)
            n_try_m+=1
            continue
       

    