# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
from restful.dataset.cls_dataset import ClsDataset
import traceback
import os.path as osp
import multiprocessing as mp
#from .cls_dataset import ClsDataset
#from .det_dataset import DetDataset
from .det_dataset_minio import DetDataset_minio
from .det_dataset import DetDataset

from .seg_dataset import SegDataset
from .ins_seg_dataset import InsSegDataset
from ..utils import set_folder_status, get_folder_status, DatasetStatus, DownloadStatus, download, list_files
from .det_dataset_minio import DetDataset_minio
#from .det_dataset_minio import DetDataset_minio
dataset_url_list = [
    'https://bj.bcebos.com/paddlex/demos/vegetables_cls.tar.gz',
    'https://bj.bcebos.com/paddlex/demos/insect_det.tar.gz',
    'https://bj.bcebos.com/paddlex/demos/optic_disc_seg.tar.gz',
    'https://bj.bcebos.com/paddlex/demos/xiaoduxiong_ins_det.tar.gz',
    'https://bj.bcebos.com/paddlex/demos/remote_sensing_seg.tar.gz'
]
dataset_url_dict = {
    'classification':
    'https://bj.bcebos.com/paddlex/demos/vegetables_cls.tar.gz',
    'detection': 'https://bj.bcebos.com/paddlex/demos/insect_det.tar.gz',
    'segmentation':
    'https://bj.bcebos.com/paddlex/demos/optic_disc_seg.tar.gz',
    'instance_segmentation':
    'https://bj.bcebos.com/paddlex/demos/xiaoduxiong_ins_det.tar.gz'
}


def _check_and_copy(dataset,dataset_path,endpoint,folder,from_folder):
    #try:
    dataset.check_dataset(endpoint,folder,dataset_path,from_folder)
    
    set_folder_status(dataset_path, DatasetStatus.XSPLITED, os.getpid())

    
def import_dataset(dataset_id, dataset_type, dataset_path, endpoint,folder,from_folder):
    set_folder_status(dataset_path, DatasetStatus.XCHECKING)
    if dataset_type == 'classification':
        ds = ClsDataset(dataset_id, dataset_path)
    elif dataset_type == 'detection':
        ds = DetDataset_minio(dataset_id, dataset_path,endpoint,folder)
    elif dataset_type == 'segmentation':
        ds = SegDataset(dataset_id, dataset_path)
    elif dataset_type == 'instance_segmentation':
        ds = InsSegDataset(dataset_id, dataset_path)
    p = mp.Process(
        target=_check_and_copy, args=(ds,dataset_path,endpoint,folder,from_folder))
    p.start()
    return p


def _download_proc(url, target_path, dataset_type):
    # 下载数据集压缩包
    from paddlex.utils import decompress
    target_path = osp.join(target_path, dataset_type)
    fname = download(url, target_path)
    # 解压
    decompress(fname)
    set_folder_status(target_path, DownloadStatus.XDDECOMPRESSED)


def download_demo_dataset(prj_type, target_path):
    url = dataset_url_list[prj_type.value]
    dataset_type = prj_type.name
    p = mp.Process(
        target=_download_proc, args=(url, target_path, dataset_type))
    p.start()
    return p


def get_dataset_status(dataset_id, dataset_type, dataset_path):
    status, message = get_folder_status(dataset_path, True)
    if status is None:
        status = DatasetStatus.XEMPTY
    if status == DatasetStatus.XCOPYING:
        items = message.strip().split()  #
        pid = None
        if len(items) < 2:
            percent = 0.0
        else:
            pid = int(items[0])
            if int(items[1]) == 0:
                percent = 1.0
            else:
                copyed_files_num = len(list_files(dataset_path)) - 1
                percent = copyed_files_num * 1.0 / int(items[1])
        message = {'pid': pid, 'percent': percent}
    if status == DatasetStatus.XCOPYDONE or status == DatasetStatus.XSPLITED:
        if not osp.exists(osp.join(dataset_path, 'statis.pkl')):
            p = import_dataset(dataset_id, dataset_type, dataset_path,
                               dataset_path)
            status = DatasetStatus.XCHECKING
    return status, message

import time
def split_dataset(dataset_id, dataset_type, dataset_path, val_split,
                  test_split,endpoint,folder):
    n_tries = 15
    n_try = 0
    wait_sec = 15
    max_wait_sec = 60
    while n_try < n_tries:
       
        status, message = get_folder_status(dataset_path, True)
        if status != DatasetStatus.XCOPYDONE and status != DatasetStatus.XSPLITED:
            time.sleep(wait_sec)
            n_try += 1
            wait_sec = min(2*wait_sec, max_wait_sec)
            continue 
        break
    while n_try < n_tries:
       
        if not osp.exists(osp.join(dataset_path, 'statis.pkl')):
            raise Exception("The dataset needs to be verified again. Please refresh the dataset before segmentation. ")
        break

    if dataset_type == 'classification':
        ds = ClsDataset(dataset_id, dataset_path)
    elif dataset_type == 'detection':
        ds = DetDataset_minio(dataset_id, dataset_path,endpoint,folder)
    elif dataset_type == 'segmentation':
        ds = SegDataset(dataset_id, dataset_path)
    elif dataset_type == 'instance_segmentation':
        ds = InsSegDataset(dataset_id, dataset_path)

    ds.load_statis_info()
    ds.split(val_split, test_split)
    set_folder_status(dataset_path, DatasetStatus.XSPLITED)


def get_dataset_details(dataset_path):
    status, message = get_folder_status(dataset_path, True)
    if status == DatasetStatus.XCOPYDONE or status == DatasetStatus.XSPLITED:
        with open(osp.join(dataset_path, 'statis.pkl'), 'rb') as f:
            details = pickle.load(f)    #将文件中的数据解析为一个Python对象
        return details
    return None
