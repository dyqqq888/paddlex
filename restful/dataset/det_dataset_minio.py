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

import os.path as osp
from ..utils import list_files
from .utils import is_pic, replace_ext, get_encoding, check_list_txt
from .datasetbase_minio import DatasetBase_minio
import xml.etree.ElementTree as ET
import os
import minio
import json
import numpy as np
import cv2

class DetDataset_minio(DatasetBase_minio):
    def __init__(self, dataset_id, path,endpoint,folder):
        super().__init__(dataset_id, path,endpoint,folder)

    def check_dataset(self, endpoint,folder,dataset_path,from_folder):

        minio_conf = {
            'endpoint': endpoint,
            'access_key': 'admin',
            'secret_key': '12345678',
            'secure': False
        }
        client = minio.Minio(**minio_conf)
        my_dir= '_json_jpg'
        image_dir='_json_jpg/images'
        json_dir='_json_jpg/labels'
        #xml_dir='_json_jpg/Annotation'
        client = minio.Minio(**minio_conf)
        if not os.path.exists(osp.join(dataset_path,'Annotations')):
            os.mkdir(osp.join(dataset_path,'Annotations'))
        if not os.path.exists(osp.join(dataset_path,'JPEGImages')):
            os.mkdir(osp.join(dataset_path,'JPEGImages'))
        
        #列出minio文件夹下所有obj
        objects_list=client.list_objects('datasets',prefix=from_folder+'/',recursive=True)
        #从minio中拿出来
        for obj in objects_list:
            file_name=osp.split(obj.object_name)[1]
            data=client.get_object('datasets',osp.join(from_folder,file_name))
            haha=os.path.splitext(file_name)[1]
            if os.path.splitext(file_name)[1]=='.jpg' :
                path=osp.join(image_dir,file_name)
                with open(path,'wb') as f_img:
                    for d in data.stream(32*1024):
                        f_img.write(d) 
            
            elif os.path.splitext(file_name)[1]=='.json':
                path=osp.join(json_dir,file_name)
                with open(path,'wb') as f_json:
                    for d in data.stream(32*1024):
                        f_json.write(d) 
        import xml.dom.minidom as minidom
        i = 0
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".jpg.labels.json")
            #json_file2 = osp.join(json_dir, img_name_part + ".png.labels.json")
            i += 1
            if not osp.exists(json_file) :
                
                os.remove(osp.join(image_dir, img_name))
                continue
            
            
            xml_doc = minidom.Document()
            root = xml_doc.createElement("annotation")
            xml_doc.appendChild(root)
            node_folder = xml_doc.createElement("folder")
            node_folder.appendChild(xml_doc.createTextNode("JPEGImages"))
            root.appendChild(node_folder)
            node_filename = xml_doc.createElement("filename")
            node_filename.appendChild(xml_doc.createTextNode(img_name))
            root.appendChild(node_filename)
            with open(json_file, mode="r", \
                              encoding=get_encoding(json_file)) as j:
                json_info = json.load(j)
                if 'imageHeight' in json_info and 'imageWidth' in json_info:
                    h = json_info["imageHeight"]
                    w = json_info["imageWidth"]
                else:
                    img_file = osp.join(image_dir, img_name)
                    im_data = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    h, w, c = im_data.shape
                node_size = xml_doc.createElement("size")
                node_width = xml_doc.createElement("width")
                node_width.appendChild(xml_doc.createTextNode(str(w)))
                node_size.appendChild(node_width)
                node_height = xml_doc.createElement("height")
                node_height.appendChild(xml_doc.createTextNode(str(h)))
                node_size.appendChild(node_height)
                node_depth = xml_doc.createElement("depth")
                node_depth.appendChild(xml_doc.createTextNode(str(3)))
                node_size.appendChild(node_depth)
                root.appendChild(node_size)
                for item in json_info["labels"]:
                    label = item['label']
                    if 'shape_type' in item:
                        if item["shape_type"] != "rectangle":
                            continue
                        (xmin, ymin), (xmax, ymax) = item["points"]
                        xmin, xmax = sorted([xmin, xmax])
                        ymin, ymax = sorted([ymin, ymax])
                    else:

                        bbox = item["value"][0]['boundingBoxes'][0]
                        for i in range(0, len(bbox),2):
                            bbox[i] = bbox[i]*w
                            bbox[i+1] = bbox[i+1] * h
                        bbox = np.array(bbox).reshape((4,2))
                        points_num = len(bbox)
                        x = [bbox[i][0] for i in range(points_num)]
                        y = [bbox[i][1] for i in range(points_num)]
                        xmin = min(x)
                        xmax = max(x)
                        ymin = min(y)
                        ymax = max(y)
                    label = item["label"]
                    node_obj = xml_doc.createElement("object")
                    node_name = xml_doc.createElement("name")
                    node_name.appendChild(xml_doc.createTextNode(label))
                    node_obj.appendChild(node_name)
                    node_diff = xml_doc.createElement("difficult")
                    node_diff.appendChild(xml_doc.createTextNode(str(0)))
                    node_obj.appendChild(node_diff)
                    node_box = xml_doc.createElement("bndbox")
                    node_xmin = xml_doc.createElement("xmin")
                    node_xmin.appendChild(xml_doc.createTextNode(str(xmin)))
                    node_box.appendChild(node_xmin)
                    node_ymin = xml_doc.createElement("ymin")
                    node_ymin.appendChild(xml_doc.createTextNode(str(ymin)))
                    node_box.appendChild(node_ymin)
                    node_xmax = xml_doc.createElement("xmax")
                    node_xmax.appendChild(xml_doc.createTextNode(str(xmax)))
                    node_box.appendChild(node_xmax)
                    node_ymax = xml_doc.createElement("ymax")
                    node_ymax.appendChild(xml_doc.createTextNode(str(ymax)))
                    node_box.appendChild(node_ymax)
                    node_obj.appendChild(node_box)
                    root.appendChild(node_obj)
            
            with open(osp.join(dataset_path,'Annotations',img_name_part.replace(' ','') + ".xml"), 'w') as fxml:
                xml_doc.writexml(
                    fxml,
                    indent='\t',
                    addindent='\t',
                    newl='\n',
                    encoding="utf-8")
            os.remove(osp.join( json_dir,img_name_part + ".jpg.labels.json"))
            #转化后的.xml放到minio里面
            client.fput_object('datasets',osp.join(folder,'JPEGImages',img_name),osp.join(image_dir,img_name),content_type='image/jpeg')
            os.remove(osp.join(image_dir,img_name))
            client.fput_object('datasets',folder+'/Annotations/'+img_name_part+'.xml',osp.join(dataset_path,'Annotations',img_name_part.replace(' ','') + ".xml"),content_type='text/xml')

        #将.jpg放到数据集路径下
        i_objects=client.list_objects('datasets',prefix=folder+'/JPEGImages/',recursive=True)
        for i_obj in i_objects:
            image_object=osp.split(i_obj.object_name)[1]
            i_data=client.get_object('datasets',osp.join(folder,'JPEGImages',image_object))
            with open(osp.join(dataset_path,'JPEGImages',image_object.replace(' ','')),'wb') as i_file:
                for i_d in i_data:
                    i_file.write(i_d)
                #datalist1.append(i_file.name)
        self.all_files = list_files(dataset_path)
        # self.all_files2 = list(datalist2)

        self.file_info = dict()
        self.label_info = dict()

        if osp.exists(osp.join(dataset_path, 'train_list.txt')):
            return self.check_splited_dataset(dataset_path)

        for f in self.all_files:
            if not is_pic(f):
                continue
            items = osp.split(f)
            if len(items) == 2 and items[0] == "JPEGImages":
                anno_name = replace_ext(items[1], "xml")
                full_anno_path = osp.join(
                    (osp.join(dataset_path, 'Annotations')), anno_name)
                if osp.exists(full_anno_path):
                    self.file_info[f] = osp.join('Annotations', anno_name)
            try:
                tree = ET.parse(full_anno_path)
            except:
                raise Exception("文件{}不是一个良构的xml文件".format(full_anno_path))
            objs = tree.findall('object')
            for i, obj in enumerate(objs):
                cname = obj.find('name').text
                if cname not in self.label_info:
                    self.label_info[cname] = list()
                if self.all_files[i] not in self.label_info[cname]:
                    self.label_info[cname].append(f)

        self.labels = sorted(self.label_info.keys())
        for label in self.labels:
            self.class_train_file_list[label] = list()
            self.class_val_file_list[label] = list()
            self.class_test_file_list[label] = list()
        # 将数据集分析信息dump到本地
        self.dump_statis_info()

    def check_splited_dataset(self, source_path):
        labels_txt = osp.join(source_path, "labels.txt")
        train_list_txt = osp.join(source_path, "train_list.txt")
        val_list_txt = osp.join(source_path, "val_list.txt")
        test_list_txt = osp.join(source_path, "test_list.txt")
        for txt_file in [labels_txt, train_list_txt, val_list_txt]:
            if not osp.exists(txt_file):
                raise Exception(
                    "已切分的数据集下应该包含labels.txt, train_list.txt, val_list.txt文件")
        check_list_txt([train_list_txt, val_list_txt, test_list_txt])

        self.labels = open(
            labels_txt, 'r',
            encoding=get_encoding(labels_txt)).read().strip().split('\n')

        for txt_file in [train_list_txt, val_list_txt, test_list_txt]:
            if not osp.exists(txt_file):
                continue
            with open(txt_file, "r") as f:
                for line in f:
                    items = line.strip().split()
                    img_file, xml_file = [items[0], items[1]]

                    if not osp.isfile(osp.join(source_path, xml_file)):
                        raise ValueError("数据目录{}中不存在标注文件{}".format(
                            osp.split(txt_file)[-1], xml_file))
                    if not osp.isfile(osp.join(source_path, img_file)):
                        raise ValueError("数据目录{}中不存在图片文件{}".format(
                            osp.split(txt_file)[-1], img_file))
                    if not xml_file.split('.')[-1] == 'xml':
                        raise ValueError("标注文件{}不是xml文件".format(xml_file))
                    img_file_name = osp.split(img_file)[-1]
                    if not is_pic(img_file_name) or img_file_name.startswith(
                            '.'):
                        raise ValueError("文件{}不是图片格式".format(img_file))

                    self.file_info[img_file] = xml_file

                    if txt_file == train_list_txt:
                        self.train_files.append(img_file)
                    elif txt_file == val_list_txt:
                        self.val_files.append(img_file)
                    elif txt_file == test_list_txt:
                        self.test_files.append(img_file)

                    try:
                        tree = ET.parse(osp.join(source_path, xml_file))
                    except:
                        raise Exception("文件{}不是一个良构的xml文件".format(xml_file))
                    objs = tree.findall('object')
                    for i, obj in enumerate(objs):
                        cname = obj.find('name').text
                        if cname in self.labels:
                            if cname not in self.label_info:
                                self.label_info[cname] = list()
                            if img_file not in self.label_info[cname]:
                                self.label_info[cname].append(img_file)
                                if txt_file == train_list_txt:
                                    if cname in self.class_train_file_list:
                                        self.class_train_file_list[
                                            cname].append(img_file)
                                    else:
                                        self.class_train_file_list[
                                            cname] = list()
                                        self.class_train_file_list[
                                            cname].append(img_file)
                                elif txt_file == val_list_txt:
                                    if cname in self.class_val_file_list:
                                        self.class_val_file_list[cname].append(
                                            img_file)
                                    else:
                                        self.class_val_file_list[cname] = list(
                                        )
                                        self.class_val_file_list[cname].append(
                                            img_file)
                                elif txt_file == test_list_txt:
                                    if cname in self.class_test_file_list:
                                        self.class_test_file_list[
                                            cname].append(img_file)
                                    else:
                                        self.class_test_file_list[
                                            cname] = list()
                                        self.class_test_file_list[
                                            cname].append(img_file)
                        else:
                            raise Exception("文件{}与labels.txt文件信息不对应".format(
                                xml_file))

        # 将数据集分析信息dump到本地
        self.dump_statis_info()

    def split(self, val_split, test_split):
        super().split(val_split, test_split)
        with open(
                osp.join(self.path, 'train_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in self.train_files:
                label = self.file_info[x]
                f.write('{} {}\n'.format(x, label))
        with open(
                osp.join(self.path, 'val_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in self.val_files:
                label = self.file_info[x]
                f.write('{} {}\n'.format(x, label))
        with open(
                osp.join(self.path, 'test_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in self.test_files:
                label = self.file_info[x]
                f.write('{} {}\n'.format(x, label))
        with open(
                osp.join(self.path, 'labels.txt'), mode='w',
                encoding='utf-8') as f:
            for l in self.labels:
                f.write('{}\n'.format(l))
        self.dump_statis_info()
