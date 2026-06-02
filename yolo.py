import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, resize_image1, show_config)
from utils.utils_bbox import DecodeBox
from utils.utils_rbox import *

from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from collections import deque
from PIL import Image
'''
训练自己的数据集必看注释！
'''

class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/yolov7_tiny_obb_uav.pth',
        "classes_path"      : 'model_data/uav_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
        #-------------------------------#
        #   是否使用DeepSort跟踪模块
        #-------------------------------#
        "track"             : True,
    }
    _defaults_3d = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/yolov10_3d.pth',
        "classes_path"      : 'model_data/kitti_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        #"anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [608, 608],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        #"nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
        #-------------------------------#
        #   是否使用DeepSort跟踪模块
        #-------------------------------#
        "track"             : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
        self.data_deque = {}    #deepsort更新数据
        self.count_num = ['',0]    
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        #show_config(**self._defaults)
        show_config(**self._defaults1)

        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                                 max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                                 nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False, trt=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
        if trt:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load('model_data/yolov7_tiny_trt.pth'))
            self.net  = model_trt
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = True):
        if self.track:
            image_font =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_font = Image.fromarray(np.uint8(image_font))
            
            #---------------------------------------------------#
            #   计算输入图片的高和宽
            #---------------------------------------------------#
            image_shape = np.array(np.shape(image_font)[0:2])
            #image_shape1 = image.shape[:2]
            #---------------------------------------------------------#
            #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
            #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
            #---------------------------------------------------------#
            image_font       = cvtColor(image_font)
            #image       = cvtColor(image)
            #---------------------------------------------------------#
            #   给图像增加灰条，实现不失真的resize
            #   也可以直接resize进行识别
            #---------------------------------------------------------#
            image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
            #---------------------------------------------------------#
            #   添加上batch_size维度
            #   h, w, 3 => 3, h, w => 1, 3, h, w
            #---------------------------------------------------------#
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
            

            with torch.no_grad():
                images = torch.from_numpy(image_data)
                if self.cuda:
                    images = images.cuda()
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                        
                if results[0] is None: 
                    return image
                

                top_label   = np.array(results[0][:, 7], dtype = 'int32')
                
                top_conf    = results[0][:, 5] * results[0][:, 6]
                
                top_rboxes  = results[0][:, :5]
                
                top_polys   = rbox2poly(top_rboxes)

                top_label1  = np.array(results[0][:, 6], dtype = 'int32')
                #print(f" top_conf={ top_conf}")
                #print(f" top_rboxes={ top_rboxes}")
                #print(f" top_label={ top_label}")
                #print(f" top_label1 = {top_label1}")
                #print(f"img = {image.shape}")
                #print(f"img_font = {image_font.size}")

            #---------------------------------------------------------#
            #   跟踪显示
            #---------------------------------------------------------#  
            outputs = self.deepsort.update(top_rboxes, top_conf, top_label1, image)

            print(f"outputs = {outputs}")
    
            #---------------------------------------------------------#
            #   设置字体与边框厚度
            #---------------------------------------------------------#
            font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image_font.size[1] + 0.5).astype('int32'))
            thickness   = (max((image_font.size[0] + image_font.size[1]) // np.mean(self.input_shape), 1)).astype('int32')
            #---------------------------------------------------------#
            #   计数
            #---------------------------------------------------------#
            if count:
                print("top_label:", top_label)
                classes_nums    = np.zeros([self.num_classes])
                for i in range(self.num_classes):
                    num = np.sum(top_label == i)
                    if num > 0 and classes_nums[i] < num:
                        print(self.class_names[i], " : ", num)
                        self.count_num[0] = self.class_names[i]
                        self.count_num[1] = num
                        classes_nums[i] = num
                print("classes_nums:", classes_nums)
            #---------------------------------------------------------#
            #   图像绘制
            #---------------------------------------------------------#
            #------------- 绘制跟踪轨迹--------------#
            if self.track and len(outputs) > 0:
                #if  len(outputs) > 0:
                height, width, _ = image.shape
                object_id = outputs[..., -1]
                identities = outputs[..., -2]
                bbox = outputs[..., :4]
                for key in list(self.data_deque):
                    if key not in identities:
                        self.data_deque.pop(key)
                # center = (int((left + right) / 2), int((top + bottom) / 2))
                for i, box in enumerate(bbox):
                    x1, y1, x2, y2 = box

                    # get ID of object
                    id = int(identities[i]) if identities is not None else 0

                    # create new buffer for new object
                    if id not in self.data_deque:
                        self.data_deque[id] = deque(maxlen=64)

                    class_id = object_id[i]
                    obj_name = self.class_names[class_id]
                    label = '%d:%s' % (id, obj_name)
                    #cv2.rectangle(image_font, (x1, y1), (x2, y2), self.colors[class_id], 1, cv2.LINE_AA)
                    #cv2.putText(image_font, label, (x1, y1 - 2), 1, 1, [0, 0, 0], 0, cv2.LINE_AA)

                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    # add center to buffer
                    self.data_deque[id].appendleft(center)
                    # draw trail
                    for i in range(1, len(self.data_deque[id])):
                        # check if on buffer value is none
                        if self.data_deque[id][i - 1] is None or self.data_deque[id][i] is None:
                            continue
                        draw = ImageDraw.Draw(image_font)
                        draw.line(xy=[self.data_deque[id][i - 1], self.data_deque[id][i]], fill=self.colors[class_id], width=3)
                        draw.text( np.array([x1, y1 - 2],np.int32), str(label), fill=self.colors[class_id], font=font)
                        del draw
            #------------- 绘制旋转框--------------#
            for i, c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                poly            = top_polys[i].astype(np.int32)
                score           = top_conf[i]

                polygon_list = list(poly)
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image_font)
                label = label.encode('utf-8')
                
                text_origin = np.array([poly[0], poly[1]], np.int32)

                draw.polygon(xy=polygon_list, outline=self.colors[c])
                if self.track == False: 
                    draw.text(text_origin, str(label,'UTF-8'), fill=self.colors[c], font=font)
                del draw

            return image_font   
        else :
            image_shape = np.array(np.shape(image)[0:2])
            image       = cvtColor(image)
            image_data  = resize_image1(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
            #---------------------------------------------------------#
            #   添加上batch_size维度
            #   h, w, 3 => 3, h, w => 1, 3, h, w
            #---------------------------------------------------------#
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
            

            with torch.no_grad():
                images = torch.from_numpy(image_data)
                if self.cuda:
                    images = images.cuda()
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                        
                if results[0] is None: 
                    return image
                
                top_label   = np.array(results[0][:, 7], dtype = 'int32')
                
                top_conf    = results[0][:, 5] * results[0][:, 6]
                
                top_rboxes  = results[0][:, :5]
                
                top_polys   = rbox2poly(top_rboxes)

                top_label1  = np.array(results[0][:, 6], dtype = 'int32')
    
            #---------------------------------------------------------#
            #   设置字体与边框厚度
            #---------------------------------------------------------#
            font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness   = (max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1)).astype('int32')

                
            #---------------------------------------------------------#
            #   计数
            #---------------------------------------------------------#
            if count:
                print("top_label:", top_label)
                classes_nums    = np.zeros([self.num_classes])
                for i in range(self.num_classes):
                    num = np.sum(top_label == i)
                    if num > 0 and self.count_num[1]< num:
                        print(self.class_names[i], " : ", num)
                        self.count_num[0] = self.class_names[i]
                        self.count_num[1] = num
                        classes_nums[i] = num
                print("classes_nums:", classes_nums)
        

            #---------------------------------------------------------#
            #   图像绘制
            #---------------------------------------------------------#
            #------------- 绘制跟踪轨迹--------------#
            
            if self.track == False: 
                for i, c in list(enumerate(top_label)):
                    predicted_class = self.class_names[int(c)]
                    poly            = top_polys[i].astype(np.int32)
                    score           = top_conf[i]

                    polygon_list = list(poly)
                    label = '{} {:.2f}'.format(predicted_class, score)
                    draw = ImageDraw.Draw(image)
                    label = label.encode('utf-8')
                    
                    text_origin = np.array([poly[0], poly[1]], np.int32)

                    draw.polygon(xy=polygon_list, outline=self.colors[c])
                    if self.track == False: 
                        draw.text(text_origin, str(label,'UTF-8'), fill=self.colors[c], font=font)
                    del draw

                return image 

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score      = np.max(sigmoid(sub_output[..., 4]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 7], dtype = 'int32')
            top_conf    = results[0][:, 5] * results[0][:, 6]
            top_rboxes  = results[0][:, :5]
            top_polys   = rbox2poly(top_rboxes)
            top_hbbs    = poly2hbb(top_polys)
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            hbb             = top_hbbs[i]
            score           = str(top_conf[i])

            xc, yc, w, h = hbb
            left   = xc - w/2
            top    = yc - h/2
            right  = xc + w/2
            bottom = yc + h/2
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
