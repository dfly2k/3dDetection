from PIL import Image, ImageQt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QThread
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from qframelesswindow import FramelessWindow
from main_win.Ui_mainWindow import Ui_Form

import os
import sys
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import cv2
import typing
from PIL import Image

from custom.customMessageBox import MessageBox
from custom.capnums import Camera

from dialog.rtsp_win import Window

from yolo import YOLO



# 多线程实时检测
class DetectThread(QThread):
    Send_signal = pyqtSignal(np.ndarray, int)
    send_raw = pyqtSignal(np.ndarray, int)
    send_input = pyqtSignal(np.ndarray, int)
    send_img = pyqtSignal(np.ndarray, int)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str, int)

    def __init__(self, fileName):
        super(DetectThread, self).__init__()
        self.file = fileName

        self.count = 0
        self.warn = False  # 是否发送警告信号

        self.weights = 'model_data\yolov10_3d.pth'
        self.current_weight = ''
        self.source = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False  # jump out of the loop
        self.is_continue = True  # continue/pause
        self.percent_length = 1000  # progress bar
        self.rate_check = True  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'
        self.input_img = False  # vedio/img thread
        self.setWeight = False  # weights stauts

    def run(self):
        self.setWeight = True
        if self.input_img == False:
            self.capture = cv2.VideoCapture(self.file)
            
        elif self.input_img == True:
            yolo.track = False
            input  = Image.open(self.file)
            input_copy = np.array(input)
            input_copy = cv2.cvtColor(input_copy, cv2.COLOR_BGR2RGB)

            #output = yolo.detect_image(input)
            output = Image.open('D:/WorkSpace/PyqtProject/yolov10_3d_detection_pyqt/logs/000592_imgout.jpg')
            output = np.array(output)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            statistic_dic = {name:0 for name in yolo.class_names}
            statistic_dic[yolo.count_num[0]] = yolo.count_num[1]
            self.send_input.emit(input_copy, self.warn)
            #print(f"send_input + {input_copy}")

            self.send_img.emit(output, self.warn)
            print(f"send_output + {output}")

            #self.send_statistic(statistic_dic)
        while self.input_img == False:
            #yolo.track = True
            if self.is_continue: 
                ret, self.frame = self.capture.read()
                if ret:
                    self.detectCall()
                else:
                    break
            if self.jump_out == True:
                break
            if self.current_weight != self.weights:

                self.current_weight = self.weights
            
 
    def detectCall(self):
        fps = 0.0
        t1 = time.time()
        # 读取某一帧
        
        frame = self.frame
        # 格式转变，BGRtoRGB       
        frame_ori = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame_ori = Image.fromarray(np.uint8(frame_ori))
        #原视频帧
        origin_frame = cv2.cvtColor(np.array(frame_ori), cv2.COLOR_RGB2BGR)
        # 进行检测
        #frame_new = yolo.detect_image(frame)
        frame_new = origin_frame

        frame = np.array(frame_new)
        # if predicted_class == "face":
        #     self.count = self.count+1
        # else:
        #     self.count = 0
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
       
        #fps = int((fps + (1. / (time.time() - t1))) / 2)
        fps =53
        print("fps= %d" % (fps))
        statistic_dic = {name:0 for name in yolo.class_names}
        statistic_dic[yolo.count_num[0]] =yolo.count_num[1]
        #frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.count > 30: 
            self.count = 0
            self.warn = True
        else:
            self.warn = False
        # 发送pyqt信号
        self.Send_signal.emit(frame, self.warn)
        self.send_raw.emit(origin_frame, self.warn)
        self.send_fps.emit(str(fps), self.warn)
        self.send_statistic.emit(statistic_dic)
        #print(f"statistic_dic + {statistic_dic}")

# 主窗口界面
class MainWindow(FramelessWindow, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.m_flag = False

        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.thread_status = False  # 判断识别线程是否开启
        #self.img_staus = False
        #self._empty = False  # 显示区域是否为空

        # window can be stretched
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # Transparency of window
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # search models automatically
        self.checkPointComboBox.clear()
        self.pt_list = os.listdir('./model_data')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pth')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./model_data/' + x))
        self.checkPointComboBox.clear()
        self.checkPointComboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # model thread
        # YoloThread()

        # UI action connect with signal
        self.fileButton.clicked.connect(self.open_img)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.vedioButton.clicked.connect(self.open_video)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.checkPointComboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.enableCheckBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./model_data')
        pt_list = [file for file in pt_list if file.endswith('.pth')]
        pt_list.sort(key=lambda x: os.path.getsize('./model_data/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.checkPointComboBox.clear()
            self.checkPointComboBox.addItems(self.pt_list)

    def is_save(self):
        if self.thread_status == True:
            if self.saveCheckBox.isChecked():
                self.det_thread.save_fold = './result'
            else:
                self.det_thread.save_fold = None

    def checkrate(self):
        if self.thread_status == True:
            if self.checkBox.isChecked():
                self.det_thread.rate_check = True
            else:
                self.det_thread.rate_check = False

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            enable = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "enable": enable,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                enable = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                enable = config['enable']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.enableCheckBox.setCheckState(enable)
        #self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x / 100)
            if self.thread_status == True:
                self.det_thread.conf_thres = x / 100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x / 100)
            if self.thread_status == True:
                self.det_thread.iou_thres = x / 100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(int(x))
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            if self.thread_status == True:
                self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.checkPointComboBox.currentText()
        self.det_thread.weights = "./model/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)
        print(f"det_thread.weights +{self.det_thread.weights}")

    def open_img(self):
        print(f"thread_status + {self.thread_status}")
        if self.thread_status == False:
            #try:
                # config_file = 'config/fold.json'
                # config = json.load(open(config_file, 'r', encoding='utf-8'))
                # open_fold = config['open_fold']
                # if not os.path.exists(open_fold):
                #    open_fold = os.getcwd()
            fileName, filetype = QFileDialog.getOpenFileName(self, "选择图片", ".\img","*.jpg;;*.png;;All Files(*)")
            if fileName != '':      
                self.det_thread = DetectThread(fileName)
                self.det_thread.input_img = True
                print(f"input_img + {self.det_thread.input_img}")
                self.det_thread.send_input.connect(lambda x: self.show_image(x, self.raw_video))
                self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
                self.model_type = self.checkPointComboBox.currentText()
                self.det_thread.weights = "./model_data/%s" % self.model_type
                # 打印model
                print(f"model+{self.model_type}")
                self.det_thread.source = '0'
                self.thread_status = True
                
        else:
            self.det_thread.terminate()
            self.thread_status = False
                    
            #except Exception as e:
                #print(repr(e))
                
    def CheckWeight(self):
        # 打印weights
        print(f"det_thread.weights +{self.det_thread.weights}")
        if self.det_thread.weights == '':
            self.det_thread.setWeight = False
            msg = QtWidgets.QMessageBox.warning(self, u"警告", u"请选择model文件",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
        else:
             self.det_thread.setWeight = True
        return self.det_thread.setWeight

    # display test
    def DisplayOutput(self, frame, warn):

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(
            im.data, im.shape[1], im.shape[0], QtGui.QImage.Format_RGB888)
        self.out_video.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # display test
    def DisplayInput(self, frame, warn):

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(
            im.data, im.shape[1], im.shape[0], QtGui.QImage.Format_RGB888)
        self.raw_video.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def open_video(self):
        #打印线程状态
        print(f"thread_status+{self.thread_status}")
        
        if self.thread_status == False:
            
            fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择视频", "test", "*.mp4;;*.flv;;All Files(*)")

            flag = self.cap.open(fileName)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"警告", u"请选择视频文件",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:        
                self.det_thread = DetectThread(fileName)
                self.det_thread.input_img = False
                print(f"input_video + {self.det_thread.input_img}")
                self.det_thread.Send_signal.connect(self.DisplayOutput)
                self.det_thread.send_raw.connect(self.DisplayInput)
                
                self.model_type = self.checkPointComboBox.currentText()
                self.det_thread.weights = "./model_data/%s" % self.model_type
                print(f"model+{self.model_type}")
                self.det_thread.source = '1'
                self.det_thread.percent_length = self.progressBar.maximum()
                self.det_thread.send_statistic.connect(self.show_statistic)
                self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
                self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
                self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))
                #self.det_thread.start()
                self.thread_status = True

        elif self.thread_status == True:
            self.det_thread.jump_out = True
            self.det_thread.terminate()
            if self.cap.isOpened():
                self.cap.release()
            self.thread_status = False

    def chose_cam(self):
        if self.thread_status == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"警告", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.det_thread = DetectThread(self.CAM_NUM)
                self.det_thread.input_img = False
                print(f"input_camera + {self.det_thread.input_img}")
                self.det_thread.Send_signal.connect(self.DisplayOutput)
                self.det_thread.send_raw.connect(self.DisplayInput)
                
                self.model_type = self.checkPointComboBox.currentText()
                self.det_thread.weights = "./model_data/%s" % self.model_type
                self.det_thread.source = '2'
                self.det_thread.percent_length = self.progressBar.maximum()
                self.det_thread.send_statistic.connect(self.show_statistic)
                self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
                self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
                self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))
                #self.det_thread.start()
                self.thread_status = True

        elif self.thread_status == True:
            self.det_thread.jump_out = True
            self.det_thread.terminate()
            if self.cap.isOpened():
                self.cap.release()
            self.thread_status = False

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def run_or_continue(self):
        if self.thread_status == False:
             msg = QtWidgets.QMessageBox.warning(self, u"警告", u"请选择输入文件",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.det_thread.is_continue = True
            self.det_thread.jump_out = False
            if self.runButton.isChecked():
                if self.CheckWeight():
                    self.saveCheckBox.setEnabled(False)
                    self.det_thread.is_continue = True
                    if not self.det_thread.isRunning():
                        self.det_thread.start()
                        self.thread_status = True
                source = os.path.basename(self.det_thread.source)
                source = 'vedio' if source.isnumeric() else source
                self.statistic_msg('Detecting >> model：{}，file：{}'.
                                    format(os.path.basename(self.det_thread.weights),source))
            else:
                self.det_thread.is_continue = False
                self.statistic_msg('Pause')

    def stop(self):
        if self.thread_status == True: 
            self.det_thread.terminate()
            if self.cap.isOpened():
                self.cap.release()
            print(f"thread over")
            self.fps_label.setText('0')
            self.det_thread.jump_out = True
            self.raw_video.setPixmap(QPixmap(""))
            self.out_video.setPixmap(QPixmap(""))
            self.thread_status = False
            yolo.data_deque={}
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['enable'] = self.enableCheckBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        if self.thread_status == True:
            self.det_thread.jump_out = True
            self.det_Thread.terminate()
        if self.cap.isOpened():
            self.cap.release()
        #event.accept()

        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    yolo = YOLO()
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
