# -*- coding: utf-8 -*-
from __future__ import print_function

import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from PIL import Image
from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.pyplot as plt

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from .log import logger


class TestParams(object):
    # params based on your local env
    gpus = [0]  # default to use CPU mode

    # loading existing checkpoint
    ckpt = './models/ckpt_epoch_60.pth'     # path to the ckpt file

    testdata_dir = './testimg/3/'

class Tester(object):

    TestParams = TestParams

    def __init__(self, model, test_params):
        assert isinstance(test_params, TestParams)
        self.params = test_params

        # load model
        self.model = model
        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES, 1 GPU is enough
        if len(self.params.gpus) > 0:
            gpu_test = str(self.params.gpus[0])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_test
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpu_test))
            self.model = self.model.cuda()

        self.model.eval()

    def test_cm(self):
        test_imgs_total=64604
        y_true = [0 for i in range(test_imgs_total)]
        y_pred = [0 for i in range(test_imgs_total)]
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        testdata_dir0 = './testimg/0/'
        testdata_dir1 = './testimg/1/'
        testdata_dir2 = './testimg/2/'
        testdata_dir3 = './testimg/3/'
        testdata_dir4 = './testimg/4/'
        testdata_dir5 = './testimg/5/'
        testdata_dir6 = './testimg/6/'
        testdata_dir7 = './testimg/7/'
        testdata_dir8 = './testimg/8/'
        testdata_dir9 = './testimg/9/'
        img_number = 0

        img_list = os.listdir(testdata_dir0)
        class_number = 0
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir0, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            #print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori)/10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        img_list = os.listdir(testdata_dir1)
        class_number = 1
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir1, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            # print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori)/10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        img_list = os.listdir(testdata_dir2)
        class_number = 2
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir2, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            # print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori) / 10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        img_list = os.listdir(testdata_dir3)
        class_number = 3
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir3, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            # print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori) / 10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        img_list = os.listdir(testdata_dir4)
        class_number = 4
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir4, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            # print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori) / 10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        img_list = os.listdir(testdata_dir5)
        class_number = 5
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir5, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            # print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori) / 10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        img_list = os.listdir(testdata_dir6)
        class_number = 6
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir6, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            # print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori) / 10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        img_list = os.listdir(testdata_dir7)
        class_number = 7
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir7, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            # print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori) / 10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        img_list = os.listdir(testdata_dir8)
        class_number = 8
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir8, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            # print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori) / 10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        img_list = os.listdir(testdata_dir9)
        class_number = 9
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(testdata_dir9, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()
            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            print('Prediction1 number1: ' + str(prediction[0]))
            # print('ori_test=' + str(score.data[:, 0]))
            extract_numble_ori = filter(str.isdigit, str(prediction[0]))
            extract_numble = int(extract_numble_ori) / 10
            y_pred[img_number] = extract_numble
            y_true[img_number] = class_number
            img_number = img_number + 1

        #y_true =[1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 8, 8, 9, 10]
        #y_pred =[2, 1, 2, 4, 3, 6, 1, 3, 7, 7, 7, 7, 7, 7, 8, 10]
        normalize = True
        title = 'CITR ResNet-34 Classification Confusion matrix'
        cmap = plt.cm.Blues
        out_name = 'output.jpg'
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(out_name, dpi=200, bbox_inches='tight')
        plt.show()

    def test_line(self):
        #prob_list = np.zeros(15).reshape(-1, 15)
        test_imgs_total = 260  #测试图像总数  手动修改！！！
        prob_list = [0 for i in range(test_imgs_total)]
        pic_number = [0 for i in range(test_imgs_total)]
        prob_list1 = [0 for i in range(test_imgs_total)]
        prob_list2 = [0 for i in range(test_imgs_total)]
        prob_list3 = [0 for i in range(test_imgs_total)]
        prob_list4 = [0 for i in range(test_imgs_total)]
        prob_list5 = [0 for i in range(test_imgs_total)]
        prob_list6 = [0 for i in range(test_imgs_total)]
        prob_list7 = [0 for i in range(test_imgs_total)]
        prob_list8 = [0 for i in range(test_imgs_total)]
        prob_list9 = [0 for i in range(test_imgs_total)]
        PROB =[prob_list,prob_list1,prob_list2,prob_list3,prob_list4,prob_list5,prob_list6,prob_list7,prob_list8,prob_list9]
        yscore1 = 0.0
        #pic_number = np.zeros(15).reshape(-1, 15)
        img_list = os.listdir(self.params.testdata_dir)
        img_number = 0
        for img_name in img_list:
            print('Processing image: ' + img_name)
            img = Image.open(os.path.join(self.params.testdata_dir, img_name)).convert('RGB')
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()

            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            for class_number in range(0, 9):
                print('Prediction1 number1: ' + str(prediction[0]))
                print('Top score: ' + str(score))
                print('ori_test='+str(score.data[:,0]))

                extract_numble=filter(str.isdigit, str(score.data[:,class_number]))
                print('filter_test:' + str(extract_numble))
                str_extract_numble = str(extract_numble)
                if str_extract_numble == '10':
                    yscore1 = 1.0
                    print('default == 1.0')
                else:
                    first_numble= float(extract_numble[0:1])
                    next_four_numble=float(extract_numble[1:5])*0.0001
                    e_numble=float(pow(0.1,int(extract_numble[5:7])))
                    yscore1=(first_numble+next_four_numble)*e_numble
                    print('default != 1.0')
                print(yscore1)
                PROB[class_number][img_number]= yscore1
                pic_number[img_number] = int(img_number)
            img_number=img_number+1

        print(prob_list)
        print(pic_number)
        #plot
        #plt.plot(prob_list,label='prob_class_0')
        #plt.plot(pic_number,prob_list1,label='prob_class_1')
        #plt.plot(pic_number, prob_list2,  label='prob_class_2')
        plt.plot(prob_list3,  label='prob_class_3')
        #plt.plot(pic_number, prob_list4,  label='prob_class_4')
        #plt.plot(prob_list5,  label='prob_class_5')
        plt.plot(prob_list6,  label='prob_class_6')
        #plt.plot(pic_number, prob_list7,  label='prob_class_7')
        #plt.plot(pic_number, prob_list8,  label='prob_class_8')
        #plt.plot(pic_number, prob_list9,  label='prob_class_9')
        plt.legend()
        plt.xlabel('Class 3 test')  # 展示使用哪种分类测试图 （如 Clas 0 test 表示测试分类0测试集） 手动修改！！
        plt.ylabel('Prob of every prediction')
        plt.show()

    def test_ros(self):
        for i in range(0,10000):
            boxtotal = 2  # box number of every image -1
            cvimg = [0 for i in range(boxtotal)]
            cvimgori= cv2.imread('./testimg/test.jpeg',cv2.IMREAD_COLOR)
            rect_x1= [0 for i in range(boxtotal)]
            rect_y1 = [0 for i in range(boxtotal)]
            rect_x2 = [0 for i in range(boxtotal)]
            rect_y2 = [0 for i in range(boxtotal)]
            tracking = [0 for i in range(boxtotal)]

            rect_x1[0]=160
            rect_y1[0]=77
            rect_x2[0] = 215
            rect_y2[0] = 167

            rect_x1[1]=210
            rect_y1[1]=110
            rect_x2[1] = 385
            rect_y2[1] = 180

            tracking[0]=1
            tracking[1]=5

            #img = Image.open(os.path.join(self.params.testdata_dir, img_name)).convert('RGB')
            for j in range(0,boxtotal):
                #read box[i][j]
                cvimgpro= cvimgori.copy()
                cvimg[j]=cvimgpro[rect_y1[j]:rect_y2[j],rect_x1[j]:rect_x2[j]]
          #      cv2.imshow('cut',cvimg)
        #        cv2.waitKey(0)
                img = Image.fromarray(cvimg[j], mode="RGB")
                img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
                img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img_input = Variable(torch.unsqueeze(img, 0))
                if len(self.params.gpus) > 0:
                    img_input = img_input.cuda()

                output = self.model(img_input)
                score = F.softmax(output, dim=1)
                _, prediction = torch.max(score.data, dim=1)
                print('Tracking number:' + str(tracking[j]))
                print('prediction image number:' + str(i))
                print('Prediction box number: ' + str(j) +'result:'+ str(prediction[0]))
                extract_numble_ori = filter(str.isdigit, str(prediction[0]))
                extract_numble = int(extract_numble_ori) / 10
                print('Prediction box number: ' + str(j) +'result:'+ str(extract_numble))
                cv2.rectangle(cvimgori, (rect_x1[j], rect_y1[j]), (rect_x2[j], rect_y2[j]), (0, 255, 0), 2)

                text = 'TN= ' + str(tracking[j]) + 'PN= ' + str(extract_numble)
                cv2.putText(cvimgori, text, (rect_x1[j] + 20, rect_y1[j] + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.imshow('detection',cvimgori)
                cv2.waitKey(10)




    def _load_ckpt(self, ckpt):
        #self.model.load_state_dict(torch.load(ckpt))
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(ckpt).items()})
