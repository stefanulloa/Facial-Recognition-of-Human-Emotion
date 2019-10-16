'''
Created on Oct 17, 2018

@author: deckyal
'''

from math import sqrt
import re

from PIL import Image,ImageFilter

import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import csv
import torchvision.transforms.functional as F
import numbers


from skimage.util import random_noise

import file_walker
from utils import *
from config import *
from ImageAugment import *
import utils
from os.path import isfile

#noiseParamList = np.asarray([[0,0,0],[1,2,3],[1,3,5],[.001,.005,.01],[.8,.5,.2],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
noiseParamList =np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]

#noiseParamListTrain = np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
noiseParamListTrain = np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]

def addGaussianNoise(img,noiseLevel = 1):
    noise = torch.randn(img.size()) * noiseLevel
    noisy_img = img + noise
    return noisy_img



class AFEWVA(data.Dataset):

    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt

    def __init__(self, data_list = ["AFEW"],dir_gt = None,onlyFace = True, image_size =224,
                 transform = None,useIT = False,augment = False, step = 1,split = False,
                 nSplit = 5, listSplit = [0,1,2,3,4],wHeatmap= False,isVideo = False, seqLength = None):

        self.seq_length = seqLength
        self.isVideo = isVideo

        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment
        self.wHeatmap = wHeatmap

        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = "/homedtic/gulloa/ValenceArousal/images/"

        list_gt = []
        list_labels_t = []
        list_labels_tE = []

        counter_image = 0
        annotL_name = 'annot'
        annotE_name = 'annot2'

        if dir_gt is not None :
            annot_name = dir_gt

        for data in data_list :
            print(("Opening "+data))
            for f in file_walker.walk(self.curDir +data+"/"):
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    for sub_f in f.walk():
                        if sub_f.isDirectory: # Check if object is directory
                            #AFEWVA
                            for sub_2_f in sub_f.walk():
                                if sub_2_f.isDirectory:
                                    list_dta = []
                                    #print(sub_f.name)
                                    if(sub_2_f.name == annotL_name) : #If that's annot, add to labels_t

                                        for sub_sub_f in sub_2_f.walk(): #this is the data
                                            if(".npy" not in sub_sub_f.full_path):
                                                list_dta.append(sub_sub_f.full_path)
                                        list_labels_t.append(sorted(list_dta))

                                    elif(sub_2_f.name == annotE_name) : #If that's annot, add to labels_t

                                        for sub_sub_f in sub_2_f.walk(): #this is the data
                                            if(".npy" not in sub_sub_f.full_path):
                                                list_dta.append(sub_sub_f.full_path)
                                        list_labels_tE.append(sorted(list_dta))

                                    elif(sub_2_f.name == 'img'): #Else it is the image
                                        for sub_sub_f in sub_2_f.walk(): #this is the data
                                            if(".npy" not in sub_sub_f.full_path):
                                                list_dta.append(sub_sub_f.full_path)
                                        list_gt.append(sorted(list_dta))
                                        counter_image+=len(list_dta)

                            #SEMAINE
                            '''list_dta = []

                            #print(sub_f.name)
                            if(sub_f.name == annotL_name) : #If that's annot, add to labels_t

                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_labels_t.append(sorted(list_dta))

                            elif(sub_f.name == annotE_name) : #If that's annot, add to labels_t

                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_labels_tE.append(sorted(list_dta))

                            elif(sub_f.name == 'img'): #Else it is the image

                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)'''

        self.length = counter_image
        print("Now opening keylabels")



        list_labelsN = []
        list_labelsEN = []

        list_labels = []
        list_labelsE = []

        for ix in range(len(list_labels_t)) : #lbl,lble in (list_labels_t,list_labels_tE) :
            lbl_68 = [] #Per folder
            lbl_2 = [] #Per folder

            lbl_n68 = [] #Per folder
            lbl_n2 = [] #Per folder
            for jx in range(len (list_labels_t[ix])): #lbl_sub in lbl :

                #print(os.path.basename(list_gt[ix][jx]))
                #print(os.path.basename(list_labels_t[ix][jx]))
                #print(os.path.basename(list_labels_tE[ix][jx]))

                lbl_sub = list_labels_t[ix][jx]
                if ('pts' in lbl_sub) :
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
                    lbl_n68.append(lbl_sub)

                lbl_subE = list_labels_tE[ix][jx]
                if ('aro' in lbl_subE) :
                    x = []
                    #print(lbl_sub)
                    with open(lbl_subE) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_2.append(np.array(x).flatten('F'))
                    lbl_n2.append(lbl_sub)


            list_labelsN.append(lbl_n68)
            list_labelsEN.append(lbl_n2)

            list_labels.append(lbl_68)
            list_labelsE.append(lbl_2)


        '''for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) :
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)

        list_labelsE = []
        for lbl in list_labels_tE :
            lbl_2 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('aro' in lbl_sub) :
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_2.append(np.array(x).flatten('F')) #1 record
            list_labelsE.append(lbl_2)'''



        '''print(len(list_labelsN[0]))
        print(len(list_labelsEN[0]))
        print(len(list_labels[0]))
        print(len(list_labelsE[0]))'''


        t_l_imgs = []
        t_l_gt = []
        t_l_gtE = []

        t_list_gt_names = []
        t_list_gtE_names = []



        if not self.isVideo :
            #Flatten it to one list
            for i in range(0,len(list_gt)): #For each dataset

                list_images = []
                list_gt_names = []
                list_gtE_names = []
                indexer = 0

                list_ground_truth = np.zeros([len(list_gt[i]),136])
                list_ground_truthE = np.zeros([len(list_gt[i]),2])

                for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                    list_images.append(list_gt[i][j])

                    list_gt_names.append(list_labelsN[i][j])
                    list_gtE_names.append(list_labelsEN[i][j])

                    list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                    list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                    indexer += 1

                t_l_imgs.append(list_images)
                t_l_gt.append(list_ground_truth)
                t_l_gtE.append(list_ground_truthE)

                t_list_gt_names.append(list_gt_names)
                t_list_gtE_names.append(list_gtE_names)

        else :
            if self.seq_length is None :
                list_ground_truth = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,136])
                indexer = 0;

                for i in range(0,len(list_gt)): #For each dataset
                    counter = 0
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize

                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        i_temp = 0

                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z]
                            i_temp+=1

                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2

                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                self.l_imgs = list_images
                self.l_gt = list_ground_truth
            else :
                counter_seq = 0;

                for i in range(0,len(list_gt)): #For each dataset

                    indexer = 0;
                    list_gt_names = []
                    list_gtE_names = []

                    list_ground_truth = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,136]) #np.zeros([counter_image,136])
                    list_ground_truthE = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,2])#np.zeros([counter_image,2])

                    counter = 0
                    list_images = []

                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        tmpn68 = []
                        tmpn2 = []

                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        temp3 = np.zeros([self.seq_length,2])
                        i_temp = 0

                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z].flatten('F')
                            temp3[i_temp] = list_labelsE[i][z].flatten('F')

                            tmpn68.append(list_labelsN[i][z])
                            tmpn2.append(list_labelsEN[i][z])

                            i_temp+=1
                            counter_seq+=1

                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                        list_ground_truthE[indexer] = temp3

                        list_gt_names.append(tmpn68)
                        list_gtE_names.append(tmpn2)

                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter

                    t_l_imgs.append(list_images)
                    t_l_gt.append(list_ground_truth)
                    t_l_gtE.append(list_ground_truthE)

                    t_list_gt_names.append(list_gt_names)
                    t_list_gtE_names.append(list_gtE_names)

        '''print('length : ',len(t_l_imgs)) #Folder
        print('lengt2 : ',len(t_l_imgs[0])) #all/seq
        print('lengt3 : ',len(t_l_imgs[0][0])) #seq
        print('length4 : ',len(t_l_imgs[0][0][0]))'''

        #[folder, all/seq,seq]

        self.l_imgs = []
        self.l_gt = []
        self.l_gtE = []

        self.list_gt_names = []
        self.list_gtE_names = []

        #print('cimage : ',counter_image)


        if split :
            '''print('splitting')
            self.l_imgs = []
            self.l_gt = []
            self.l_gtE = []

            totalData = len(list_images)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x)
                begin = x*perSplit
                if x == nSplit-1 :
                    end = begin + (totalData - begin)
                else :
                    end = begin+perSplit
                print(begin,end,totalData)
                for x2 in range(begin,end) :
                    #print(x2,totalData)
                    self.l_imgs.append(list_images[x2])
                    self.l_gt.append(list_ground_truth[x2])
                    self.l_gtE.append(list_ground_truthE[x2])'''
            indexer = 0

            self.l_gt = []
            self.l_gtE = []
            '''else :
                self.l_gt = np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])'''

            totalData = len(t_l_imgs)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x)
                begin = x*perSplit
                if x == nSplit-1 :
                    end = begin + (totalData - begin)
                else :
                    end = begin+perSplit
                print(begin,end,totalData)

                if not self.isVideo :
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])):
                            #print('append ',t_l_imgs[i][j])
                            self.l_imgs.append(t_l_imgs[i][j])
                            self.l_gt.append(t_l_gt[i][j])
                            self.l_gtE.append(t_l_gtE[i][j])

                            self.list_gt_names.append(t_list_gt_names[i][j])
                            self.list_gtE_names.append(t_list_gtE_names[i][j])
                            indexer+=1

                else :
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): #seq counter

                            t_img = []
                            t_gt = []
                            t_gtE = []
                            t_gt_N = []
                            t_gt_EN = []
                            tmp = 0

                            for k in range(len(t_l_imgs[i][j])): #seq size
                                t_img.append(t_l_imgs[i][j][k])
                                t_gt.append(t_l_gt[i][j][k])
                                t_gtE.append(t_l_gtE[i][j][k])

                                t_gt_N.append(t_list_gt_names[i][j][k])
                                t_gt_EN.append(t_list_gtE_names[i][j][k])
                                tmp+=1

                            #print('append ',t_img)
                            self.l_imgs.append(t_img)
                            self.l_gt.append(t_gt)
                            self.l_gtE.append(t_gtE)

                            self.list_gt_names.append(t_gt_N)
                            self.list_gtE_names.append(t_gt_EN)
                            indexer+=1

                print(len(self.l_imgs))

            self.l_gt = np.asarray(self.l_gt)
            self.l_gtE = np.asarray(self.l_gtE)
        else :
            if not self.isVideo :
                self.l_gt = np.zeros([counter_image,136])
                self.l_gtE = np.zeros([counter_image,2])
                indexer = 0


                for i in range(len(t_l_imgs)):
                    for j in range(len(t_l_imgs[i])):
                        self.l_imgs.append(t_l_imgs[i][j])
                        print(i,j,'-',len(t_l_imgs[i]))
                        self.l_gt[indexer] = t_l_gt[i][j]
                        self.l_gtE[indexer] = t_l_gtE[i][j]

                        self.list_gt_names.append(t_list_gt_names[i][j])
                        self.list_gtE_names.append(t_list_gtE_names[i][j])
                        indexer+=1

            else :
                self.l_gt= np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])

                indexer = 0

                for i in range(len(t_l_imgs)): #dataset
                    for j in range(len(t_l_imgs[i])): #seq counter

                        t_img = []

                        t_gt = np.zeros([self.seq_length,136])
                        t_gte = np.zeros([self.seq_length,2])

                        t_gt_n = []
                        t_gt_en = []
                        i_t = 0

                        for k in range(len(t_l_imgs[i][j])): #seq size

                            t_img.append(t_l_imgs[i][j][k])
                            t_gt[i_t] = t_l_gt[i][j][k]
                            t_gte[i_t] = t_l_gtE[i][j][k]

                            t_gt_n.append(t_list_gt_names[i][j][k])
                            t_gt_en.append(t_list_gtE_names[i][j][k])

                            i_t+=1

                        self.l_imgs.append(t_img)
                        self.l_gt[indexer] = t_gt
                        self.l_gtE[indexer] = t_gte

                        self.list_gt_names.append(t_gt_n)
                        self.list_gtE_names.append(t_gt_en)

                        indexer+=1

        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height]
        #Im image : [batch_size, channel, width, height]

        l_imgs = []; l_VA = []; l_ldmrk = []; l_nc = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]

        if not self.isVideo :
            x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_l = [self.l_gt[index].copy()];label_n =[self.list_gt_names[index]]
        else :
            x_l = self.l_imgs[index];labelE_l =self.l_gtE[index].copy();label_l = self.l_gt[index].copy();label_n =self.list_gt_names[index]

        for x,labelE,label,ln in zip(x_l,labelE_l,label_l,label_n) :
            #print(x,labelE,label,ln)
            tImage = Image.open(x).convert("RGB")
            tImageB = None

            if self.onlyFace :
                #crop the face region
                #t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),displacementxy = random.uniform(-.5,.5))
                t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label.copy(),div_x = 8,div_y = 8,images = cv2.imread(x))#,displacementxy = random.uniform(-.5,.5))

                area = (x1,y1, x2,y2)
                tImage =  tImage.crop(area)

                label[:68] -= x_min
                label[68:] -= y_min

                tImage = tImage.resize((self.imageWidth,self.imageWidth))

                label[:68] *= truediv(self.imageWidth,(x2 - x1))
                label[68:] *= truediv(self.imageHeight,(y2 - y1))

            newChannel = None

            if self.wHeatmap :
                theMiddleName = 'img'
                filePath = x.split(os.sep)
                ifolder = filePath.index(theMiddleName)

                print(ifolder)
                image_name = filePath[-1]

                annot_name_H = os.path.splitext(image_name)[0]+'.npy'

                sDirName = filePath[:ifolder]
                dHeatmaps = '/'.join(sDirName)+'/heatmaps'

                finalTargetH = dHeatmaps+'/'+annot_name_H
                print(finalTargetH)

                if isfile(finalTargetH) and False:
                    newChannel  = np.load(finalTargetH)
                    newChannel = Image.fromarray(newChannel)
                else :
                    checkDirMake(dHeatmaps)

                    tImageTemp = cv2.cvtColor(np.array(tImage),cv2.COLOR_RGB2BGR)
                    #tImageTemp = cv2.imread(x)#tImage.copy()

                    b_channel,g_channel,r_channel = tImageTemp[:,:,0],tImageTemp[:,:,1],tImageTemp[:,:,2]
                    newChannel = b_channel.copy(); newChannel[:] = 0

                    t0,t1,t2,t3 = utils.get_bb(label[0:68], label[68:])

                    l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.05)
                    height, width,_ = tImageTemp.shape

                    wx = t2-t0
                    wy = t3-t1

                    scaler = 255/np.max(rv)

                    for iter in range(68) :
                        ix,iy = int(label[iter]),int(label[iter+68])

                        #Now drawing given the center
                        for iter2 in range(len(l_cd)) :
                            value = int(rv[iter2]*scaler)
                            if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value :
                                newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)

                    '''tImage2 = cv2.merge((b_channel, newChannel,newChannel, newChannel))
                    cv2.imshow("combined",tImage2)
                    cv2.waitKey(0)'''

                    np.save(finalTargetH,newChannel)
                    newChannel = Image.fromarray(newChannel)

            if self.augment :
                sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 :
                    pass
                elif sel == 1 :
                    flip = RandomHorizontalFlip_WL(1)
                    tImage,label,newChannel = flip(tImage,label,newChannel)
                elif sel == 2 :
                    rot = RandomRotation_WL(45)
                    tImage,label,newChannel = rot(tImage,label,newChannel)
                elif sel == 3 :
                    occ = Occlusion_WL(1)
                    tImage,label,newChannel = occ(tImage,label,newChannel)

                #random crop
                if True :
                    rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    tImage,label,newChannel= rc(tImage,label,newChannel)

                #additional blurring
                if (np.random.randint(1,3)%2==0) and True :
                    sel_n = np.random.randint(1,6)
                    #sel_n = 4
                    rc = GeneralNoise_WL(1)
                    tImage,label= rc(tImage,label,sel_n,np.random.randint(0,3))

            if self.useIT :
                tImage = self.transformInternal(tImage)
            else :
                tImage = self.transform(tImage)

            if not self.wHeatmap :
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]
            else :
                newChannel = transforms.Resize(224)(newChannel)
                newChannel = transforms.ToTensor()(newChannel)
                newChannel = newChannel.sub(125)
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label)); l_nc.append(newChannel)
                #return tImage,torch.FloatTensor(labelE),torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]

        if not self.isVideo :
            if self.wHeatmap :
                return l_imgs[0], l_VA[0], l_ldmrk[0], l_nc[0]
            else :
                return l_imgs[0], l_VA[0], l_ldmrk[0]
        else :
            #lImgs = torch.Tensor(len(l_imgs),3,self.imageHeight,self.imageWidth)
            #lVA = torch.Tensor(len(l_VA),2)
            #lLD = torch.Tensor(len(l_ldmrk),136)
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)

            #print(lImgs.shape, l_imgs[0].shape, l_VA[0].shape,len(lImgs))

            #torch.cat(l_imgs, out=lImgs)
            #torch.cat(l_VA, out=lVA)
            #torch.cat(l_ldmrk, out=lLD)

            if self.wHeatmap :
                #lnc = torch.Tensor(len(l_nc),1,self.imageHeight,self.imageWidth)
                #torch.cat(l_nc, out=lnc)
                lnc = torch.stack(l_nc)

                return lImgs, lVA, lLD, lnc
            else :
                return lImgs, lVA, lLD


    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img


    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl

    def __len__(self):
        return len(self.l_imgs)





def convertName(input):
    number = int(re.search(r'\d+', input).group())
    if 'train' in input :
        return number
    elif 'dev' in input :
        return 10+number
    elif  'test' in input :
        return 20+number

def main():



    image_size = 224
    batch_size = 1

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])

    #AFEW-VA-PP

    isVideo = False


    #data = AFEWVA(["temp"], None, True, image_size, transform, True, False, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)
    #data = AFEWVA(["temp"], None, True, image_size, transform, True, True, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)#,isVideo=True, seqLength = 7)
    data = AFEWVA(["AFEW-VA-Fixed"], None, True, image_size, transform, True, False, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True,isVideo=isVideo, seqLength = 6)

    #AFEW-VA-Fixed #SEM-temp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    print(len(dataloader))
    # Plot some training images
    for real_batch,va,gt,htmp in (dataloader) :

        print(htmp[0].shape)
        pass

        '''plt.figure(figsize=(batch_size,batch_size))
        plt.axis("off")
        plt.title("Trainings Images")
        plt.imshow(np.transpose(vutils.make_grid(htmp[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()'''

        print(real_batch.shape, va.shape, gt.shape,htmp.shape)

        if isVideo :
            rb = real_batch[0]
            rgt = gt[0]
            rva = va[0]
        else :
            rb = real_batch
            rgt = gt
            rva = va

        img = unnormalizedAndLandmark(rb, inputPred = rgt,inputGT = None, customNormalize = np.array([91.4953, 103.8827, 131.0912]))
        #print(va,x,lbl)
        for ig,vva in zip(img,rva) :
            print(vva)
            cv2.imshow('t',ig)
            cv2.waitKey(0)



    image_size = 224
    batch_size = 1

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])

    '''data = AVChallenge(["AVC-Train"], dir_gt = None, onlyFace = True, image_size = image_size,
                       transform = transform, useIT = True, augment = False,isTest = False,wHeatmap=True,
                       split=True,listSplit=[0,1,2,3])#,isVideo=True, seqLength = 7)'''

    '''data = AVChallenge(["AVC-Test"], dir_gt = None, onlyFace = True, image_size = image_size,
                       transform = transform, useIT = True, augment = False,isTest = True,wHeatmap=True)#,isVideo=True, seqLength = 7)'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    print(len(dataloader))
    # Plot some training images
    for real_batch,va,gt,htmp in (dataloader) :

        '''print(htmp[0].shape)
        pass

        plt.figure(figsize=(batch_size,batch_size))
        plt.axis("off")
        plt.title("Trainings Images")
        plt.imshow(np.transpose(vutils.make_grid(htmp[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()'''

        print(real_batch.shape, va.shape, gt.shape,htmp.shape)
        print(va)
        img = unnormalizedAndLandmark(real_batch, inputPred = gt,inputGT = None, customNormalize = np.array([91.4953, 103.8827, 131.0912]))
        #print(va,x,lbl)
        for ig in img :
            cv2.imshow('t',ig)
            cv2.waitKey(0)

'''testSplit = 0
nSplit = 5
listSplit = []
for i in range(nSplit):
    if i!=testSplit :
        listSplit.append(i)

transform =transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
#main()
AFEWVA(["SEMAINE"], None, True, 224, transform, True, False, 1,split=True, nSplit = nSplit,listSplit=listSplit,wHeatmap=False,isVideo=False,seqLength=5)'''
