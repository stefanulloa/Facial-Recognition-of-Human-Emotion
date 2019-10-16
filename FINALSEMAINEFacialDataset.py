
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

import file_walker
from utils import *
import utils
from os.path import isfile


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
                            '''for sub_2_f in sub_f.walk():
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
                                        counter_image+=len(list_dta)'''

                            #SEMAINE
                            list_dta = []

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
                                counter_image+=len(list_dta)

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


      
        t_l_imgs = []
        t_l_gt = []
        t_l_gtE = []

        t_list_gt_names = []
        t_list_gtE_names = []

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


        self.l_imgs = []
        self.l_gt = []
        self.l_gtE = []

        self.list_gt_names = []
        self.list_gtE_names = []


        if split :

            indexer = 0

            self.l_gt = []
            self.l_gtE = []

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

                for i in range(begin,end) :
                    for j in range(len(t_l_imgs[i])):
                        #print('append ',t_l_imgs[i][j])
                        self.l_imgs.append(t_l_imgs[i][j])
                        self.l_gt.append(t_l_gt[i][j])
                        self.l_gtE.append(t_l_gtE[i][j])

                        self.list_gt_names.append(t_list_gt_names[i][j])
                        self.list_gtE_names.append(t_list_gtE_names[i][j])
                        indexer+=1

                print(len(self.l_imgs))

            self.l_gt = np.asarray(self.l_gt)
            self.l_gtE = np.asarray(self.l_gtE)
        else :

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


        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #Im image : [batch_size, channel, width, height]

        l_imgs = []; l_VA = []; l_ldmrk = []; l_nc = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]

        x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_l = [self.l_gt[index].copy()];label_n =[self.list_gt_names[index]]

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


            if self.useIT :
                tImage = self.transformInternal(tImage)
            else :
                tImage = self.transform(tImage)

            l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]

        return l_imgs[0], l_VA[0], l_ldmrk[0]


    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def __len__(self):
        return len(self.l_imgs)


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    print(len(dataloader))
    # Plot some training images
    for real_batch,va,gt,htmp in (dataloader) :


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
