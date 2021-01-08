from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import cv2
from numpy import random


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
from django.http import JsonResponse
UNKNOWN_DIR = 'media'
def index(request):
   
    
    #img = cv2.imread('/media/82.jpg')
    # for filename in os.listdir(UNKNOWN_FACES_DIR):
    #     print(f'Filename {filename}', end='')
    #     img = cv2.imread(f'{UNKNOWN_FACES_DIR}/{filename}');
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
    
    # img = cv2.resize(img,(700,500))

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return render(request, 'nhandangbiensoxe.html')

def find(request):
    UNKNOWN_DIR = 'media'
    for i in os.listdir(UNKNOWN_DIR):
        file_path = os.path.join(UNKNOWN_DIR, i)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    
    # if request.method == 'POST' and request.FILES['myfile']:
    #     myfile = request.FILES['myfile']
    #     # print("===================12311414")
    #     # print(myfile.name)
    #     # print(myfile)
    #     # print("===================12311414")
    #     fs = FileSystemStorage()
    #     filename = fs.save('media/' + myfile.name, myfile)
    #     uploaded_file_url = fs.url(filename)
    #     print(uploaded_file_url)
    #     print(myfile.name)
    #     print("=========12313")
    #     file_bath = uploaded_file_url.replace("/","", 1)
    #     print(file_bath)
    #     img = cv2.imread(f'{file_bath}');
     
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
    #     list_contours = []
    #     # cv2.imshow("img", img)
    #     # cv2.waitKey()
    #     img = cv2.resize(img, (1200, 900))
    #     # cv2.imshow("img_1", img)
    #     cv2.waitKey()
    #     im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     im_gray = cv2.equalizeHist(im_gray)

    #     r, binary = cv2.threshold(im_gray, 150, 255, cv2.THRESH_BINARY)
    #     list_contours, bien_so_xe = nhandien(binary, img, 5)
    #     print(len(list_contours))
    #     if len(list_contours) != 8:
    #         for i in range(5, 12, 2):
    #             for j in range(100, 250, 1):
    #                 # print(i,'--------', j)
    #                 r, binary = cv2.threshold(im_gray, j, 255, cv2.THRESH_BINARY)
    #                 list_contours, bien_so_xe = nhandien(binary, img,  i)
    #                 if len(list_contours) == 8:
    #                     # print('hello')
    #                     break
    #             if len(list_contours) == 8:
    #                 # print('hello')
    #                 break
    #     crop_characters = []
    #     # cv2.imshow('img', bien_so_xe)
    #     # cv2.waitKey(0)
    #     cv2.imwrite("media/bienso.jpg", bien_so_xe)
    #     for contours in list_contours:
    #         # cv2.imshow(contours)
    #         # cv2.waitKey(0)
    #         image = cv2.resize(contours,(30,60))
    #         # convert to grayscale and blur the image
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         blur = cv2.GaussianBlur(gray,(7,7),0)
            
    #         # Applied inversed thresh_binary 
    #         binary = cv2.threshold(blur, 200, 255,
    #                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
    #         # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #         # thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    #         crop_characters.append(binary)
    #         # cv2_imshow(binary)
            
    #         # cv2.waitKey(0)

    #     #cv2.imwrite("1_1123121.jpg", crop_characters[7])
    #     cv2.destroyAllWindows()

    #     # Load model architecture, weight and labels
    #     json_file = open('MobileNets_character_recognition.json', 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     model = model_from_json(loaded_model_json)
    #     model.load_weights("License_character_recognition_weight.h5")
    #     print("[INFO] Model loaded successfully...")

    #     labels = LabelEncoder()
    #     labels.classes_ = np.load('license_character_classes.npy')
    #     print("[INFO] Labels loaded successfully...")
    #     def predict_from_model(image,model,labels):
    #         image = cv2.resize(image,(80,80))
    #         image = np.stack((image,)*3, axis=-1)
    #         prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    #         return prediction
    #     fig = plt.figure(figsize=(15,3))

    #     cols = len(crop_characters)
    #     grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

    #     final_string = ''
    #     for i,character in enumerate(crop_characters):
    #         fig.add_subplot(grid[i])
    #         title = np.array2string(predict_from_model(character,model,labels))
    #         plt.title('{}'.format(title.strip("'[]"),fontsize=20))
    #         final_string+=title.strip("'[]")
    #         plt.axis(False)
    #         plt.imshow(character,cmap='gray')
    #     plt.savefig('media/kytubien.jpg')

    #     #plt.show()

    #     print(final_string)
    # return render(request, 'test.html', {'img': uploaded_file_url, 'bienso':final_string,'anhbienso':'/media/bienso.jpg', 'kytubien':'/media/kytubien.jpg'})
    if request.is_ajax() and request.method == "POST":
        myfile = request.FILES.get('files')
        fs = FileSystemStorage()
        filename = fs.save('media/' + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        file_bath = uploaded_file_url.replace("/","", 1)
       
        img = cv2.imread(f'{file_bath}');
     
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        list_contours = []
        # cv2.imshow("img", img)
        # cv2.waitKey()
        img = cv2.resize(img, (1200, 900))
        # cv2.imshow("img_1", img)
        cv2.waitKey()
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.equalizeHist(im_gray)

        r, binary = cv2.threshold(im_gray, 150, 255, cv2.THRESH_BINARY)
        list_contours, bien_so_xe = nhandien(binary, img, 5)
        print(len(list_contours))
        if len(list_contours) != 8:
            for i in range(5, 12, 2):
                for j in range(100, 250, 1):
                    # print(i,'--------', j)
                    r, binary = cv2.threshold(im_gray, j, 255, cv2.THRESH_BINARY)
                    list_contours, bien_so_xe = nhandien(binary, img,  i)
                    if len(list_contours) == 8:
                        # print('hello')
                        break
                if len(list_contours) == 8:
                    # print('hello')
                    break
        crop_characters = []
        # cv2.imshow('img', bien_so_xe)
        # cv2.waitKey(0)
        str_bienso = "media/bienso" + str(random.randint(0,1000)) + ".jpg"
        cv2.imwrite(str_bienso, bien_so_xe)
        for contours in list_contours:
            # cv2.imshow(contours)
            # cv2.waitKey(0)
            image = cv2.resize(contours,(30,60))
            # convert to grayscale and blur the image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(7,7),0)
            
            # Applied inversed thresh_binary 
            binary = cv2.threshold(blur, 200, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            crop_characters.append(binary)
            # cv2_imshow(binary)
            
            # cv2.waitKey(0)

        #cv2.imwrite("1_1123121.jpg", crop_characters[7])
        cv2.destroyAllWindows()

        # Load model architecture, weight and labels
        json_file = open('MobileNets_character_recognition.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("License_character_recognition_weight.h5")
        print("[INFO] Model loaded successfully...")

        labels = LabelEncoder()
        labels.classes_ = np.load('license_character_classes.npy')
        print("[INFO] Labels loaded successfully...")
        def predict_from_model(image,model,labels):
            image = cv2.resize(image,(80,80))
            image = np.stack((image,)*3, axis=-1)
            prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
            return prediction
        fig = plt.figure(figsize=(15,3))

        cols = len(crop_characters)
        grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

        final_string = ''
        for i,character in enumerate(crop_characters):
            fig.add_subplot(grid[i])
            title = np.array2string(predict_from_model(character,model,labels))
            plt.title('{}'.format(title.strip("'[]"),fontsize=20))
            final_string+=title.strip("'[]")
            plt.axis(False)
            plt.imshow(character,cmap='gray')
        str_kytubien = "media/kytubien" + str(random.randint(0,1000)) + ".jpg"
        plt.savefig(str_kytubien)

        #plt.show()

        print(final_string)
        return JsonResponse({'img': uploaded_file_url, 'bienso':final_string,'anhbienso':str_bienso, 'kytubien':str_kytubien})


def swap(a, b):
    c = a
    a = b
    b = c
    return a, b

def sx(kq, list_contours):
    for i in range(len(kq)-1):
        for j in range(i+1, len(kq)):
            if kq[i] > kq[j]:
                kq[i], kq[j] = swap(kq[i], kq[j])
                list_contours[i], list_contours[j] = swap(list_contours[i], list_contours[j])
    return list_contours



def min_khoangcach(kq, min_x):
    k = max(kq)
    for i in kq:
        if (min_x - i) < k and (min_x-i) >= 0:
            k = (min_x - i)
    return k

def timContour(binary, k):

    kernel = np.ones((k, k), np.uint8)

    erosion = cv2.erode(binary, kernel, iterations=1)

    closing = binary - erosion


    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    return contours


def cuctri(contour):
    x_min = contour[0][0][0]
    x_max = 0
    y_min = contour[0][0][1]
    y_max = 0
    for c in contour:
        if c[0][0] > x_max:
            x_max = c[0][0]
        if c[0][0] < x_min:
            x_min = c[0][0]
        if c[0][1] > y_max:
            y_max = c[0][1]
        if c[0][1] < y_min:
            y_min = c[0][1]
    return x_min, x_max, y_min, y_max

def nhandien(binary,img, k):
  
    list_contours = []
    bien_sau_khi_cat = img
    contours = timContour(binary, k)
    for contour in contours:
        x_min, x_max, y_min, y_max = cuctri(contour)
        if y_max != y_min:
            t = (x_max-x_min) / (y_max-y_min)
        else:
            t = 0

        if  2.3 < t and 4.2 > t:
            new_img = binary[y_min: y_max, x_min:x_max]
            new_img = cv2.resize(new_img, (700, 500))
            img_return = img[y_min: y_max, x_min:x_max]
            bien_sau_khi_cat = img_return
            img_return = cv2.resize(img_return, (700, 500))
            # cv2.imshow("im", bien_sau_khi_cat)
            # cv2.waitKey(0)
            new_contours = timContour(new_img, 3)


            kq = []
            list_contours = []
            for c in new_contours:
                x_min_1, x_max_1, y_min_1, y_max_1 = cuctri(c)
                if y_max_1 != y_min_1:
                    t = (x_max_1 - x_min_1) / (y_max_1 - y_min_1)
                else:
                    t = 0
                new_img_1 = img_return[y_min_1   : y_max_1   , x_min_1 + 2  :x_max_1 + 2 ]
                w = x_max_1-x_min_1
                h = y_max_1-y_min_1
                # print(w,'-,',h)
                if w > 20 and w < 80 and h > 215 and h < 420:
                    if kq == []:
                        kq.append(x_min_1)
                        list_contours.append(new_img_1)

                        # cv2.imshow("new_image_1", new_img_1)
                        # cv2.waitKey()
                    else:
                        min_kc = min_khoangcach(kq, x_min_1)
                        if min_kc > 50:
                            kq.append(x_min_1)
                            list_contours.append(new_img_1)
                            # cv2.imshow("new_image_1", new_img_1)
                            # cv2.waitKey()
                    if len(list_contours) == 8:
                        list_contours = sx(kq, list_contours)
                        return list_contours, bien_sau_khi_cat
    return list_contours, bien_sau_khi_cat





