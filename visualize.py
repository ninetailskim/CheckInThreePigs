###

import tkinter as tk
from tkinter import filedialog
import glob
import cv2
from PIL import Image, ImageTk
from paddlex.cls import transforms
import numpy as np
import os

MW = tk.Tk()
MW.title('View Transform')
MW.geometry('1024x512')

ImagePath = None
CurrentIndex = 0

frame_l = tk.Frame(MW)
frame_l.place(x=0, y=0, anchor='nw')
frame_r = tk.Frame(MW)
frame_r.place(x=260, y=4, anchor='nw')

var = tk.StringVar()
l = tk.Label(frame_l, textvariable=var, bg='green', fg='white', font=('Arial', 12), width=30, height=2)
l.pack()

def CBBtnSelectImgPath():
    global ImagePath
    SelectImgPath = filedialog.askdirectory()
    ImagePath = glob.glob(SelectImgPath + "/*.jpg")
    if ImagePath is not None and len(ImagePath) > 0:
        var.set(os.path.basename(ImagePath[CurrentIndex]))
BtnSelectImgPath = tk.Button(frame_l, text='Select File Folder', font=('Arial', 12), width=20, height=2, command=CBBtnSelectImgPath)
BtnSelectImgPath.pack()

Brightness = 0.0
Contrastness = 0.0
Saturation = 0.0
Hue = 0
AllCompose = None
BchangeState = False

def CBGetNewCompose():
    global AllCompose
    global BchangeState
    BchangeState = True
    AllCompose = transforms.Compose([transforms.RandomDistort(brightness_range=Brightness, brightness_prob=1, contrast_range=Contrastness, contrast_prob=1, saturation_range=Saturation, saturation_prob=1, hue_range=Hue, hue_prob=1)])
BtnNext = tk.Button(frame_l, text='Refresh', font=('Arial', 12), width=20, height=2, command=CBGetNewCompose)
BtnNext.pack()


def CBBright(v):
    global Brightness
    Brightness = float(v)
    CBGetNewCompose()

def CBContrast(v):
    global Contrastness
    Contrastness = float(v)
    CBGetNewCompose()

def CBSaturation(v):
    global Saturation
    Saturation = float(v)
    CBGetNewCompose()

def CBHue(v):
    global Hue
    Hue = int(v)
    CBGetNewCompose()

ScBrightRange = tk.Scale(frame_l, label="Brightness Range", from_=0, to=1, orient=tk.HORIZONTAL, length=250, showvalue=0,tickinterval=0.1, resolution=0.01, command=CBBright)
ScContrastRange = tk.Scale(frame_l, label="Contrast Range", from_=0, to=1, orient=tk.HORIZONTAL, length=250, showvalue=0,tickinterval=0.1, resolution=0.01, command=CBContrast)
ScSaturationRange = tk.Scale(frame_l, label="Saturation Range", from_=0, to=1, orient=tk.HORIZONTAL, length=250, showvalue=0,tickinterval=0.1, resolution=0.01, command=CBSaturation)
ScHueRange = tk.Scale(frame_l, label="Hue Range", from_=0, to=18, orient=tk.HORIZONTAL, length=250, showvalue=0,tickinterval=1, resolution=1, command=CBHue)

ScBrightRange.pack()
ScContrastRange.pack()
ScSaturationRange.pack()
ScHueRange.pack()

def CBBtnLast():
    global CurrentIndex
    global BchangeState
    if ImagePath is not None:
        CurrentIndex = (CurrentIndex - 1) % len(ImagePath)
    BchangeState = True
    var.set(os.path.basename(ImagePath[CurrentIndex]))
BtnLast = tk.Button(frame_l, text='Last', font=('Arial', 12), width=20, height=2, command=CBBtnLast)
BtnLast.pack(side='bottom')

def CBBtnNext():
    global CurrentIndex
    global BchangeState
    if ImagePath is not None:
        CurrentIndex = (CurrentIndex + 1) % len(ImagePath)
    BchangeState = True
    var.set(os.path.basename(ImagePath[CurrentIndex]))
BtnNext = tk.Button(frame_l, text='Next', font=('Arial', 12), width=20, height=2, command=CBBtnNext)
BtnNext.pack(side='bottom')

LbImage = tk.Label(frame_r)
LbImage.pack()

State = True
StateName = tk.StringVar() 
StateName.set('Origin')

def CBBtnChangeState():
    global State
    global StateName
    if State == False:
        State = True
        StateName.set('Origin')
    else:
        State = False
        StateName.set('Transformed')

BtnChangeState = tk.Button(frame_l, textvariable=StateName, font=('Arial', 12), width=20, height=2, command=CBBtnChangeState)
BtnChangeState.pack()

resimg = None

def show_img():
    global BchangeState
    global resimg
    if ImagePath is not None and len(ImagePath) > 0:
        tmpimg = cv2.imread(ImagePath[CurrentIndex])
        #cv2.imshow("123", tmpimg)
        #cv2.waitKey(100)
        
        ss = tmpimg.shape
        
        xx = ss[0] / 504
        yy = ss[1] / 760
        print(ss, xx, yy)
        if (ss[1] / xx) > 760:
            ss = (round(ss[1] / yy), 504)
        else:
            ss= (760, round(ss[0] / xx))
        print(ss)
        tmpimg = cv2.resize(tmpimg,ss,interpolation=cv2.INTER_CUBIC) 
        cv2image = cv2.cvtColor(tmpimg, cv2.COLOR_BGR2RGBA)
        transimage = tmpimg.copy()
        if BchangeState:
            timg = transimage.astype(np.float32) / 255
            resimg = (AllCompose.__call__(timg)[0] * 255).astype('uint8')
            BchangeState = False
        if State:
            Imageimg = Image.fromarray(cv2image)
        else:
            Imageimg = Image.fromarray(resimg)
        imgtk = ImageTk.PhotoImage(image=Imageimg)
        LbImage.imgtk = imgtk
        LbImage.config(image=imgtk)
    frame_r.after(1, show_img)

CBGetNewCompose()

show_img()

MW.mainloop()

cv2.destroyAllWindows()