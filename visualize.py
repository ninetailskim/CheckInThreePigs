###

import tkinter as tk
from tkinter import filedialog
import glob
import cv2
from PIL import Image, ImageTk

MW = tk.Tk()
MW.title('View Transform')
MW.geometry('1024x512')

ImagePath = None
CurrentIndex = 0


frame_l = tk.Frame(MW)
frame_l.place(x=0, y=0, anchor='nw')
frame_r = tk.Frame(MW)
frame_r.place(x=256, y=0, anchor='nw')


var = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
l = tk.Label(frame_l, textvariable=var, bg='green', fg='white', font=('Arial', 12), width=30, height=2)
# 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
l.pack()

def CBBtnSelectImgPath():
    global ImagePath
    SelectImgPath = filedialog.askdirectory()
    ImagePath = glob.glob(SelectImgPath + "/*.jpg")

BtnSelectImgPath = tk.Button(frame_l, text='Select File Folder', font=('Arial', 12), width=20, height=2, command=CBBtnSelectImgPath)
BtnSelectImgPath.pack()

def CBRbSelectTask():
    l.config(text=var.get())
R1 = tk.Radiobutton(frame_l, text="Classification", variable=var, value=1, command=CBRbSelectTask)
R2 = tk.Radiobutton(frame_l, text="Classification", variable=var, value=2, command=CBRbSelectTask)
R3 = tk.Radiobutton(frame_l, text="Classification", variable=var, value=3, command=CBRbSelectTask)
R1.pack()
R2.pack()
R3.pack()

LbImage = tk.Label(frame_r)
LbImage.pack()
def show_img():
    if ImagePath is not None:
        print(len(ImagePath))
        print(ImagePath[CurrentIndex])
        tmpimg = cv2.imread(ImagePath[CurrentIndex])
        cv2image = cv2.cvtColor(tmpimg, cv2.COLOR_BGR2RGBA)
        Imageimg = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=Imageimg)
        LbImage.config(image=imgtk)
    frame_r.after(1, show_img)

























































show_img()

MW.mainloop()