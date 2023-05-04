from tkinter import *
import os
from tkinter import messagebox, filedialog
import imageio
from PIL import Image, ImageTk
import predict


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.place()
        self.createWidget()

    def createWidget(self):
        global photo
        photo = None
        self.label03 = Label(self, image=photo)
        self.label03.grid(column=0, row=0)
        # self.label03.place(relx=0.5, rely=0.5, anchor=CENTER)

        self.btn01 = Button(self, text='Open', command=self.getfile, bg='white', anchor='s')
        self.btn01.grid(column=0, row=1)
        # self.btn01.place(relx=0.8, rely=0.5, anchor=CENTER)

        self.init_data_Text = Text(self, width=35, height=1)
        self.init_data_Text.grid(row=2, column=0, rowspan=10, columnspan=10)

    def getfile(self):
        file_path = filedialog.askopenfilename(title='Choose file', filetypes=[('PNG', '*.png'), ('All Files', '*')])
        img = Image.open(file_path)
        width, height = img.size
        word = predict.predict1(file_path)
        if word == "0":
            word = "Negative_IDC"
        else:
            word = "Positive_IDC"



        img = img.resize((224, int(224 / width * height)))

        global photo
        photo = ImageTk.PhotoImage(img)
        self.label03.configure(image=photo)
        self.label03.image = photo
        self.init_data_Text.delete(1.0, END)
        self.init_data_Text.insert(1.0, word)


#predict.predict1('F:/Old directory/test/0/16551_idx5_x2451_y1101_class0.png')
root = Tk()

root.geometry('350x300')
root.title("Image recognition")
app = Application(master=root)

root.mainloop()
