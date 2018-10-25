#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score
from sklearn.externals import joblib

class data_prediction:
    def __init__(self, master):
        self.master = master
        master.title('Data Prediction Demo')
        master.geometry('300x400')
        tk.Label(master, text = 'Data Prediction Demo').pack(pady = 10)

        #--- right eye and left eye --- #
        frm1 = tk.Frame(master)
        frm1.pack(fill = 'x')
        label_right = tk.Label(frm1, text = '右眼', font = (12),
                            bg = 'green', fg = 'white', width = 5, height = 2)
        label_right.pack(side = 'left', padx = 80)

        label_left = tk.Label(frm1, text = '左眼', font = (12),
                            bg = 'green', fg = 'white', width = 5, height = 2)
        label_left.pack(side = 'left')

        #--- Frame 2: different data
        frm2 = tk.Frame(master)
        frm2.pack(fill = 'x')

        # eyes view
        label_v = tk.Label(frm2, text = '视力', height = 2)
        label_v.grid(row = 0, column = 0, padx = 10)
        self.entry_r_v = tk.Entry(frm2, show = None, width = 7)
        self.entry_r_v.grid(row = 0, column = 1, padx = 15)
        self.entry_l_v = tk.Entry(frm2, show = None, width = 7)
        self.entry_l_v.grid(row = 0, column = 2, padx = 50)

        # sphere
        label_s = tk.Label(frm2, text = '球镜S', height = 2)
        label_s.grid(row = 1, column = 0, padx = 10)
        self.entry_r_s = tk.Entry(frm2, show = None, width = 7)
        self.entry_r_s.grid(row = 1, column = 1, padx = 15)
        self.entry_l_s = tk.Entry(frm2, show = None, width = 7)
        self.entry_l_s.grid(row = 1, column = 2, padx = 50)

        # cylinder
        label_c = tk.Label(frm2, text = '柱镜C', height = 2)
        label_c.grid(row = 2, column = 0, padx = 10)
        self.entry_r_c = tk.Entry(frm2, show = None, width = 7)
        self.entry_r_c.grid(row = 2, column = 1, padx = 14)
        self.entry_l_c = tk.Entry(frm2, show = None, width = 7)
        self.entry_l_c.grid(row = 2, column = 2, padx = 50)

        # angle
        label_a = tk.Label(frm2, text = '轴位A', height = 2)
        label_a.grid(row = 3, column = 0, padx = 10)
        self.entry_r_a = tk.Entry(frm2, show = None, width = 7)
        self.entry_r_a.grid(row = 3, column = 1, padx = 15)
        self.entry_l_a = tk.Entry(frm2, show = None, width = 7)
        self.entry_l_a.grid(row = 3, column = 2, padx = 50)

        # pupil distance
        label_pd = tk.Label(frm2, text = '瞳距PD', height = 2)
        label_pd.grid(row = 4, column = 0, padx = 10)
        self.entry_pd = tk.Entry(frm2, show = None, width = 7)
        self.entry_pd.grid(row = 4, column = 1, padx = 15)

        #--- Frame 3: click button and display the result
        frm3 = tk.Frame(master)
        frm3.pack(fill = 'x')

        button = tk.Button(frm3, text = '诊断结果',
                        width = 10, height = 2, command = self.show_result)
        button.pack(side = 'left', padx = 10)
        self.text = tk.Text(frm3, width = 20, height = 2)
        self.text.pack(side = 'left', padx = 10)
        #--- load pre-trained model
        self.clf = joblib.load("train_model.m")

    def show_result(self):
        self.text.delete(0.0, 'end')
        r_v = self.entry_r_v.get()
        l_v = self.entry_l_v.get()
        r_s = self.entry_r_s.get()
        l_s = self.entry_l_s.get()
        r_c = self.entry_r_c.get()
        l_c = self.entry_l_c.get()
        r_a = self.entry_r_a.get()
        l_a = self.entry_l_a.get()
        p_distance  = self.entry_pd.get()

        if  r_v == '' or \
            l_v == '' or \
            r_s == '' or \
            l_s == '' or \
            r_c == '' or \
            l_c == '' or \
            r_a == '' or \
            r_c == '' or \
            p_distance == '':
            print('null value')
            self.text.insert('insert', '参数输入不齐全')
        else:
            try:
                r_v = float(r_v)
                l_v = float(l_v)
                r_s = float(r_s)
                l_s = float(l_s)
                r_c = float(r_c)
                l_c = float(l_c)
                r_a = int(r_a)
                l_a = int(l_a)
                p_distance  = int(p_distance)
                data = pd.DataFrame([[l_c, l_s, l_a, p_distance, r_c, r_s, r_a, l_v, r_v]],
                                    columns = ['L_DC', 'L_DS', 'L_axis', 'PD', 'R_DC', 'R_DS', 'R_axis', 'left_eye', 'right_eye'])
                result = str(self.clf.predict(data))[2:-2]
                print(result)
                self.text.insert('insert', result)
            except Exception as e:
                print('error value: ', str(e))
                self.text.insert('insert', 'Oops, ERROR!')

root = tk.Tk()
my_gui = data_prediction(root)
root.mainloop()