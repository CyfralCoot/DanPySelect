# -*- coding: utf-8 -*-

def get_os():
    if os.name == 'nt':
        return "Windows"
    elif os.name == 'posix':
        return "Linux"
    else:
        return "Unknown"

import cv2
from PIL import Image, ImageTk, ImageEnhance
if not hasattr(Image, 'Resampling'):  # Pillow<9.0 Linux
    Image.Resampling = Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import glob
import os
import shutil
import time
import math
from copy import copy

import tkinter as tk
from tkinter import ttk, filedialog

if get_os() == 'Windows': #If fails on Linux Mint, change to try except
    from idlelib.tooltip import Hovertip
else:
    from tktooltip import ToolTip as Hovertip #If on linux

from threading import Thread
import configparser
import subprocess

from run_deleter_cluster import *
import fit_polar
import logging_module as logm
import plot_trace

import logging
logging.basicConfig(level=logging.INFO)


#Technical
batch_operation = False


class MyWindowObject:
    def __init__(self):
        self.window = None
    
    def checkwindow(self):
        query = self.window is not None and tk.Toplevel.winfo_exists(self.window)
        if query:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()
        return query

class MyCanvasObject:
    def __init__(self):
        self.id = None
    
    def visible(self) -> bool:
        """Returns whether the object is present or not"""
        return self.id is not None

class RectangleSelection(MyCanvasObject):
    def __init__(self, canvas, seltype=True, title=None):
        self.canvas = canvas
        self.seltype = seltype
        self.color = 'green' if seltype == True else 'red'
        self.id = None
        self.cx1 = None
        self.cx2 = None
        self.cy1 = None
        self.cy2 = None
        self.title = title
    
    def image_coords(self, image_delta_x, image_delta_y):
        x1 = self.cx1 + image_delta_x
        x2 = self.cx2 + image_delta_x
        y1 = self.cy1 + image_delta_y
        y2 = self.cy2 + image_delta_y
        
        x1 = clip(x1, 0, settings.image_res[0])
        x2 = clip(x2, 0, settings.image_res[0])
        y1 = clip(y1, 0, settings.image_res[1])
        y2 = clip(y2, 0, settings.image_res[1])        
        return x1, x2, y1, y2
    
    def check_coords(self) -> bool:
        """Check if rectangle is big enough"""
        if self.cx1 is None:
            return False
        if abs(self.cx1 - self.cx2) < 10:
            return False
        if abs(self.cy1 - self.cy2) < 10:
            return False
        return True

    def draw_rectangle(self):
        if self.id is not None:
            self.canvas.delete(self.id)
        self.id = self.canvas.create_rectangle(self.cx1, self.cy1, self.cx2, self.cy2, dash=2, outline=self.color)

    def hide_rectangle(self):
        if self.id is not None:
            self.canvas.delete(self.id)
            self.id = None
            self.cx1 = None
            self.cx2 = None
            self.cy1 = None
            self.cy2 = None

    def update_rectangle(self):
        if self.id is not None:
            self.canvas.coords(self.id, self.cx1, self.cy1, self.cx2, self.cy2)
    
    def on_click(self, event):
        self.cx1 = event.x
        self.cy1 = event.y
        self.cx2 = self.cx1
        self.cy2 = self.cy1
        logger.debug(f'Click coords: {self.cx1}, {self.cy1}')
        self.draw_rectangle()
    
    def on_drag(self, event):
        global cur_x, cur_y
        self.cx2 = event.x
        self.cy2 = event.y
        self.update_rectangle()
    
    def on_release(self, event):
        if self.check_coords():
            self.cx1, self.cx2 = sorted([self.cx1, self.cx2])
            self.cy1, self.cy2 = sorted([self.cy1, self.cy2])
        else:
            self.hide_rectangle()
        if batch_operation:
            batch.crop_update()
        else:
            single.crop_update()

class HorizontalLine(MyCanvasObject):
    def __init__(self, canvas, y_pos):
        self.canvas = canvas
        self.canv_y = y_pos
        self.id = None
        self.last_call_time = 404
    
    def updatefn(self):
        print('hline updatefn unassigned')
    
    def true_y(self, image_delta_y) -> float:
        pixel_height = -self.canv_y*settings.image_downscale - image_delta_y
        return pixel_height*settings.grid_d
    
    def hide(self):
        if self.id:
            self.canvas.delete(self.id)
            self.id = None
            self.call_update()
    
    def draw_hline(self):
        if self.id:
            self.canvas.delete(self.id)
        self.id = self.canvas.create_line(0, self.canv_y, frame_w, self.canv_y, fill="lime")
        self.call_update()
        
    def call_update(self):
        current_time = time.time()
        if current_time - self.last_call_time > 0.2:#update only if the function hasn't been called recently
            self.updatefn()
        self.last_call_time = current_time
    
    def move(self):
        if self.id:
            self.canvas.coords(self.id, 0, self.canv_y, frame_w, self.canv_y)
            self.call_update()
        else:
            self.draw_hline()
    
    def move_down(self, event=None):
        self.canv_y += 1
        self.move()
    
    def move_up(self, event=None):
        self.canv_y -= 1
        self.move()
    
    def set_y(self, y_or_event):
        if isinstance(y_or_event, int) or isinstance(y_or_event, float):
            self.canv_y = y_or_event
        elif isinstance(y_or_event, tk.Event):
            self.canv_y = y_or_event.y
        self.move()
    
    def set_state(self, state):
        if state == self.visible():
            return
        else:
            if state == True:
                self.draw_hline()
            else:
                self.hide()
    
    def toggle(self, event=None):
        if self.visible():
            self.hide()
        else:
            self.draw_hline()

class MainHorizontalLine(HorizontalLine):
    def updatefn(self):
        if batch_operation:
            batch.crop_update()
            batch.update_plot()
        else:
            single.crop_update()
            single.update_plot()            

class LensHorizontalLine(HorizontalLine):
    def updatefn(self):
        lens_canvas.update()
    
    def hover_update(self, y):
        self.set_y(settings.lens_size[1]/2 + settings.lens_zoom*(hline.canv_y - y + 0.5))
        self.set_state(hline.visible())  

class CanvasImage:
    def __init__(self, canvas, target_size):
        self.canvas = canvas
        self.anchor = tk.NE
        self.t_width = target_size[0]
        self.t_height = target_size[1]
        self.id = self.canvas.create_image(self.t_width, 0, anchor=self.anchor)
        self.image = None
        self.tk_img = None
        self.delta_x = 0
        self.delta_y = 0

    def update_internal(self, pil_img):
        self.image = pil_img
        if pil_img is None:
            self.tk_img = None
        else:
            self.tk_img = ImageTk.PhotoImage(pil_img)
            try:
                self.canvas.itemconfig(self.id, anchor=self.anchor, image=self.tk_img)
            except Exception as e:
                logger.error(f'Error updating image: {e}')
    
    def set(self, pil_img):
        self.update_internal(pil_img)

    def clear(self):
        self.set(None)

class MainCanvasImage(CanvasImage):
    def update_deltas(self):
        if self.image is not None:
            imsize = self.image.size
            self.delta_x = imsize[0] - settings.image_res[0]
            self.delta_y = 0
            logger.debug(f'Image res {imsize}')
        else:
            self.delta_x = 0
            self.delta_y = 0

    def set(self, pil_img):
        self.update_internal(pil_img)
        self.update_deltas()

class SettingsHandler(MyWindowObject):
    def __init__(self):
        self.window = None
        
        # Read configuration file
        config = configparser.ConfigParser()
        with open('config.ini', 'r', encoding='utf-8') as f:
            config.read_file(f)        
        self.fullscreen = config.getboolean('InitialSettings', 'fullscreen')
        self.image_res = tuple(map(int, config.get('InitialSettings', 'image_res').split(',')))
        self.image_downscale = config.getint('InitialSettings', 'image_downscale')
        self.initial_hline_height = config.getint('InitialSettings', 'initial_hline_height')
        self.lens_size = tuple(map(int, config.get('InitialSettings', 'lens_size').split(',')))
        self.lens_zoom = config.getint('InitialSettings', 'lens_zoom')
        self.default_path = config.get('InitialSettings', 'default_path')
        self.out_file_name = config.get('InitialSettings', 'out_file_name')
        self.model_path = config.get('InitialSettings', 'model_path')
        self.default_model = config.get('InitialSettings', 'default_model')
        self.default_treshold = config.getfloat('InitialSettings', 'default_treshold')
        self.fit_epochs = config.getint('InitialSettings', 'fit_epochs')
        self.do_canny_on_selection = config.getboolean('InitialSettings', 'do_canny_on_selection')
        self.show_canny_image = config.getboolean('InitialSettings', 'show_canny_image')
        self.hline_crop = config.getboolean('InitialSettings', 'hline_crop')
        self.grid_d = config.getfloat('InitialSettings', 'grid_d')
        self.batch_prev_count = config.getint('InitialSettings', 'batch_preview_count')
        self.select_dir = config.get('InitialSettings', 'select_dir')
        self.ooo_temp_dir = config.get('InitialSettings', 'ooo_temp_dir')
        self.select_name = config.get('InitialSettings', 'select_name')
        self.select_s_name = config.get('InitialSettings', 'select_s_name')
        self.select_waiting_time = int(config.get('InitialSettings', 'select_waiting_time (sec)'))
        self.autosave = config.getboolean('InitialSettings', 'autosave_results')
        self.autosave_plot = config.getboolean('InitialSettings', 'autosave_plots')
        #Canny parameters:
        self.t_lower = config.getint('CannyParameters', 't_lower')
        self.t_upper = config.getint('CannyParameters', 't_upper')
        self.aperture_size = config.getint('CannyParameters', 'aperture_size')
        self.l2gradient = config.getboolean('CannyParameters', 'L2Gradient')
        self.subpixel_diagonals = config.getboolean('CannyParameters', 'subpixel_diagonals')
        
        self.init_tkinter_vars()
        self.update()
    
    def init_tkinter_vars(self):
        self.var_treshold = tk.StringVar(value = str(self.default_treshold))
        self.var_pendant = tk.BooleanVar()
        self.var_select_in_cmd = tk.BooleanVar()
        self.var_t_lower = tk.StringVar(value = str(self.t_lower))
        self.var_t_upper = tk.StringVar(value = str(self.t_upper))
        self.var_aperture_size = tk.StringVar(value = str(self.aperture_size))
        self.var_l2gradient = tk.BooleanVar(value = self.l2gradient)
        self.var_subpixel_diagonals = tk.BooleanVar(value = self.subpixel_diagonals)
        self.var_do_canny_on_selection = tk.BooleanVar(value = self.do_canny_on_selection)
        self.var_show_canny_image = tk.BooleanVar(value = self.show_canny_image)
        self.var_autosave = tk.BooleanVar(value = self.autosave)
        self.var_autosave_plot = tk.BooleanVar(value = self.autosave_plot)
        self.var_hline_crop = tk.BooleanVar(value = self.hline_crop)     
        
        self.var_t_upper.trace_add("write", lambda name, index, mode, sv=None: self.update())#callback for elements we can't assign command to
        self.var_t_lower.trace_add("write", lambda name, index, mode, sv=None: self.update())
        self.var_aperture_size.trace_add("write", lambda name, index, mode, sv=None: self.update())
    
    def update(self, event=None):
        #logger.debug('Updated settings') #not defined yet
        
        self.treshold = float(self.var_treshold.get())
        self.pendant = self.var_pendant.get()
        self.select_in_cmd = self.var_select_in_cmd.get()
        self.t_lower = int(self.var_t_lower.get())
        self.t_upper = int(self.var_t_upper.get())
        self.aperture_size = int(self.var_aperture_size.get())
        self.l2gradient = self.var_l2gradient.get()
        self.subpixel_diagonals = self.var_subpixel_diagonals.get()
        self.do_canny_on_selection = self.var_do_canny_on_selection.get()
        self.show_canny_image = self.var_show_canny_image.get()
        self.autosave = self.var_autosave.get()
        self.autosave_plot = self.var_autosave_plot.get()
        self.hline_crop = self.var_hline_crop.get()
    
    def open_window(self):
        if self.checkwindow():
            return
        
        self.window = tk.Toplevel(root)
        self.window.title('Settings')
        self.window.geometry('450x600')
        self.window.attributes('-topmost', 'true')
        
        validate_num = self.window.register(validate_numeric_input)
        n = 0
        tk.Label(self.window, text='--- Edge detection settings ---', font = "Verdana 11").grid(row=n, column=0, padx=5, pady=10)
        n += 1
        tk.Label(self.window, text='Lower treshold:').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        canny_field1 = tk.Entry(self.window, textvariable=settings.var_t_lower, validate='key', validatecommand=(validate_num, '%P'))
        canny_field1.grid(row=n, column=1, padx=10, pady=10)
        n += 1
        tk.Label(self.window, text='Upper treshold:').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        canny_field2 = tk.Entry(self.window, textvariable=settings.var_t_upper, validate='key', validatecommand=(validate_num, '%P'))
        canny_field2.grid(row=n, column=1, padx=10, pady=5)
        n += 1
        tk.Label(self.window, text='Apperture size:').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        spinbox1 = ttk.Combobox(self.window, textvariable=settings.var_aperture_size, values=tuple(range(3, 32, 2)))
        spinbox1.grid(row=n, column=1, padx=10, pady=5)
        n += 1
        tk.Label(self.window, text='L2Gradient').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        switch1 = tk.Checkbutton(self.window, variable=settings.var_l2gradient, state='disabled', command=self.update)
        switch1.grid(row=n, column=1, padx=10, pady=5)
        n += 1
        tk.Label(self.window, text='Subpixel diagonals').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        switch1 = tk.Checkbutton(self.window, variable=settings.var_subpixel_diagonals, command=self.update)
        switch1.grid(row=n, column=1, padx=10, pady=5)
        n += 1
        tk.Label(self.window, text='--- Plotting ---', font = "Verdana 11").grid(row=n, column=0, padx=5, pady=10)
        n += 1
        tk.Label(self.window, text='Run Canny on file selection').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        switch2 = tk.Checkbutton(self.window, variable=settings.var_do_canny_on_selection, command=self.update)
        switch2.grid(row=n, column=1, padx=10, pady=5)
        n += 1
        tk.Label(self.window, text='Show Canny image').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        switch3 = tk.Checkbutton(self.window, variable=settings.var_show_canny_image, command=self.update)
        switch3.grid(row=n, column=1, padx=10, pady=5)
        n += 1
        tk.Label(self.window, text='Remove points under the line').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        switch3 = tk.Checkbutton(self.window, variable=settings.var_hline_crop, command=self.update)
        switch3.grid(row=n, column=1, padx=10, pady=5)
        n += 1    
        tk.Label(self.window, text='--- Auto-save ---', font = "Verdana 11").grid(row=n, column=0, padx=5, pady=10)
        n += 1
        tk.Label(self.window, text='Save results').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        switch4 = tk.Checkbutton(self.window, variable=settings.var_autosave, command=self.update)
        switch4.grid(row=n, column=1, padx=10, pady=5)
        n += 1
        tk.Label(self.window, text='Save fit plot').grid(row=n, column=0, padx=5, pady=5, sticky='w')
        switch5 = tk.Checkbutton(self.window, variable=settings.var_autosave_plot, command=self.update)
        switch5.grid(row=n, column=1, padx=10, pady=5)      
        
        #settings.fit_epochs = 600
        #logging.basicConfig(level=logging.INFO)

class PointSelector(MyWindowObject):
    def __init__(self, root, canvas):
        self.root = root
        self.canvas = canvas
        self.window = None
        self.point_color = 'cyan'
        self.canvas.bind("<Button-2>", self.set_origin)
        root.bind("<Delete>", self.remove_point)

    def plot_points(self):
        if self.df is None:
            return
        for index, row in self.df.iterrows():
            x, y = row['pixel_x'], row['pixel_y']
            #x += main_image.delta_x
            #y += main_image.delta_y
            point = self.canvas.create_oval(x-1, y-1, x+1, y+1, outline=self.point_color, fill=self.point_color)
            self.points.append((x, y, point))

    def find_nearest_point(self, x, y):
        nearest_point = None
        min_distance = float('inf')
        for px, py, point_id in self.points:
            distance = math.sqrt((px - x) ** 2 + (py - y) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = (px, py, point_id)
        return nearest_point

    def set_origin(self, event):
        self.origin_x = event.x
        self.origin_y = event.y
        if self.origin_id is not None:
            self.canvas.delete(self.origin_id)
        self.origin_id = self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, outline='purple', fill='purple')
        self.select_point_from_origin()

    def deselect_point(self):
        if self.selected_point is not None:
            self.canvas.itemconfig(self.selected_point[2], outline=self.point_color, fill=self.point_color)
            self.selected_point = None

    def select_point_from_origin(self):
        self.deselect_point()
        nearest_point = self.find_nearest_point(self.origin_x, self.origin_y)
        if nearest_point:
            self.selected_point = nearest_point
            self.canvas.itemconfig(nearest_point[2], outline='red', fill='red')

    def remove_point(self, event=None):
        if self.selected_point:
            self.canvas.delete(self.selected_point[2])
            self.points.remove(self.selected_point)
            self.df = self.df[(self.df['pixel_x'] != self.selected_point[0]) | (self.df['pixel_y'] != self.selected_point[1])]
            self.selected_point = None
            self.select_point_from_origin()

    def update(self, mydf):
        self.df = mydf
        if hasattr(self, 'points'):
            self.clear()
        self.points = []
        self.origin_id = None
        self.origin_x = None
        self.origin_y = None
        self.selected_point = None
        self.plot_points()
    
    def open_window(self):
        if self.checkwindow():
            return
        
        self.window = tk.Toplevel(root)
        self.window.title('Settings')
        self.window.geometry('180x180')
        self.window.attributes('-topmost', 'true')
        self.update(single.df)
        
        button = tk.Button(self.window, text = "Exit", bg = "red", font = "Verdana 11", command = self.exit)
        button.pack(side=tk.TOP)
        self.window.bind('<Escape>', self.exit)
        tip = Hovertip(button,'Bind: Esc')
    
        button = tk.Button(self.window, text = "Ok", bg = "green", font = "Verdana 11", command = self.save_and_exit)
        button.pack(side=tk.TOP)
        self.window.bind('<Return>', self.save_and_exit)
        tip = Hovertip(button,'Bind: Enter')

    def clear(self):
        for px, py, point_id in self.points:
            self.canvas.delete(point_id)
        if self.origin_id is not None:
            self.canvas.delete(self.origin_id)
    
    def exit(self, event=None):
        self.clear()
        self.window.destroy()
    
    def save_and_exit(self, event=None):
        single.df = self.df
        single.update_plot()
        self.exit()

class SelectHandler:
    def __init__(self):
        self.running = False
        self.process = None
        self.path = None
        self.last_output = ''
        self.last_output_time = 404
    
    def parse_errors(self):
        linelist = self.last_output.splitlines()
        for line in reversed(linelist):
            if len(line) > 3:
                return line

    def parse_iterline(self, myline):
        myline = collapse_spaces(myline, 3)
        myline = myline[1:]
        numbers = []
        for num in myline.split(' '):
            if '*' in num:#If Select printed **** instead of a number
                numbers.append(None)
                continue
            try:
                number = int(num)
            except:
                number = float(num)
            numbers.append(number)
        
        for i in [4, 2, 1]: #Remove X0, Z0 and q4: not needed
            numbers.pop(i)
        
        names = ['Iteration', 'Apex radius', 'ErrMin', 'sigma/ro', 'Contact angle', 'Contact d.', 'Drop Volume', 'Drop Surface', 'Point count']
        mydict = {}
        for i in range(len(names)):
            mydict[names[i]] = numbers[i]
        return mydict

    def parse_frame(self, mystr):
        mystr = mystr.replace('#    ', '')
        mystr = collapse_spaces(mystr, 4)
        linelist = mystr.splitlines()
        linelist.pop(4) #"Contact angle for sphere - plane approximation"
        linelist.pop(1) #"Method with horizontal line"
        numbers = []
        for line in linelist:
            for comb in line.split(' '):
                try:
                    numbers.append(float(comb))
                except:
                    if '*' in comb:#If Select printed **** instead of a number
                        numbers.append(None)
        numbers.pop(3)# Remove 2 "Z" values
        numbers.pop(2)
        names = ['ErrMin', 'Contact angle', 'sigma/ro', 'Contact d.', 'Drop Surface', 'Drop Volume']
        mydict = {}
        for i in range(len(names)):
            mydict[names[i]] = numbers[i]
        return mydict

    def parse_output(self):
        pos = self.last_output.find('ErrMin')
        if pos > 0: #If the program finished
            frame = self.last_output[pos:]
            linelist = self.last_output[:pos].splitlines()
            dict_frame = self.parse_frame(frame)
        else:
            linelist = self.last_output.splitlines()
        
        for line in reversed(linelist):
            if 'Iter' in line: #Skipping the headers
                continue
            ln = len(line)
            if ln > 75 and ln < 85: #target length = 80
                dict_linelist = self.parse_iterline(line)
                break
        
        if pos > 0:
            val_dict = dict_linelist | dict_frame #merging
            return val_dict
        else:
            return dict_linelist

    def data_stream(self):
        self.last_output = ''
        for line in self.process.stdout:
            logger.debug(line, end='')
            self.last_output += line
            self.last_output_time = time.time()
            if 'Drop Volume:' in line:
                self.finished = True
                return

    def handler(self, exe_name):
        self.running = True
        self.finished = False
        self.interrupted = False
        if batch_operation:
            button = batch.select_button
        else:
            button = single.select_button
        #button.config(relief='sunken')
        
        self.path = copy(single.sel_path) #storing the file path locally because it can change during processing
        self.process = subprocess.Popen(settings.select_dir + exe_name, cwd=settings.select_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        stream = Thread(target=self.data_stream)
        stream.start()
        
        self.last_output_time = time.time()
        while True:
            if self.finished:
                self.process.kill()
                logger.info(f'\n{exe_name} finished with following output:')
                break
            if self.process.poll() is not None:#Select is not running
                err = self.parse_errors()
                logger.error(f'\n{exe_name} has crashed: {err}')
                break
            if self.interrupted:
                self.process.kill()
                logger.info(f'\nKilled {exe_name}')
                break
            if self.last_output_time + settings.select_waiting_time < time.time():#Timeout
                self.process.kill()
                logger.error(f'\n{exe_name} has timed out')
                break
            time.sleep(0.25)
            
        val_dict = self.parse_output()
        logger.info(f'File: {self.path}')
        val_table = self.text_table(val_dict)
        logger.info(val_table)
        self.savedata(val_dict, val_table, self.path)
        #button.config(relief='raised')
        self.running = False
    
    def interrupt(self):
        self.interrupted = True

    def run_exe(self, exe_name):
        if settings.select_in_cmd:
            logger.info(f'Launching {exe_name} in CMD')
            os.startfile(settings.select_dir + exe_name, cwd=settings.select_dir)
        else:
            logger.info(f'Launching {exe_name} as subprocess')
            sel_thread = Thread(target=self.handler, args=[exe_name])
            sel_thread.start()

    def run_select(self):
        try:
            pixel_path = os.path.join(settings.select_dir, 'pixel.dat')
            shutil.copy2(settings.out_file_name, pixel_path)
            os.remove(settings.out_file_name)
            
            self.run_exe(settings.select_name)
        except Exception as e:
            logger.error(f'Error: {e}')

    def run_select_s(self):
        try:
            with open(settings.select_dir+'pti00.dat','w') as linefile:
                linefile.write(f'0 {hline.true_y(main_image.delta_y)}')
            pts_path = os.path.join(settings.select_dir, 'pts00.dat')
            shutil.copy2(settings.out_file_name, pts_path)
            os.remove(settings.out_file_name)
            
            self.run_exe(settings.select_s_name)
        except Exception as e:
            logger.error(f'Error: {e}')

    def run_either(self, event=None):
        """Run select.exe or select_s.exe"""
        if self.running:
            logger.error('Cancelled: Select is already running!')
            return
        single.df[['x','y']].to_csv(settings.out_file_name, sep=' ', index=False, header=False)
        if settings.pendant:
            self.run_select()
        else:
            self.run_select_s()
    
    def text_table(self, mydict):
        #logger.info("{:<15} {:<10}".format('Key','Number'))
        n = 0
        string = ''
        for name, val in mydict.items():
            string += "{:<13} {:<10}".format(name, val)
            n += 1
            if n%3 == 0:
                string += '\n'
        return string
    
    def save_data_line(self, d, filepath):
        time_format = '%H:%M:%S %d.%m.%Y %a'
        cur_time_str = time.strftime(time_format)
        
        file_m_time = os.path.getmtime(filepath)
        m_ti = time.ctime(file_m_time)
        t_obj = time.strptime(m_ti)
        file_m_time_str = time.strftime(time_format, t_obj)
        
        with open('autosave//autosave.dat', 'a+') as logfile:
            if logfile.tell() == 0:
                logger.info('The save file is empty, inserting a header')
                logfile.write('sigma/ro, ErrMin, Contact angle, Contact diameter, Apex radius, Point count, Surface area, Volume, Iteration, IsPendant, File modified, File processed, File path\n')
            logfile.write(f"{d['sigma/ro']}, {d['ErrMin']}, {d['Contact angle']}, {d['Contact d.']}, {d['Apex radius']}, {d['Point count']}, {d['Drop Surface']}, {d['Drop Volume']}, {d['Iteration']}, {settings.pendant}, {file_m_time_str}, {cur_time_str}, {self.path}\n")        
    
    def save_trace_pic(self, val_table, filepath):
        trace_dir = os.path.dirname(filepath) + '/traces'
        file_name = os.path.basename(filepath)
        trace_name = os.path.splitext(file_name)[0] + '_trace.png'
        trace_path = os.path.join(trace_dir, trace_name)
        
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)
            logger.info(f'Created trace dir: {trace_dir}')
        
        logger.info(f'Saving trace file: {trace_path}')
        try:
            plot_trace.plot(settings.select_dir, settings.pendant, True, trace_path, val_table)
        except Exception as e:
            logger.error(f'Plotting error: {e}')           
    
    def savedata(self, mydict, val_table, filepath):
        """Save data associated with filepath"""
        if macro.running == False:
            save_data = settings.autosave
            save_trace = settings.autosave_plot
        else:
            save_data = True
            save_trace = macro.save_trace
            
        if save_data:
            self.save_data_line(mydict, filepath)
        
        if save_trace:
            self.save_trace_pic(val_table, filepath)

class MacroHandler(MyWindowObject):
    def __init__(self):
        self.macro = []
        self.compiled_macro = []  #macro as callable functions
        self.window = None
        self.running = False
        self.save_trace = False
    
    def log(self, action_name, *args):
        self.macro.append(action_name)
        logger.info(f'Added action to the macro: {action_name}')
    
    def save_edits(self, text_widget):
        """Save the edits back to the macro list."""
        updated_macro = text_widget.get("1.0", "end").strip().split("\n")
        self.macro = [action for action in updated_macro if action.strip()]  # Remove empty lines
        logger.info("Macro updated successfully")
        if self.window:
            self.window.destroy()
    
    def save_macro_file(self, text_widget):
        """Save macro directly to a file from the editor."""
        self.save_edits(text_widget)  # Ensure macro is updated from editor
        self.save_macro()  # Save to file
        self.checkwindow()
    
    def load_macro_file(self, text_widget):
        """Load a macro file and update the editor."""
        self.load_macro()  # Load macro from file
        # Update the text widget with the loaded macro
        text_widget.delete("1.0", "end")
        text_widget.insert("1.0", "\n".join(self.macro))
        self.checkwindow()
    
    def save_macro(self):
        """Save the macro list to a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialdir='macros/',
            title="Save Macro"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("\n".join(self.macro))
                logger.info(f"Macro saved to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save macro: {e}")
    
    def load_macro(self):
        """Load a macro list from a file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialdir='macros/',
            title="Load Macro"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.macro = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"Macro loaded from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load macro: {e}")
    
    def open_editor(self, event=None):
        """Open the macro editor window."""
        if self.checkwindow():
            return
        
        self.window = tk.Toplevel()  # Create a new window for the editor
        self.window.title("Macro Editor")
        self.window.geometry("800x500")
        
        text_widget = tk.Text(self.window, wrap="word", width=40, height=20)
        text_widget.grid(row=0, column=0, padx=10, pady=10)
        
        label = tk.Label(self.window, text='Commands:\ncanny edge detection: canny_ed\nModel delete points: mdp <model_name> <treshold value>\nRun Select: select [-drawtrace]\nSave points to file: savepoints', justify='left', font=10)
        label.grid(row=0, column=1, padx=10, pady=10)
        
        button_frame = tk.Frame(self.window)
        button_frame.grid(row=1, column=0, padx=10, pady=10)
        
        # Populate with current macro actions
        text_widget.insert("1.0", "\n".join(self.macro))
        
        tk.Button(button_frame, text="Ok", bg='green', command=lambda: self.save_edits(text_widget)).pack(pady=5)
        self.window.bind('<Return>', self.save_edits)
        tk.Button(button_frame, text="Open macro", command=lambda: self.load_macro_file(text_widget)).pack(pady=5)
        tk.Button(button_frame, text="Save macro", command=lambda: self.save_macro_file(text_widget)).pack(pady=5)
        self.window.bind('<Control-s>', self.save_macro)
        self.window.bind('<Escape>', self.window.destroy)
    
    def parse_macro_line(self, line):
        """Parse a single macro line into command and arguments."""
        parts = line.split()
        if not parts:
            return None, None
        command_name = parts[0]
        args = []
        for arg in parts[1:]:# Remaining parts are arguments
            try:
                args.append(float(arg))
            except:
                args.append(arg)
        return command_name, args
    
    def compile_macro(self):
        """
        Compile the macro into a list of callables for efficient execution.
        Raises an exception if any command is invalid.
        """
        # Command-to-function mapping
        command_map = {
            'canny_ed': single.call_canny,
            'mdp': single.model_delete_points_noplot,
            'select': self.enqueue_select,
            'savepoints': self.save_df
        }

        self.compiled_macro = []  # Clear previously compiled macro

        for line in self.macro:
            command_name, arguments = self.parse_macro_line(line)
            if command_name in command_map:
                if command_name is None:
                    continue
                func = command_map[command_name]
                try:
                    # Pre-compile the function with arguments
                    self.compiled_macro.append((func, arguments))
                except Exception as e:
                    raise ValueError(f"Error in line '{line}': {e}")
            else:
                raise ValueError(f"Unknown command: '{command_name}'")

        logger.info('Macro compiled successfully')
    
    def mainloop(self):
        self.running = True
        for file in batch.total_paths:
            logger.info(f"Processing file: {file}")
            self.open_image(file)
    
            # Execute precompiled macro
            for func, arguments in self.compiled_macro:
                if self.running == False:
                    break
                try:
                    func(*arguments)
                except Exception as e:
                    logger.error(f"Error executing macro step: {e}")
                    break
        if select.running:
            logger.info('Mainloop stopped. Waiting for Select to finish')
        while select.running:
            time.sleep(0.5)
        self.running = False
        logger.info("Macro stopped")
    
    def execute_macro(self):
        logger.info('Compiling macro')
        self.compile_macro()
        if not self.compiled_macro:
            return
        try:
            logger.info('Starting mainloop')
            macro_thread = Thread(target=self.mainloop)
            macro_thread.start()
        except Exception as e:
            logger.error(f"Error starting macro: {e}")
    
    #Functions:
    def open_image(self, mypath):
        single.sel_path = mypath
        single.process_image(False)
    
    def enqueue_select(self, tracearg=False):
        if select.running:
            logger.info('Waiting the queue')
        while select.running:
            time.sleep(0.5)
            
        self.save_trace = bool(tracearg)
        select.run_either()
    
    def save_df(self):
        df_dir = os.path.dirname(single.sel_path) + '/points'
        file_name = os.path.basename(single.sel_path)
        df_name = os.path.splitext(file_name)[0] + '_points.dat'
        df_path = os.path.join(df_dir, df_name)
        
        if not os.path.exists(df_dir):
            os.makedirs(df_dir)
            logger.info(f'Created point dir: {df_dir}')        
        single.save_df(df_path)

class Operations:
    def define_shared_elements(self, myframe):
        self.control_frame = tk.Frame(myframe)
        
        self.files_frame = tk.Frame(self.control_frame)
        self.files_frame.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        self.ed_frame = tk.Frame(self.control_frame)
        self.ed_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        self.nn_frame = tk.Frame(self.control_frame)
        self.nn_frame.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        
        self.ooo_frame = tk.Frame(self.control_frame)
        self.ooo_frame.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        settings_button = tk.Button(self.files_frame, text='Settings', font = "Verdana 11", command = settings.open_window)
        settings_button.grid(row=1, column=1, sticky='w')
        
        button = tk.Button(self.ed_frame, text = "Canny Edge detection", bg = "red", font = "Verdana 12", command = self.call_canny)
        button.pack()
        #tip = Hovertip(button,'Bind: E')
        
        #Frame for hline things
        line_frame = tk.Frame(self.ed_frame)
        line_frame.pack()
        
        button = tk.Button(line_frame, text = "Place/hide line", font = "Verdana 9", command = hline.toggle)
        button.pack(side=tk.LEFT)
        tip = Hovertip(button,'Bind: L')
        
        button = tk.Button(line_frame, text = "￪", font = "Verdana 9", command = hline.move_up)
        button.pack(side=tk.LEFT)
        tip = Hovertip(button,'Bind: Up')
        
        button = tk.Button(line_frame, text = "￬", font = "Verdana 9", command = hline.move_down)
        button.pack(side=tk.LEFT)
        tip = Hovertip(button,'Bind: Down')
        
        # Frame for "delete points with" button and dropdown menu
        delete_points_frame = tk.Frame(self.nn_frame)
        delete_points_frame.pack()
        
        button = tk.Button(delete_points_frame, text="Delete points", font="Verdana 10", command = self.model_delete_points)
        button.pack(side=tk.LEFT)
        tip = Hovertip(button,'Bind: Clrl+D')
        
        label = tk.Label(delete_points_frame, text="with", font="Verdana 10")
        label.pack(side=tk.LEFT)
        
        model_selector = tk.OptionMenu(delete_points_frame, var_model, *model_list)
        model_selector.pack(side=tk.LEFT)
        
        # Frame for Threshold label and slider
        threshold_frame = tk.Frame(self.nn_frame)
        threshold_frame.pack()
        
        slider_label = tk.Label(threshold_frame, text="Threshold:", font="Verdana 11")
        slider_label.pack(side=tk.LEFT)
        
        slider = tk.Scale(threshold_frame, from_=0, to=1, resolution=0.01, tickinterval=0.5, length=150, sliderlength=20, orient='horizontal', variable=settings.var_treshold, command = self.on_slider_change)
        slider.set(settings.default_treshold)
        slider.pack(side=tk.LEFT)        

class SingleOperation(Operations):
    def __init__(self, myframe):
        self.sel_path = None
        self.cv_image = None
        self.canny_image = None
        self.df = None
        self.eval_df = None
        
        self.define_shared_elements(myframe)
        
        button = tk.Button(self.files_frame, text = "Open file", bg = "light blue", font = "Verdana 13", cursor='hand2', command = self.choose_file)
        button.grid(row=0, column=0, sticky='w')
        tip = Hovertip(button,'Bind: C')
        
        button = tk.Button(self.files_frame, text='Save file', font = "Verdana 11", command = self.save_points)
        button.grid(row=1, column=0, sticky='w')
        tip = Hovertip(button,'Bind: S')
        
        mode_button = tk.Button(self.files_frame, text='to Batch Mode', font = "Verdana 11", command = switch_operation_mode)
        mode_button.grid(row=0, column=1, sticky='w')
        
        #Frame for Select things
        select_frame = tk.Frame(self.ooo_frame)
        select_frame.pack()
        
        button = tk.Button(select_frame, text = "Yoink points from ooo", font = "Verdana 11", command = self.import_points_from_ooo)
        button.grid(row=0, column=1, sticky='w')
        
        button = tk.Button(select_frame, text = "Yeet points to ooo", font = "Verdana 11", command = self.inject_df_to_ooo)
        button.grid(row=1, column=1, sticky='w')
        
        self.select_button = tk.Button(select_frame, text = "Run Select", font = "Verdana 11", command = select.run_either)
        self.select_button.grid(row=0, column=0, sticky='w')
        tip = Hovertip(self.select_button,'Bind: R')
        
        trace_button = tk.Button(select_frame, text = "Plot trace", font = "Verdana 11", command = plot_ooo_trace)
        trace_button.grid(row=1, column=0, sticky='w')
        tip = Hovertip(trace_button,'Bind: T')
        
        checkbox = tk.Checkbutton(select_frame, text='Pendant', variable=settings.var_pendant, onvalue=True, offvalue=False, command=settings.update)
        checkbox.grid(row=2, column=0, sticky='w')
        
        checkbox = tk.Checkbutton(select_frame, text='Run Select in CMD', variable=settings.var_select_in_cmd, onvalue=True, offvalue=False, command=settings.update)
        checkbox.grid(row=2, column=1, sticky='w')
        
        #button = tk.Button(self.nn_frame, text = "Run TF_fit (x)", font = "Verdana 10", command = run_fit)
        #button.pack()
        
        button = tk.Button(self.nn_frame, text="Delete manually", font="Verdana 10", command = point_deleter.open_window)
        button.pack()
    
    def load_df_raw(self):
        """Only x and y columns. Compatible with ooo files"""
        try:
            self.df = pd.read_csv(self.sel_path, delimiter=' ', names=['x','y'])
            self.eval_df = None #Reset existing eval_df
            if 'pixel_x' not in self.df.columns:
                self.df = df_restore_pixel_values(self.df)
        except Exception as e:
            logger.error(f'Error reading file: {e}')
    
    def load_df(self):
        """load df with columns: pixel_x, pixel_y, x, y"""
        try:
            self.df = pd.read_csv(self.sel_path, delimiter=' ')
            self.eval_df = None #Reset existing eval_df
            if 'pixel_x' not in self.df.columns:
                self.load_df_raw()
        except Exception as e:
            logger.error(f'Error reading file: {e}')
    
    def model_delete_points_noplot(self, model_name, treshold):
        self.df, self.eval_df = model_delete_points_modular(self.df, model_name, treshold)
    
    def model_delete_points(self, event=None):
        model_name = var_model.get()
        self.df, self.eval_df = model_delete_points_modular(self.df, model_name, settings.treshold)
        
        plt.scatter(self.eval_df['x'], self.eval_df['y'], s=0.5, c=self.eval_df['weight'])
        plt.show()
        self.update_plot()
    
    def on_slider_change(self, strvalue):
        settings.update()
        logger.debug(f'Slider value: {strvalue}')
        if self.eval_df is not None:
            self.df = cut_df_by_treshold(self.eval_df, settings.treshold)
            self.update_plot()
    
    def import_points_from_ooo(self):
        self.sel_path = os.path.join(settings.ooo_temp_dir, 'ooo.dat')
        self.load_df_raw()
        self.update_plot()
        self.clear_images()
        logger.info(f'Yoinked file: {self.sel_path}')
    
    def inject_df_to_ooo(self):
        logger.info('Injecting points')
        self.df[['x','y']].to_csv(settings.out_file_name, sep=' ', index=False, header=False)
        target_path = os.path.join(settings.ooo_temp_dir, 'ooo.dat')
        shutil.copy2(settings.out_file_name, target_path)
        os.remove(settings.out_file_name)
        with open(settings.ooo_temp_dir+'ooo.dat_plus', 'w') as colordata: #clear color data
            colordata.write('')
    
    def draw_plot(self, color='blue'):
        global graph_canvas, plot, mpltoolbar
        plot.clear()
        #plot.yaxis.set_major_formatter(y_ticker)
        plot.set_title(self.sel_path)
        plot.title.set_fontsize(8)
        
        plot.scatter(self.df['x'], self.df['y'], color=color, s=0.2)
        
        if hline.visible():
            pos = hline.true_y(main_image.delta_y)
            plot.axhline(y=pos, color='lime', linestyle='-')
            plot.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            plot.set_yticks([round(pos,1)]) #add y tick to hline
        graph_canvas.draw()
        mpltoolbar.update()    
    
    def update_plot(self):
        if self.df is not None:
            self.draw_plot('blue')
    
    def crop_update(self):
        if self.eval_df is None: #restrict modifications if eval_df is present
            cropped_canny_image = crop_canny_image(self.canny_image)
            self.df = df_from_canny_image(cropped_canny_image)
            self.update_plot()
    
    def call_canny(self, event=None):
        self.eval_df = None #Reset existing eval_df
        self.canny_image = canny_edge_detection(self.cv_image)
        self.crop_update()
        
        if settings.show_canny_image:
            color = (0, 127, 255)
            overlay_image.set(overlay_from_canny_image(self.canny_image, color))
        else:
            overlay_image.clear()
    
    def process_image(self, apply_settings=True):
        self.cv_image, pil_image = load_image(self.sel_path)
        main_image.set(pil_image)
        if apply_settings:
            if settings.do_canny_on_selection:
                self.call_canny()
                #if var_do_ai_on_selection.get():
                    #self.model_delete_points()
    
    def choose_file(self, event=None):
        path = os.path.join(filedialog.askopenfilename(initialdir=settings.default_path, title="Select file",
            filetypes=(("all files", "*.*"),("dat files", "*.dat"),("bmp files", "*.bmp"))))
        if path:
            self.sel_path = path
            logger.info(f'Selected file: {self.sel_path}')
            self.clear_images()
            if self.sel_path.endswith('.dat'):
                self.load_df()
                self.update_plot()
            else:
                self.process_image()
    
    def save_df(self, mypath):
        self.df.to_csv(mypath, sep=' ', index=False, header=True)
        logger.info(f'Saved points: {mypath}')
    
    def save_points(self, event=None):
        path = filedialog.asksaveasfilename(initialdir=settings.default_path, title="Save file",
            filetypes=(("dat files", "*.dat"),("all files", "*.*")))
        if path == '':
            return
        if path.endswith('.dat') == False:
            path += '.dat'
        self.save_df(path)
    
    def clear_images(self):
        self.cv_image = None
        self.canny_image = None
        main_image.clear()
        overlay_image.clear()
        lens_image.clear()

class BatchOperation(Operations):
    def __init__(self, myframe):
        self.colors = [(0, 127, 255), (0, 255, 255), (255, 0, 255), (127, 0, 255), (0, 0, 255),
                  (127, 127, 255), (127, 255, 127), (255, 0, 0), (255, 255, 0), (255, 127, 0)]
        self.colors01 = [(0, 0.5, 1), (0, 1, 1), (1, 0, 1), (0.5, 0, 1), (0, 0, 1),
                  (0.5, 0.5, 1), (0.5, 1, 0.5), (1, 0, 0), (1, 1, 0), (1, 0.5, 0)]
        self.cv_images = []
        self.canny_images = []
        self.dfs = []
        self.eval_dfs = []
        self.sel_dir = None
        self.prev_paths = []
        self.total_paths = []
        self.operation_index = 0
        
        self.define_shared_elements(myframe)
        
        button = tk.Button(self.files_frame, text = "Browse dir", bg = "light blue", font = "Verdana 13", cursor='hand2', command = self.choose_dir)
        button.grid(row=0, column=0, sticky='w')
             
        button = tk.Button(self.files_frame, text='Add Save', bg = "red", font = "Verdana 11", command = lambda: macro.log('savepoints'))
        button.grid(row=1, column=0, sticky='w')
        
        mode_button = tk.Button(self.files_frame, text='to Normal Mode', font = "Verdana 11", command = switch_operation_mode)
        mode_button.grid(row=0, column=1, sticky='w')
        
        #Frame for Select things
        select_frame = tk.Frame(self.ooo_frame)
        select_frame.pack()
        
        self.select_button = tk.Button(select_frame, text = "Run Select", font = "Verdana 11", command = self.run_select)
        self.select_button.grid(row=0, column=0, sticky='w')
        
        trace_button = tk.Button(select_frame, text = "Plot trace", font = "Verdana 11", command = plot_ooo_trace)
        trace_button.grid(row=1, column=0, sticky='w')
        tip = Hovertip(trace_button,'Bind: T')
        
        macro_editor_button = tk.Button(select_frame, text = "Macro Editor", font = "Verdana 11", command = macro.open_editor)
        macro_editor_button.grid(row=0, column=1, sticky='w')
        
        macro_button = tk.Button(select_frame, text = "Run Macro", font = "Verdana 11", command = macro.execute_macro)
        macro_button.grid(row=1, column=1, sticky='w')
        
        checkbox = tk.Checkbutton(select_frame, text='Pendant', variable=settings.var_pendant, onvalue=True, offvalue=False, command=settings.update)
        checkbox.grid(row=2, column=0, sticky='w')
        
        checkbox = tk.Checkbutton(select_frame, text='Run Select in CMD', variable=settings.var_select_in_cmd, onvalue=True, offvalue=False, command=settings.update)
        checkbox.grid(row=2, column=1, sticky='w')        
        
        #button = tk.Button(self.nn_frame, text = "Run TF_fit (x)", font = "Verdana 10", command = run_fit)
        #button.pack()
    
    #def appear(self):
        #self.control_frame.pack()
    
    #def unpack(self):
        #self.control_frame.pack_forget()
    
    def run_select(self, event=None):
        logger.info(f'Processing file [{self.operation_index+1}/{settings.batch_prev_count}] {self.prev_paths[self.operation_index]}')
        single.df = self.dfs[self.operation_index]
        select.run_either()
        self.operation_index += 1
        if self.operation_index >= settings.batch_prev_count:
            self.operation_index = 0
    
    def on_slider_change(self, strvalue):
        settings.update()
        logger.debug(f'Slider value: {strvalue}')
        if len(self.eval_dfs) > 0:
            self.dfs = []
            for myeval_df in self.eval_dfs:
                self.dfs.append(cut_df_by_treshold(myeval_df, settings.treshold))
            self.update_plot()
    
    def model_delete_points(self, event=None):
        try:
            model_name = var_model.get()
            logger.info(f'Deleting points with {model_name}')
            
            self.eval_dfs = []
            newdfs = []
            for ndf in self.dfs:
                eval_df = model_evaluate(ndf, model_name)
                self.eval_dfs.append(eval_df)
                newdfs.append(cut_df_by_treshold(eval_df, settings.treshold))
                plt.scatter(eval_df['x'], eval_df['y'], s=0.5, c=eval_df['weight'])
                
            logger.info(f'Deleted points')
            self.dfs = newdfs
            plt.show()
            self.update_plot()
        except Exception as e:
            logger.error(f'Error: {e}')
    
    def draw_plot(self, colors):
        global graph_canvas, plot, mpltoolbar
        plot.clear()
        #plot.yaxis.set_major_formatter(y_ticker)
        plot.set_title(self.sel_dir)
        plot.title.set_fontsize(8)
        
        for i in range(len(self.dfs)):
            plot.scatter(self.dfs[i]['x'], self.dfs[i]['y'], color=colors[i], s=0.2)
        
        if hline.visible():
            pos = hline.true_y(main_image.delta_y)
            plot.axhline(y=pos, color='lime', linestyle='-')
            plot.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            plot.set_yticks([round(pos,1)]) #add y tick to hline
        graph_canvas.draw()
        mpltoolbar.update()    
    
    def update_plot(self):
        if len(self.dfs) > 0:
            self.draw_plot(self.colors01)    
    
    def crop_update(self):
        if len(self.eval_dfs) == 0:
            self.dfs = []
            for image in self.canny_images:
                cropped_canny_image = crop_canny_image(image)
                self.dfs.append(df_from_canny_image(cropped_canny_image))      
            self.update_plot()
    
    def call_canny(self, event=None):
        self.canny_images = []
        self.dfs = []
        self.eval_dfs = [] #Reset existing eval_dfs
        for cv_image in self.cv_images:
            canny_image = canny_edge_detection(cv_image)
            self.canny_images.append(canny_image)
            cropped_canny_image = crop_canny_image(canny_image)
        self.crop_update()
        
        if settings.show_canny_image:
            ov_ov = overlay_canny_images(self.canny_images)#overlayed overlay images
            overlay_image.set(ov_ov)
        else:
            overlay_image.clear()
        
    def process_batch(self):
        logger.info(f'Batch operation directory: {self.sel_dir}')
        files = [os.path.join(self.sel_dir, f) for f in os.listdir(self.sel_dir) if f.endswith('.bmp')]
        self.total_paths = sorted(files)
    
        indices = preview_file_indices(len(self.total_paths), settings.batch_prev_count)
        self.prev_paths = [self.total_paths[i] for i in indices]
        
        self.cv_images = []
        pil_images = []
        for file in self.prev_paths:
            cv_image, pil_image = load_image(file)
            self.cv_images.append(cv_image)
            pil_images.append(pil_image)
        ov_image = overlay_images(pil_images)
        main_image.set(ov_image)
    
        if settings.do_canny_on_selection:
            self.call_canny()
    
    def choose_dir(self, event=None):
        sel_dir = filedialog.askdirectory(initialdir=settings.default_path, title="Select dir")
        if sel_dir == '':
            return
        self.sel_dir = sel_dir
        single.sel_path = sel_dir#for plot to be normally working
        self.operation_index = 0 #reset
        self.process_batch()
    
def collapse_spaces(mystr, n=3):
    for m in range(n):
        mystr = mystr.replace('  ', ' ')
    return mystr

def load_model_list():
    models = os.listdir(settings.model_path)
    return models

def y_ticker(x, pos):
    return f'{x*1e-3:1.1f} mm'

def cut_df_by_treshold(myeval_df, treshold):
    if 'pixel_x' in myeval_df.columns:
        newdf = myeval_df[myeval_df['weight'] >= treshold][['pixel_x','pixel_y','x','y']]
    else:
        newdf = myeval_df[myeval_df['weight'] >= treshold][['x','y']]
    return newdf

def model_evaluate(mydf, model_name):
    path = os.path.join(settings.model_path + model_name)
    evaluated_df = mydf.copy()
    evaluated_df['weight'] = inference(path, model_name, mydf, settings.grid_d)
    return evaluated_df

def model_delete_points_modular(mydf, model_name, treshold):
    try:
        logger.info(f'Deleting points with {model_name}')
        
        eval_df = model_evaluate(mydf, model_name)
        
        points_before = mydf.shape[0]
        newdf = cut_df_by_treshold(eval_df, treshold)
        points_after = newdf.shape[0]
        logger.info(f'Deleted {points_before - points_after} points')
        return newdf, eval_df
    except Exception as e:
        logger.error(f'Error: {e}')

def plot_ooo_trace(event=None):
    logger.debug(f'Drawing fit from {settings.select_dir}')
    try:
        plot_trace.plot(settings.select_dir, settings.pendant, True)
    except Exception as e:
        logger.error(f'Plotting error: {e}')

def run_fit():
    logger.info('Fitting using TensorFlow...')
    try:
        error, opt_apex_radius, opt_sigma, opt_x_offset, opt_y_offset = fit_polar.process_df(df, settings.pendant, settings.fit_epochs)
        logger.info(f'Sigma = {opt_sigma}')
        logger.info(f'Apex_radius = {opt_apex_radius}')
        logger.info(f'Error = {error}')
    except Exception as e:
        logger.error(f'Error: {e}')

def df_from_canny_image(myimg):
    if myimg is None:
        return None
    a = np.argwhere(myimg == 255)

    df = pd.DataFrame(a,columns=['pixel_y','pixel_x'])#transpose
    df = df[['pixel_x','pixel_y']]
    df[['x','y']] = settings.grid_d*df[['pixel_x','pixel_y']]
    if settings.subpixel_diagonals:
        df = df_move_corners(df, settings.grid_d)
    df['y'] *= -1
    return df

def crop_canny_image(canny_image):
    """Not exactly crop, but whatever"""
    if canny_image is None:
        return None
    
    if rect_pos.visible():
        px1, px2, py1, py2 = rect_pos.image_coords(main_image.delta_x, main_image.delta_y)
        img_array = np.zeros(shape=canny_image.shape, dtype=canny_image.dtype)
        img_array[py1:py2, px1:px2] = canny_image[py1:py2, px1:px2]
    else:
        img_array = canny_image.copy()
        
    if rect_neg.visible():
        nx1, nx2, ny1, ny2 = rect_neg.image_coords(main_image.delta_x, main_image.delta_y)
        img_array[ny1:ny2, nx1:nx2] = 0
    
    if settings.hline_crop and hline.visible():
        max_x, max_y = settings.image_res
        y = hline.canv_y
        img_array[y:max_y, 0:max_x] = 0        
    
    return img_array

def overlay_from_canny_image(myimg, rgb_color):
    rgba_color = rgb_color + (255,)
    pil_canny_image = Image.fromarray(myimg).convert('L')
    mask = Image.new('L', pil_canny_image.size, 0)
    mask.paste(pil_canny_image, (0, 0))
    #mask = ImageEnhance.Brightness(mask).enhance(2)
    transparent_image = Image.new('RGBA', main_image.image.size)
    transparent_image.paste(rgba_color, (0, 0), mask)
    return transparent_image

def canny_edge_detection(cv_image):
    logger.debug('Running Canny edge detection')
  
    # Applying the Canny Edge filter  
    # with Aperture Size and L2Gradient 
    canny_image = cv2.Canny(cv_image,
                int(settings.t_lower),
                int(settings.t_upper),
                apertureSize = int(settings.aperture_size),
                L2gradient = settings.l2gradient)
    return canny_image

def preview_file_indices(total, n):
    """returns n equally spaced file indexes"""
    indices = [0]
    if total > 2:
        step = total//(n-1)
        indices.extend([step*(n+1) for n in range(n-2)])
    indices.append(total - 1)
    return indices

def overlay_images(imglist):
    alpha = 255//len(imglist)
    base_image = imglist[0].convert('RGBA')
    for img in imglist[1:]:
        img = img.convert('RGBA')
        img.putalpha(alpha)
        base_image = Image.alpha_composite(base_image, img)
    return base_image

def overlay_canny_images(imglist):
    """10 images max"""
    overlays = []
    for i in range(len(imglist)):
        overlays.append(overlay_from_canny_image(imglist[i], batch.colors[i]))
    
    base_image = overlays[0]
    for img in overlays[1:]:
        base_image = Image.alpha_composite(base_image, img)
    return base_image 

def load_image(mypath):
    try:
        pil_image = Image.open(mypath)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) #Convert from PIL to cv2 image
        #if settings.image_downscale != 1:
            #pil_image = pil_image.resize((frame_w, frame_h), Image.LANCZOS)
        return cv_image, pil_image
    except Exception as e:
        logger.error(f"Error loading image: {e}")

def df_restore_pixel_values(mydf):
    mydf['pixel_x'] = mydf['x']/settings.grid_d
    mydf['pixel_y'] = -mydf['y']/settings.grid_d
    mydf['pixel_x'] = mydf['pixel_x'].astype(int)
    mydf['pixel_y'] = mydf['pixel_y'].astype(int)
    return mydf

def df_move_corners(matrix_df, grid_d):
    """finds and moves corner points (to subpixels)"""
    move_amount = grid_d/4
    
    df_px = matrix_df[['pixel_x','pixel_y']]
    df_nx = matrix_df[['pixel_x','pixel_y']]
    df_py = matrix_df[['pixel_x','pixel_y']]
    df_ny = matrix_df[['pixel_x','pixel_y']]
    
    df_px['pixel_x'] += 1
    df_nx['pixel_x'] -= 1
    df_py['pixel_y'] += 1
    df_ny['pixel_y'] -= 1
    
    px_sect = pd.merge(matrix_df, df_px, how='inner', on=['pixel_x','pixel_y'], suffixes=('_x', None))
    nx_sect = pd.merge(matrix_df, df_nx, how='inner', on=['pixel_x','pixel_y'], suffixes=('_x', None))
    py_sect = pd.merge(matrix_df, df_py, how='inner', on=['pixel_x','pixel_y'], suffixes=('_x', None))
    ny_sect = pd.merge(matrix_df, df_ny, how='inner', on=['pixel_x','pixel_y'], suffixes=('_x', None))
    
    corners_pxpy = pd.merge(px_sect, py_sect, how='inner', on=['pixel_x','pixel_y'], suffixes=('_x', None))[['pixel_x','pixel_y']]
    corners_pxny = pd.merge(px_sect, ny_sect, how='inner', on=['pixel_x','pixel_y'], suffixes=('_x', None))[['pixel_x','pixel_y']]
    corners_nxpy = pd.merge(nx_sect, py_sect, how='inner', on=['pixel_x','pixel_y'], suffixes=('_x', None))[['pixel_x','pixel_y']]
    corners_nxny = pd.merge(nx_sect, ny_sect, how='inner', on=['pixel_x','pixel_y'], suffixes=('_x', None))[['pixel_x','pixel_y']]
    
    corners_pxpy['x'] = corners_pxpy['pixel_x']*grid_d - move_amount
    corners_pxpy['y'] = corners_pxpy['pixel_y']*grid_d - move_amount
    corners_pxny['x'] = corners_pxny['pixel_x']*grid_d - move_amount
    corners_pxny['y'] = corners_pxny['pixel_y']*grid_d + move_amount
    corners_nxpy['x'] = corners_nxpy['pixel_x']*grid_d + move_amount
    corners_nxpy['y'] = corners_nxpy['pixel_y']*grid_d - move_amount
    corners_nxny['x'] = corners_nxny['pixel_x']*grid_d + move_amount
    corners_nxny['y'] = corners_nxny['pixel_y']*grid_d + move_amount
    
    allcorners = pd.concat([corners_pxpy, corners_pxny, corners_nxpy, corners_nxny], ignore_index=True)
    
    matrix_df.set_index(['pixel_x', 'pixel_y'], inplace=True)
    allcorners.set_index(['pixel_x', 'pixel_y'], inplace=True)
    
    # Merge the DataFrames by index
    merged_df = matrix_df.merge(allcorners, left_index=True, right_index=True, how='outer', suffixes=('_matrix', '_allcorners'))
    
    # Update x and y columns from allcorners where matching pixel_x and pixel_y
    merged_df['x'] = merged_df['x_allcorners'].combine_first(merged_df['x_matrix'])
    merged_df['y'] = merged_df['y_allcorners'].combine_first(merged_df['y_matrix'])
    
    # Reset index to convert back to the original structure
    final_df = merged_df.reset_index()[['pixel_x', 'pixel_y', 'x', 'y']]
    return final_df

def crop_lens(event):
    x,y = event.x, event.y
    lens_hline.hover_update(y)
    x += main_image.delta_x
    y += main_image.delta_y
    if main_image.image is not None:
        tmp = main_image.image.crop((x-hz_x, y-hz_y, x+hz_x, y+hz_y))
        zimg = tmp.resize((settings.lens_size[0], settings.lens_size[1]), Image.Resampling.NEAREST)
        lens_image.set(zimg)

def clip(mycoord, mymin, mymax):
    if mycoord < mymin:
        return mymin
    if mycoord > mymax:
        return mymax
    return mycoord

def drag_widget(event):
    xpos = clip(event.x_root, 0, frame_w)
    ypos = clip(event.y_root, 0, frame_h-10)
    logger.debug(f'Widget coords: {xpos}, {ypos}')
    event.widget.place(x=xpos, y=ypos, anchor=tk.CENTER)

def toggle_fullscreen(event):
    global root
    settings.fullscreen = not settings.fullscreen
    root.attributes("-fullscreen", settings.fullscreen)

def validate_numeric_input(value: str) -> bool:
    return value.isdigit() or value == ''

def switch_operation_mode():
    global batch_operation
    batch_operation = not batch_operation
    if batch_operation:
        single.control_frame.pack_forget()
        batch.control_frame.pack(side=tk.TOP, fill='x')
        batch.update_plot()
    else:
        batch.control_frame.pack_forget()
        single.control_frame.pack(side=tk.TOP, fill='x')
        single.update_plot()


root = tk.Tk()
root.geometry("1800x1200")
root.title('DanPySelect Sigma')
if get_os() == 'Windows':
    root.state('zoomed')
else:
    root.attributes('-zoomed', True) #If on linux
root.bind("<F11>", toggle_fullscreen)


#Setting variables
settings = SettingsHandler()

#Calculating constants
frame_w = settings.image_res[0]//settings.image_downscale
frame_h = settings.image_res[1]//settings.image_downscale
initial_line_y = frame_h - settings.initial_hline_height
hz_x = int(settings.lens_size[0]/settings.lens_zoom/2)
hz_y = int(settings.lens_size[1]/settings.lens_zoom/2)

canvas_frame = tk.Frame(root)
canvas_frame.pack(side = tk.LEFT)

base_frame = tk.Frame(root)
base_frame.pack(side=tk.TOP)


image_canvas = tk.Canvas(canvas_frame, bg='white', bd=0, relief='ridge', highlightthickness=0, width = frame_w, height = frame_h)
main_image = MainCanvasImage(image_canvas, [frame_w, frame_h])
overlay_image = CanvasImage(image_canvas, [frame_w, frame_h])
image_canvas.pack(fill='both', side=tk.LEFT)
rect_pos = RectangleSelection(image_canvas, True)
image_canvas.bind('<Button 1>', rect_pos.on_click)
image_canvas.bind('<B1-Motion>', rect_pos.on_drag)
image_canvas.bind('<ButtonRelease-1>', rect_pos.on_release)
rect_neg = RectangleSelection(image_canvas, False)
image_canvas.bind('<Button 3>', rect_neg.on_click)
image_canvas.bind('<B3-Motion>', rect_neg.on_drag)
image_canvas.bind('<ButtonRelease-3>', rect_neg.on_release)
image_canvas.bind("<Motion>", crop_lens)

lens_canvas = tk.Canvas(image_canvas, bg='white', highlightthickness=1, width = settings.lens_size[0], height = settings.lens_size[1], cursor='boat')
lens_canvas.place(x=0, y=0, anchor=tk.NW)
lens_canvas.bind("<B1-Motion>", drag_widget)
lens_image = CanvasImage(lens_canvas, settings.lens_size)
lens_hline = LensHorizontalLine(lens_canvas, 0)

figure = Figure(figsize=(5.9, 5.9), dpi=100)
graph_canvas = FigureCanvasTkAgg(figure, base_frame)
mpltoolbar = NavigationToolbar2Tk(graph_canvas, base_frame)
graph_canvas.get_tk_widget().pack(side=tk.BOTTOM)
plot = figure.add_subplot(111)

hline = MainHorizontalLine(image_canvas, initial_line_y)
image_canvas.bind('<Control-Button-1>', hline.set_y)
root.bind('<l>', hline.toggle)

model_list = load_model_list()
var_model = tk.StringVar(base_frame)
if settings.default_model in model_list:
    var_model.set(settings.default_model)
else:
    print(f'{settings.default_model} is not in the list')


#single operations
select = SelectHandler()
point_deleter = PointSelector(root, image_canvas)
single = SingleOperation(base_frame)
single.control_frame.pack(side=tk.TOP)


#batch things
macro = MacroHandler()
batch = BatchOperation(base_frame)

   
root.bind('<c>', single.choose_file)
root.bind('<s>', single.save_points)
root.bind('<e>', single.call_canny)
root.bind('<r>', select.run_either)
root.bind('<t>', plot_ooo_trace)
root.bind('<Control-d>', single.model_delete_points)
root.bind('<Down>', hline.move_down)
root.bind('<Up>', hline.move_up)


#logger
st = tk.scrolledtext.ScrolledText(root, state='disabled')
st.configure(font='Consolas 11')
st.pack(side=tk.BOTTOM)

# Create textLogger
text_handler = logm.TextHandler(st)

# Add the handler to logger
logger = logging.getLogger()
logger.addHandler(text_handler)

logger.info('DanPySelect V4.2.0 (03.12.2024)\n')

root.mainloop()

