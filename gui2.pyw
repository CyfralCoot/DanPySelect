# -*- coding: utf-8 -*-
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

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import threading
import configparser

from run_deleter_cluster import *
import fit_polar
import logging_module as logm
import plot_trace

import logging
logging.basicConfig(level=logging.INFO)

# Read configuration file
config = configparser.ConfigParser()
with open('config.ini', 'r', encoding='utf-8') as f:
    config.read_file(f)

#Initial settings
fullscreen = config.getboolean('InitialSettings', 'fullscreen')
image_res = tuple(map(int, config.get('InitialSettings', 'image_res').split(',')))
image_downscale = config.getint('InitialSettings', 'image_downscale')
initial_hline_height = config.getint('InitialSettings', 'initial_hline_height')
lens_size = tuple(map(int, config.get('InitialSettings', 'lens_size').split(',')))
lens_zoom = config.getint('InitialSettings', 'lens_zoom')
default_path = config.get('InitialSettings', 'default_path')
out_file_name = config.get('InitialSettings', 'out_file_name')
model_path = config.get('InitialSettings', 'model_path')
default_model = config.get('InitialSettings', 'default_model')
default_treshold = config.getfloat('InitialSettings', 'default_treshold')
fit_epochs = config.getint('InitialSettings', 'fit_epochs')
do_canny_on_selection = config.getboolean('InitialSettings', 'do_canny_on_selection')
show_canny_image = config.getboolean('InitialSettings', 'show_canny_image')
grid_d = config.getfloat('InitialSettings', 'grid_d')
select_dir = config.get('InitialSettings', 'select_dir')
ooo_temp_dir = config.get('InitialSettings', 'ooo_temp_dir')

#Canny parameters:
t_lower = config.getint('CannyParameters', 't_lower')
t_upper = config.getint('CannyParameters', 't_upper')
aperture_size = config.getint('CannyParameters', 'aperture_size')
L2Gradient = config.getboolean('CannyParameters', 'L2Gradient')
subpixel_diagonals = config.getboolean('CannyParameters', 'subpixel_diagonals')

#Calculating constants
frame_w = image_res[0]//image_downscale
frame_h = image_res[1]//image_downscale
initial_line_y = frame_h - initial_hline_height
hz_x = int(lens_size[0]/lens_zoom/2)
hz_y = int(lens_size[1]/lens_zoom/2)

#Technical
zimg_id = None
pil_image = None
cv_image = None
canny_image = None
df = None
eval_df = None
settings_window = None

class my_canvas_object():
    def __init__(self):
        self.id = None
    
    def visible(self) -> bool:
        """Returns whether the rectangle is present or not"""
        return self.id is not None

class rectangle_selection(my_canvas_object):
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
        
        x1 = clip(x1, 0, image_res[0])
        x2 = clip(x2, 0, image_res[0])
        y1 = clip(y1, 0, image_res[1])
        y2 = clip(y2, 0, image_res[1])        
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
        cropped_canny_image = crop_canny_image()
        if eval_df is None: #restrict modifications if eval_df is present
            df_from_canny_image(cropped_canny_image)

class horizontal_line(my_canvas_object):
    def __init__(self, canvas, y_pos, updatefn):
        self.canvas = canvas
        self.canv_y = y_pos
        self.id = None
        self.last_call_time = 404
        self.updatefn = updatefn
    
    def true_y(self, image_delta_y) -> float:
        pixel_height = -self.canv_y*image_downscale - image_delta_y
        return pixel_height*grid_d
    
    def hide(self):
        if self.id:
            self.canvas.delete(self.id)
            self.id = None
            display_df()
    
    def draw_hline(self):
        if self.id:
            self.canvas.delete(self.id)
        self.id = self.canvas.create_line(0, self.canv_y, frame_w, self.canv_y, fill="lime")
        self.call_update()
        
    def call_update(self):
        current_time = time.time()
        if self.visible():
            if current_time - self.last_call_time > 0.2:#update only if the function hasn't been called recently
                self.updatefn()
        self.last_call_time = current_time
    
    def move(self):
        if self.id:
            self.canvas.coords(self.id, 0, self.canv_y, frame_w, self.canv_y)
            self.call_update()
    
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

class canvas_image():
    def __init__(self, canvas):
        self.canvas = canvas
        self.id = self.canvas.create_image(frame_w, frame_h)
        self.image = None
        self.delta_x = 0
        self.delta_y = 0

    def set(self, myimg):
        try:
            self.id = image_canvas.create_image(frame_w, frame_h, anchor=tk.SE, image=myimg)
            self.canvas.tag_raise(self.id)
            self.canvas.image = myimg
        except Exception as e:
            logger.error(f'Error updating image: {e}')

    def clear(self):
        if self.id:
            self.canvas.delete(self.id)
            self.id = None
            self.image = None

class main_canvas_image(canvas_image):
    def update_deltas(self):
        if self.image is not None:
            imsize = pil_image.size
            self.delta_x = imsize[0] - image_res[0]
            self.delta_y = imsize[1] - image_res[1]
            logger.debug(f'Image res {imsize}')
        else:
            self.delta_x = 0
            self.delta_y = 0

    def set(self, myimg):
        try:
            self.canvas.itemconfig(self.id, anchor=tk.SE, image=myimg)
        except Exception as e:
            logger.error(f'Error updating image: {e}')
        self.update_deltas()

    def clear(self):
        self.image = None
        self.set(None)


def get_os():
    if os.name == 'nt':
        return "Windows"
    elif os.name == 'posix':
        return "Linux"
    else:
        return "Unknown"

def load_model_list():
    models = os.listdir(model_path)
    return models

def load_df():
    global df, eval_df
    try:
        df = pd.read_csv(sel_path, delimiter=' ', names=['x','y'])
        eval_df = None #Reset existing eval_df
    except Exception as e:
        logger.error(f'Error reading file: {e}')

def y_ticker(x, pos):
    return f'{x*1e-3:1.1f} mm'

def draw_plot(x,y):
    global graph_canvas, plot, mpltoolbar
    plot.clear()
    #plot.yaxis.set_major_formatter(y_ticker)
    plot.set_title(sel_path)
    plot.scatter(x, y, color="blue", s=0.2)
    if not pendant.get() and hline.visible():
        pos = hline.true_y(main_image.delta_y)
        plot.axhline(y=pos, color='lime', linestyle='-')
        plot.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        plot.set_yticks([round(pos,1)]) #add y tick to hline
    graph_canvas.draw()
    mpltoolbar.update()

def display_df():
    if df is not None:
        draw_plot(df['x'], df['y'])

def cut_df_by_treshold():
    global df
    treshold = float(var_treshold.get())
    if 'pixel_x' in df.columns:
        df = eval_df[eval_df['weight'] >= treshold][['pixel_x','pixel_y','x','y']]
    else:
        df = eval_df[eval_df['weight'] >= treshold][['x','y']]
    display_df()

def model_delete_dots():
    global df, eval_df
    try:
        model_name = model_var.get()
        logger.info(f'Deleting points with {model_name}')
        path = os.path.join(model_path+model_name)
        
        points_before = df.shape[0]
        eval_df = df.copy()
        eval_df['weight'] = inference(path, model_name, df, grid_d)
        plt.scatter(eval_df['x'], eval_df['y'], s=0.5, c=eval_df['weight'])
        plt.show()
        
        cut_df_by_treshold()
        points_after = df.shape[0]
        logger.info(f'Deleted {points_before - points_after} points')
    except Exception as e:
        logger.error(f'Error: {e}')

def run_select():
    logger.info('Launching Select.exe')
    try:
        pixel_path = os.path.join(select_dir, 'pixel.dat')
        shutil.copy2(out_file_name, pixel_path)
        os.remove(out_file_name)
        os.startfile(select_dir + 'SELECT.exe', cwd=select_dir)
    except Exception as e:
        logger.error(f'Error: {e}')

def run_select_s():
    logger.info('Launching Select_s.exe')
    try:
        with open(select_dir+'pti00.dat','w') as linefile:
            linefile.write(f'0 {hline.true_y(image_delta_y)}')
        pts_path = os.path.join(select_dir, 'pts00.dat')
        shutil.copy2(out_file_name, pts_path)
        os.remove(out_file_name)
        os.startfile(select_dir + 'Select_s.exe', cwd=select_dir)
    except Exception as e:
        logger.error(f'Error: {e}')

def run_either_select():
    """Run select.exe or select_s.exe"""
    global df
    df[['x','y']].to_csv(out_file_name, sep=' ', index=False, header=False)
    if pendant.get():
        run_select()
    else:
        run_select_s()

def import_dots_from_ooo():
    global sel_path
    sel_path = os.path.join(ooo_temp_dir, 'ooo.dat')
    load_df()
    display_df()
    clear_images()
    logger.info(f'Yoinked file: {sel_path}')

def inject_df_to_ooo():
    logger.info('Injecting dots')
    df[['x','y']].to_csv(out_file_name, sep=' ', index=False, header=False)
    target_path = os.path.join(ooo_temp_dir, 'ooo.dat')
    shutil.copy2(out_file_name, target_path)
    os.remove(out_file_name)
    with open(ooo_temp_dir+'ooo.dat_plus', 'w') as colordata: #clear color data
        colordata.write('')

def plot_ooo_trace():
    logger.info(f'Drawing fit from {select_dir}')
    try:
        plot_trace.plot(select_dir, pendant.get(), True)
    except Exception as e:
        logger.error(f'Plotting error: {e}')

def run_fit():
    logger.info('Fitting using TensorFlow...')
    try:
        error, opt_apex_radius, opt_sigma, opt_x_offset, opt_y_offset = fit_polar.process_df(df, pendant.get(), fit_epochs)
        logger.info(f'Sigma = {opt_sigma}')
        logger.info(f'Apex_radius = {opt_apex_radius}')
        logger.info(f'Error = {error}')
    except Exception as e:
        logger.error(f'Error: {e}')

def df_move_corners(matrix_df, grid_d):
    """finds and moves corner dots (to subpixels)"""
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

def df_from_canny_image(myimg):
    global df, eval_df
    if myimg is None:
        return
    a = np.argwhere(myimg == 255)

    df = pd.DataFrame(a,columns=['pixel_y','pixel_x'])#transpose
    df = df[['pixel_x','pixel_y']]
    df[['x','y']] = grid_d*df[['pixel_x','pixel_y']]
    if var_subpixel_diagonals.get():
        df = df_move_corners(df, grid_d)
    df['y'] *= -1
    display_df()
    eval_df = None #Reset existing eval_df

def crop_canny_image():
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
    
    return img_array

def canny_edge_detection(event=None):
    logger.info('Running Canny edge detection')
    global canny_image
  
    # Applying the Canny Edge filter  
    # with Aperture Size and L2Gradient 
    canny_image = cv2.Canny(cv_image,
                int(var_t_lower.get()),
                int(var_t_upper.get()),
                apertureSize = int(var_aperture_size.get()),
                L2gradient = var_l2gradient.get())

    cropped_canny_image = crop_canny_image()
    
    if var_show_canny_image.get():
        pil_canny_image = Image.fromarray(canny_image).convert('L')
        mask = Image.new('L', pil_canny_image.size, 0)
        mask.paste(pil_canny_image, (0, 0))
        #mask = ImageEnhance.Brightness(mask).enhance(2)
        transparent_image = Image.new('RGBA', pil_image.size)
        transparent_image.paste((0, 127, 255, 255), (0, 0), mask)
        
        transparent_image = ImageTk.PhotoImage(transparent_image)
        overlay_image.set(transparent_image)
    
    df_from_canny_image(cropped_canny_image)

def choose_file(event=None):
    global sel_path
    sel_path = os.path.join(filedialog.askopenfilename(initialdir=default_path, title="Select file",
        filetypes=(("all files", "*.*"),("dat files", "*.dat"),("bmp files", "*.bmp"))))
    if sel_path:
        logger.info(f'Selected file: {sel_path}')
        if sel_path.endswith('.dat'):
            load_df()
            display_df()
            clear_images()
        else:
            load_image()
            main_image.set(img)
            if var_do_canny_on_selection.get():
                canny_edge_detection()

def save_df(event=None):
    path = filedialog.asksaveasfilename(initialdir=default_path, title="Save file",
        filetypes=(("dat files", "*.dat"),("all files", "*.*")))
    if path == '':
        return
    if not path.endswith('.dat'):
        path += '.dat'
    df.to_csv(path, sep=' ', index=False, header=False)
    logger.info(f'Saved file: {path}')

def load_image():
    global img, pil_image, cv_image, canny_image
    try:
        pil_image = Image.open(sel_path)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) #Convert from PIL to cv2 image
        if image_downscale != 1:
            pil_image = pil_image.resize((frame_w, frame_h), Image.LANCZOS)
        img = ImageTk.PhotoImage(pil_image)
        canny_image = None #Reset
    except Exception as e:
        logger.error(f"Error loading image: {e}")

def clear_images():
    global img, pil_image, cv_image, canny_image
    img = None
    pil_image = None
    cv_image = None
    canny_image = None
    zimg = None
    update_lens_image()
    main_image.clear()
    overlay_image.clear()

def update_lens_image():
    global lens_container, zimg, lens_canvas
    try:
        lens_canvas.itemconfig(lens_container, anchor=tk.SE, image=zimg)
    except Exception as e:
        logger.error(f'Error updating lens: {e}')

def update_lens_hline(y):
    lens_hline.set_y(lens_size[1]/2 + lens_zoom*(hline.canv_y - y + 0.5))
    lens_hline.set_state(hline.visible())

def crop_lens(event):
    global zimg
    x,y = event.x, event.y
    update_lens_hline(y)
    x += main_image.delta_x
    y += main_image.delta_y
    if pil_image is not None:
        tmp = pil_image.crop((x-hz_x, y-hz_y, x+hz_x, y+hz_y))
        zimg = ImageTk.PhotoImage(tmp.resize((lens_size[0], lens_size[1]), Image.Resampling.NEAREST))
        update_lens_image()

def on_slider_change(strvalue):
    logger.debug(f'Slider value: {strvalue}')
    if eval_df is not None:
        cut_df_by_treshold()

def clip(mycoord,mymin,mymax):
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
    global root, fullscreen
    fullscreen = not fullscreen
    root.attributes("-fullscreen", fullscreen)

def validate_numeric_input(value: str) -> bool:
    return value.isdigit() or value == ''

def open_settings_window():
    global settings_window, t_lower
    if settings_window is not None and tk.Toplevel.winfo_exists(settings_window):
        settings_window.deiconify()
        settings_window.lift()
        settings_window.focus_force()
        return
    
    settings_window = tk.Toplevel(root)
    settings_window.title('Settings')
    settings_window.geometry('400x600')
    settings_window.attributes('-topmost', 'true')
    
    validate_num = settings_window.register(validate_numeric_input)

    tk.Label(settings_window, text='--- Edge detection settings ---', font = "Verdana 11").grid(row=0, column=0, padx=5, pady=10)

    tk.Label(settings_window, text='Lower treshold:').grid(row=1, column=0, padx=5, pady=5, sticky='w')
    canny_field1 = tk.Entry(settings_window, textvariable=var_t_lower, validate='key', validatecommand=(validate_num, '%P'))
    canny_field1.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(settings_window, text='Upper treshold:').grid(row=2, column=0, padx=5, pady=5, sticky='w')
    canny_field2 = tk.Entry(settings_window, textvariable=var_t_upper, validate='key', validatecommand=(validate_num, '%P'))
    canny_field2.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(settings_window, text='Apperture size:').grid(row=3, column=0, padx=5, pady=5, sticky='w')
    spinbox1 = ttk.Combobox(settings_window, textvariable=var_aperture_size, values=tuple(range(3, 32, 2)))
    spinbox1.grid(row=3, column=1, padx=10, pady=5)

    tk.Label(settings_window, text='L2Gradient').grid(row=4, column=0, padx=5, pady=5, sticky='w')
    switch1 = tk.Checkbutton(settings_window, variable=var_l2gradient, state='disabled')
    switch1.grid(row=4, column=1, padx=10, pady=5)
    
    tk.Label(settings_window, text='Subpixel diagonals').grid(row=5, column=0, padx=5, pady=5, sticky='w')
    switch1 = tk.Checkbutton(settings_window, variable=var_subpixel_diagonals)
    switch1.grid(row=5, column=1, padx=10, pady=5)

    tk.Label(settings_window, text='--- Plotting ---', font = "Verdana 11").grid(row=6, column=0, padx=5, pady=10)

    tk.Label(settings_window, text='Run Canny on file selection').grid(row=7, column=0, padx=5, pady=5, sticky='w')
    switch2 = tk.Checkbutton(settings_window, variable=var_do_canny_on_selection)
    switch2.grid(row=7, column=1, padx=10, pady=5)
    
    tk.Label(settings_window, text='Show Canny image').grid(row=8, column=0, padx=5, pady=5, sticky='w')
    switch3 = tk.Checkbutton(settings_window, variable=var_show_canny_image)
    switch3.grid(row=8, column=1, padx=10, pady=5)
    
    #fit_epochs = 600
    #logging.basicConfig(level=logging.INFO)


root = tk.Tk()
root.geometry("1800x1200")
root.title('DanPySelect Beta')
if get_os() == 'Windows':
    root.state('zoomed')
else:
    root.attributes('-zoomed', True) #If on linux
root.bind("<F11>", toggle_fullscreen)

#Setting variables
var_t_lower = tk.StringVar(value=str(t_lower))
var_t_upper = tk.StringVar(value=str(t_upper))
var_aperture_size = tk.StringVar(value=str(aperture_size))
var_l2gradient = tk.BooleanVar(value=L2Gradient)
var_subpixel_diagonals = tk.BooleanVar(value=subpixel_diagonals)
var_do_canny_on_selection = tk.BooleanVar(value=do_canny_on_selection)
var_show_canny_image = tk.BooleanVar(value=show_canny_image)
var_treshold = tk.StringVar(value=default_treshold)

canvas_frame = tk.Frame(root)
canvas_frame.pack(side=tk.LEFT)

image_canvas = tk.Canvas(canvas_frame, bg='white', bd=0, relief='ridge', highlightthickness=0, width = frame_w, height = frame_h)
main_image = canvas_image(image_canvas)
overlay_image = canvas_image(image_canvas)
image_canvas.pack(fill='both', side=tk.LEFT)
rect_pos = rectangle_selection(image_canvas, True)
image_canvas.bind('<Button 1>', rect_pos.on_click)
image_canvas.bind('<B1-Motion>', rect_pos.on_drag)
image_canvas.bind('<ButtonRelease-1>', rect_pos.on_release)
rect_neg = rectangle_selection(image_canvas, False)
image_canvas.bind('<Button 3>', rect_neg.on_click)
image_canvas.bind('<B3-Motion>', rect_neg.on_drag)
image_canvas.bind('<ButtonRelease-3>', rect_neg.on_release)
image_canvas.bind("<Motion>", crop_lens)

lens_canvas = tk.Canvas(image_canvas, bg='white', highlightthickness=1, width = lens_size[0], height = lens_size[1], cursor='boat')
lens_container = lens_canvas.create_image(lens_size[0], lens_size[1])
lens_canvas.place(x=0, y=0, anchor=tk.NW)
lens_canvas.bind("<B1-Motion>", drag_widget)
lens_hline = horizontal_line(lens_canvas, 0, update_lens_image)

base_frame = tk.Frame(root)
base_frame.pack(side=tk.TOP)

control_frame = tk.Frame(base_frame)
control_frame.pack(side=tk.TOP)

figure = Figure(figsize=(5.7, 5.7), dpi=100)
graph_canvas = FigureCanvasTkAgg(figure, base_frame)
mpltoolbar = NavigationToolbar2Tk(graph_canvas, base_frame)
graph_canvas.get_tk_widget().pack(side=tk.BOTTOM)
plot = figure.add_subplot(111)

files_frame = tk.Frame(control_frame)
files_frame.grid(row=0, column=0, padx=5, pady=5, sticky='w')

ed_frame = tk.Frame(control_frame)
ed_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')

nn_frame = tk.Frame(control_frame)
nn_frame.grid(row=1, column=0, padx=5, pady=5, sticky='w')

ooo_frame = tk.Frame(control_frame)
ooo_frame.grid(row=1, column=1, padx=5, pady=5, sticky='w')

button = tk.Button(files_frame, text = "Browse file", bg = "light blue", font = "Verdana 13", cursor='hand2', command = choose_file)
button.pack()
root.bind('<c>', choose_file)

save_button = tk.Button(files_frame, text='Save points', font = "Verdana 10", command = save_df)
save_button.pack(side=tk.RIGHT)
root.bind('<s>', save_df)

button = tk.Button(files_frame, text='Batch operations (x)', font = "Verdana 10", command = choose_file)
button.pack(side=tk.RIGHT)

button = tk.Button(ed_frame, text = "Canny Edge detection", bg = "red", font = "Verdana 12", command = canny_edge_detection)
button.pack()
root.bind('<e>', canny_edge_detection)

#Frame for Select things
select_frame = tk.Frame(ooo_frame)
select_frame.pack()

button = tk.Button(select_frame, text = "Yoink dots from ooo", font = "Verdana 11", command = import_dots_from_ooo)
button.grid(row=0, column=1, sticky='w')

button = tk.Button(select_frame, text = "Yeet dots to ooo", font = "Verdana 11", command = inject_df_to_ooo)
button.grid(row=1, column=1, sticky='w')

button = tk.Button(select_frame, text = "Run Select", font = "Verdana 11", command = run_either_select)
button.grid(row=0, column=0, sticky='w')

button = tk.Button(select_frame, text = "Plot trace", font = "Verdana 11", command = plot_ooo_trace)
button.grid(row=1, column=0, sticky='w')

pendant = tk.BooleanVar()
checkbox = tk.Checkbutton(select_frame, text='Pendant', variable=pendant, onvalue=True, offvalue=False)
checkbox.grid(row=2, column=0, sticky='w')

button = tk.Button(nn_frame, text = "Run TF_fit (x)", font = "Verdana 11", command = run_fit)
button.pack()

#Frame for hline things
line_frame = tk.Frame(ed_frame)
line_frame.pack()
hline = horizontal_line(image_canvas, initial_line_y, display_df)
image_canvas.bind('<Control-Button-1>', hline.set_y)
root.bind('<l>', hline.toggle)

button = tk.Button(line_frame, text = "Place line", font = "Verdana 9", command = hline.draw_hline)
button.pack(side=tk.LEFT)

button = tk.Button(line_frame, text = "￪", font = "Verdana 9", command = hline.move_up)
button.pack(side=tk.LEFT)
root.bind('<Up>', hline.move_up)

button = tk.Button(line_frame, text = "￬", font = "Verdana 9", command = hline.move_down)
button.pack(side=tk.LEFT)
root.bind('<Down>', hline.move_down)

button = tk.Button(line_frame, text = "Hide line", font = "Verdana 9", command = hline.hide)
button.pack(side=tk.LEFT)

# Frame for "delete dots with" button and dropdown menu
delete_dots_frame = tk.Frame(nn_frame)
delete_dots_frame.pack()

delete_dots_button = tk.Button(delete_dots_frame, text="Delete points", font="Verdana 11", command = model_delete_dots)
delete_dots_button.pack(side=tk.LEFT)

label = tk.Label(delete_dots_frame, text="with", font="Verdana 11")
label.pack(side=tk.LEFT)

model_list = load_model_list()
model_var = tk.StringVar(nn_frame)
model_var.set(model_list[0])

model_selector = tk.OptionMenu(delete_dots_frame, model_var, *model_list)
model_selector.pack(side=tk.LEFT)

# Frame for Threshold label and slider
threshold_frame = tk.Frame(nn_frame)
threshold_frame.pack()

slider_label = tk.Label(threshold_frame, text="Threshold:", font="Verdana 11")
slider_label.pack(side=tk.LEFT)

slider = tk.Scale(threshold_frame, from_=0, to=1, resolution=0.01, tickinterval=0.5, length=150, sliderlength=20, orient='horizontal', variable=var_treshold, command=on_slider_change)
treshold = default_treshold
slider.set(treshold)
slider.pack(side=tk.LEFT)

settings_button = tk.Button(files_frame, text='Settings', font = "Verdana 11", command = open_settings_window)
settings_button.pack(side=tk.RIGHT, padx=5)

#logger
st = tk.scrolledtext.ScrolledText(root, state='disabled')
st.configure(font='TkFixedFont')
st.pack(side=tk.BOTTOM)

# Create textLogger
text_handler = logm.TextHandler(st)

# Add the handler to logger
logger = logging.getLogger()
logger.addHandler(text_handler)

# Log some messages
#logger.debug('Example debug message')
#logger.info('Example info message')
#logger.warning('Example warn message')
#logger.error('Example error message')
#logger.critical('Example critical message')

root.mainloop()


