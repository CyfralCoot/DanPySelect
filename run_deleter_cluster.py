import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_num_between(mystring, mystart, myend=None):
    xstart = mystring.find(mystart) + len(mystart)
    xend = mystring.find(myend)
    if xend == -1:
        return None
    myvalue = mystring[xstart:xend]
    return int(myvalue)

def get_matrix_df(mydf, grid_d, dim=127):
    """Get df with positive int x any y for image construction"""
    matrix_df = mydf[['x','y']]
    matrix_df['x'] = matrix_df['x']//grid_d
    matrix_df['y'] = matrix_df['y']//grid_d
    min_x, min_y = matrix_df.min()
    matrix_df['x'] += dim - min_x
    matrix_df['y'] += dim - min_y
    return matrix_df.astype(int)

def get_pixel_matrix_df(mydf, dim=127):
    """Get df with positive int x any y for image construction from df['pixel_x'] and df['pixel_y']"""
    matrix_df = mydf[['pixel_x','pixel_y']]
    min_x, min_y = matrix_df.min()
    matrix_df['x'] = matrix_df['pixel_x'] + dim - min_x
    matrix_df['y'] = matrix_df['pixel_y'] + dim - min_y
    return matrix_df[['x','y']]

def construct_full_image(matrix_df, dim=127):
    max_x, max_y = matrix_df.max()
    
    matrix = np.zeros((max_x+dim+1, max_y+dim+1), dtype=bool)
    for row in matrix_df.itertuples(index=False):
        matrix[row[0],row[1]] = 1    
    return matrix

def construct_full_image_2(matrix_df, dim=127):
    """Make sure matrix_df has no duplicates before running this function. Works faster than V1"""
    max_x, max_y = matrix_df[['x','y']].max()
    
    matrix = np.zeros((max_x+dim+1, max_y+dim+1), dtype='bool')
    np.add.at(matrix, (matrix_df['x'], matrix_df['y']), 1)
    return matrix

def cut_matrix(matrix_df, mymatrix, dim=127):
    """Cuts a full image into square small images centered around matrix_df dots"""
    matrix_list = []
    for row in matrix_df.itertuples(index=False):
        cut_2darray = mymatrix[row[0]-dim:row[0]+dim+1, row[1]-dim:row[1]+dim+1]
        matrix_list.append(cut_2darray)
    return matrix_list

def cut_matrix_2(matrix_df, mymatrix, dim=127):
    """Cuts a full image into series of square small images centered around matrix_df dots. Version 2. Works faster"""
    # Get the coordinates
    x_coords = matrix_df['x'].values
    y_coords = matrix_df['y'].values
    
    # Generate the range of offsets for cropping
    offsets = np.arange(-dim, dim+1)
    
    # Generate the grid of indices for cropping
    x_indices = x_coords[:, None] + offsets
    y_indices = y_coords[:, None] + offsets
    
    # Extract the cropped images using advanced indexing
    cropped_images = mymatrix[x_indices[:, None, :], y_indices[:, :, None]]
    return cropped_images.astype(bool)

def inference(model_path, model_name, df, grid_d=5.2):
    model = load_model(model_path)
    try:
        dim = get_num_between(model_name, 'cluster', '_V')
    except:
        dim = get_num_between(model_name, 'cluster', '_2d')
    
    #if 'pixel_x' in df.columns:
    m_df = get_pixel_matrix_df(df, dim)
    img = construct_full_image_2(m_df, dim)
    #else:
        #m_df = get_matrix_df(df, grid_d, dim)
        #img = construct_full_image(m_df, dim)
    matrix_list = cut_matrix_2(m_df, img, dim)
    
    if '2d' in model_name:
        matrix_list = np.expand_dims(matrix_list, axis=-1)
    
    return model.predict(matrix_list, verbose = 0)[:,0]
    
