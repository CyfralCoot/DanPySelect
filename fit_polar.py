#Polar coordinates. Succsessful!

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt

print("TensorFlow version:", tf.__version__)


def df_rectangular_to_polar(mydf):
    negative_x_df = mydf[mydf['x'] < 0]
    mydf = mydf[mydf['x'] >= 0]
    
    polar_df = pd.DataFrame()
    polar_df_neg_x = pd.DataFrame()
    polar_df['ro'] = np.sqrt(mydf['x']**2 + mydf['y']**2)
    polar_df['phi'] = -1*np.arctan(mydf['y']/mydf['x']) + np.pi
    polar_df_neg_x['ro'] = np.sqrt(negative_x_df['x']**2 + negative_x_df['y']**2)
    polar_df_neg_x['phi'] = -1*np.arctan(negative_x_df['y']/negative_x_df['x'])
    polar_df = pd.concat([polar_df,polar_df_neg_x],ignore_index=True)
    return polar_df

def df_polar_to_rectangular(mypolardf):
    mypolardf['phi'] += np.pi/2
    df = pd.DataFrame()
    df['x'] = mypolardf['ro']*np.cos(mypolardf['phi'])*-1
    df['y'] = mypolardf['ro']*np.sin(mypolardf['phi'])
    return df

def sigma_from_aspherisity(asp):
    """asp = +- g*delta(p)/sigma"""
    g = 9.8
    dp = 1
    sigma = g*dp/asp
    return sigma

def prepare_data(mydf):
    mydf['x'] /= 1000
    mydf['y'] /= 1000

    #rough centering
    max_x,max_y = mydf.max()
    min_x,min_y = mydf.min()
    mydf['x'] -= np.average([max_x,min_x])
    mydf['y'] -= np.average([max_y,min_y])

    polar_df = df_rectangular_to_polar(mydf)
    polar_df['phi'] -= np.pi/2
    polar_df = polar_df.sort_values('phi')
    return polar_df

def model_s(alpha, apex_radius, aspherisity, x_offset, y_offset):
    """Sitting drops"""
    ro = apex_radius - ((aspherisity * (apex_radius**3)) / 3) * tf.cos(alpha) * tf.math.log(1 + tf.cos(alpha))
                                
    x = ro * tf.cos(alpha)
    y = ro * tf.sin(alpha)
    x_prime = x + x_offset
    y_prime = y + y_offset
                                    
    # Convert back to polar coordinates
    ro_prime = tf.sqrt(x_prime**2 + y_prime**2)
    return ro_prime

def model_p(alpha, apex_radius, aspherisity, x_offset, y_offset):
    """Pendant drops"""
    ro = apex_radius - ((aspherisity * (apex_radius**3)) / 3) * tf.cos(alpha) * tf.math.log(1 - tf.cos(alpha))
                                
    x = ro * tf.cos(alpha)
    y = ro * tf.sin(alpha)
    x_prime = x + x_offset
    y_prime = y + y_offset
                                    
    # Convert back to polar coordinates
    ro_prime = tf.sqrt(x_prime**2 + y_prime**2)
    return ro_prime

def fit(polar_df, model, num_epochs=600):
    phi_data = tf.constant(polar_df['phi'])
    observed_ro = tf.constant(polar_df['ro'])

    # Define TensorFlow variables for parameters to be optimized
    apex_radius = tf.Variable(1.0, dtype=tf.float64)
    aspherisity = tf.Variable(0.0, dtype=tf.float64)
    x_offset = tf.Variable(0.0, dtype=tf.float64)
    y_offset = tf.Variable(0.0, dtype=tf.float64)

    def mse_loss(phi_data, observed_ro):
        """Mean Squared error"""
        predictions = model(phi_data, apex_radius, aspherisity, x_offset, y_offset)
        return tf.reduce_mean(tf.square(predictions - observed_ro))

    optimizer = tf.optimizers.Adam(learning_rate=0.04)

    # Training step function
    @tf.function
    def train_step(phi_data, observed_ro):
        with tf.GradientTape() as tape:
            loss = mse_loss(phi_data, observed_ro)
        gradients = tape.gradient(loss, [apex_radius, aspherisity, x_offset, y_offset])
        optimizer.apply_gradients(zip(gradients, [apex_radius, aspherisity, x_offset, y_offset]))
        apex_radius.assign(tf.clip_by_value(apex_radius, 0.01, 10.0))
        return loss

    # Train the model
    for epoch in range(num_epochs):
        loss = train_step(phi_data, observed_ro)
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}, apex_radius: {apex_radius.numpy().round(6)}, aspherisity: {aspherisity.numpy().round(6)}, offsets: {x_offset.numpy().round(6)}, {y_offset.numpy().round(6)}')
    return loss.numpy(), apex_radius.numpy(), aspherisity.numpy(), x_offset.numpy(), y_offset.numpy()

def process_df(mydf, pendant=False, n_epochs=600):
    df = mydf[['x','y']]
    polar_df = prepare_data(df)
    phi_data = tf.constant(polar_df['phi'])
    observed_ro = tf.constant(polar_df['ro'])

    if pendant:
        model = model_p
    else:
        model = model_s

    loss, opt_apex_radius, opt_aspherisity, opt_x_offset, opt_y_offset = fit(polar_df, model, n_epochs)
    
    opt_sigma = sigma_from_aspherisity(opt_aspherisity)
    error = sqrt(loss)
    
    # After training, the optimized parameters are in apex_radius and aspherisity
    print(f'Optimized apex_radius: {opt_apex_radius}')
    print(f'Optimized aspherisity: {opt_aspherisity}')
    print(f'X offset: {opt_x_offset}')
    print(f'Y offset: {opt_y_offset}')
    print(f'Sigma: {opt_sigma}')

    predicted_ro = model(phi_data, opt_apex_radius, opt_aspherisity, opt_x_offset, opt_y_offset)
    predicted_polar_df = pd.DataFrame({'phi':phi_data, 'ro':predicted_ro})
    predicted_df = df_polar_to_rectangular(predicted_polar_df)

    fig, ax = plt.subplots(figsize=(10,8))
    plt.scatter(df['x'], df['y'], s=0.4)
    plt.plot(predicted_df['x'], predicted_df['y'], c='red', lw=0.6)
    plt.show()
    return error, opt_apex_radius, opt_sigma, opt_x_offset, opt_y_offset
    
if __name__ == '__main__':
    path = '66r-114_1.dat'
    df = pd.read_csv(path, delimiter=' ', names=['x','y'])
    process_df(df, True)


#polar display
#rmax = polar_df['ro'].max()
#fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#ax.plot(phi_data, observed_ro)
#ax.plot(phi_data, predicted_ro, c='red')
#ax.set_rmax(rmax+0.01)
#ax.set_rticks([rmax/2,rmax])  # Less radial ticks
#ax.set_title("drop", va='bottom')
#plt.show()
