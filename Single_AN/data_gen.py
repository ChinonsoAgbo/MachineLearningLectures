import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
def bubble_set_normal(mx, my, number, s):
    x = np.random.normal(0, s, number) + mx
    y = np.random.normal(0, s, number) + my
    return x, y

def two_blobs(n1=400, n2=400):
    np.random.seed(42)
    dataset = np.ones((n1 + n2, 3))
    (dataset[0:n1, 0], dataset[0:n1, 1]) = bubble_set_normal(4, -2, n1, 0.8)
    (dataset[n1:n1 + n2, 0], dataset[n1:n1 + n2, 1]) = bubble_set_normal(2.0, -3.0, n2, 0.8)
    dataset[:, :2] = scaler.fit_transform(dataset[:, :2])
    dataset[0:n1, 2] = -1
    return dataset

def circle(noise_amplitude=0.1, n_points=800):
    phi = np.linspace(0, 2 * np.pi, n_points)
    x1 = 1.5 * np.cos(phi)
    y1 = 1.5 * np.sin(phi)
    x2 = 0.5 * np.cos(phi)
    y2 = 0.5 * np.sin(phi)
    x = np.vstack((np.append(x1, x2), np.append(y1, y2))).T
    x = x + noise_amplitude * np.random.normal(size=x.shape)
    label0 = np.zeros_like(x1)
    label1 = np.ones_like(x2)
    label = np.append(label0, label1)
    label = np.expand_dims(label, axis=-1)
    x = scaler.fit_transform(x)
    data = np.hstack((x, label))
    return data

def generate_mesh_grid(x_range, y_range, grid_points):
    """
    Generated a uniform 2D meshgrid to test the prediction of machine learning models and visualize the decision
    boundary
    :param x_range: tuple of minimum and maximum x value for meshgrid
    :param y_range: tuple of minimum and maximum y value for meshgrid
    :param grid_points: int number of points along each axis
    :return: pd.DataFrame of 2D meshgrid coordinates
    """
    x = np.linspace(x_range[0], x_range[1], grid_points)
    y = np.linspace(y_range[0], y_range[1], grid_points)
    x_grid, y_grid = np.meshgrid(x, y)
    xy_flat = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)
    return xy_flat

def two_moons_generator(samples_per_moon=240, noise_amplitude=0.1):
    """
    Generates a 2D binary data set for testing machine learning models. The data distribution looks like two intertwined
    moons
    :param samples_per_moon: int number of data points generated per class
    :param noise_amplitude: float amplitude of random noise added to the data distributions
    :return: tuple of two pd.DataFrames containing the data and labels, and an array of colors for plotting the points
    """
    # show shape
    # generating a numpy array of samples_per_moon equidistant numbers between at 0 and pi
    moon_0 = np.linspace(0, np.pi, samples_per_moon)
    # show min, max, mean
    # applying trigonometric functions in numpy
    moon_0_x = np.cos(moon_0)
    moon_0_y = np.sin(moon_0)
    moon_1_x = 1 - np.cos(moon_0)
    moon_1_y = 0.5 - np.sin(moon_0)
    # aks for shape, explain axis, generates new axis
    # stacking 1D arrays of shape (samples_per_moon,) to 2D array of shape (samples_per_moon, 2)
    data = np.stack([np.append(moon_0_x, moon_1_x), np.append(moon_0_y, moon_1_y)], axis=1)
    # random number generation
    # adding random noise
    data += noise_amplitude * np.random.normal(size=data.shape)
    # explain two different methods of generating 0s and 1s
    # generating an array of class labels 0, 1 for the two moon distributions
    data = scaler.fit_transform(data)
    labels = np.hstack([np.zeros(samples_per_moon), np.ones_like(moon_0)])
    data = np.hstack((data, np.expand_dims(labels, axis=-1)))
    return data