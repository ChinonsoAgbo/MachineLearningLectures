import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
def bubble_set_normal(mx, my, number, s):
    x = np.random.normal(0, s, number) + mx
    y = np.random.normal(0, s, number) + my
    return x, y


def four_blobs(n1=400, n2=400, n3=400, n4=400):
    np.random.seed(42)
    dataset = np.zeros((n1 + n2 + n3 + n4, 2))
    (dataset[0:n1, 0], dataset[0:n1, 1]) = bubble_set_normal(2.5, 1.0, n1, 0.5)
    (dataset[n1:n1 + n2, 0], dataset[n1:n1 + n2, 1]) = bubble_set_normal(2.0, -3.0, n2, 0.3)
    (dataset[n1 + n2:n1 + n2 + n3, 0], dataset[n1 + n2:n1 + n2 + n3, 1]) = bubble_set_normal(-2.0, 5.0, n3, 0.6)
    (dataset[n1 + n2 + n3:n1 + n2 + n3 + n4, 0], dataset[n1 + n2 + n3:n1 + n2 + n3 + n4, 1]) = bubble_set_normal(-4.0,
                                                                                                                 -1.0,
                                                                                                                 n4,
                                                                                                                 0.9)
    return scaler.fit_transform(dataset)


def mouse_shape():
    np.random.seed(42)
    dataset = np.zeros((1000, 2))
    (dataset[0:150, 0], dataset[0:150, 1]) = bubble_set_normal(-0.75, 0.75, 150, 0.15)
    (dataset[150:300, 0], dataset[150:300, 1]) = bubble_set_normal(0.75, 0.75, 150, 0.15)
    (dataset[300:1000, 0], dataset[300:1000, 1]) = bubble_set_normal(0, 0, 700, 0.29)
    return scaler.fit_transform(dataset)


def two_moons(samples_per_moon=240, p_noise=2):
    np.random.seed(42)
    t_moon0 = np.linspace(0, np.pi, samples_per_moon)
    t_moon1 = np.linspace(0, np.pi, samples_per_moon)
    moon0x = np.cos(t_moon0)
    moon0y = np.sin(t_moon0)
    moon1x = 1 - np.cos(t_moon1)
    moon1y = 0.5 - np.sin(t_moon1)
    x = np.vstack((np.append(moon0x, moon1x), np.append(moon0y, moon1y))).T
    x = x + p_noise / 100 * np.random.normal(size=x.shape)
    return scaler.fit_transform(x)


def circle():
    np.random.seed(42)
    phi = np.linspace(0, 2 * np.pi, 800)
    x1 = 1.5 * np.cos(phi)
    y1 = 1.5 * np.sin(phi)
    x2 = 0.5 * np.cos(phi)
    y2 = 0.5 * np.sin(phi)
    x = np.vstack((np.append(x1, x2), np.append(y1, y2))).T
    x = x + 0.1 * np.random.normal(size=x.shape)
    return scaler.fit_transform(x)

def noise():
    np.random.seed(42)

    return scaler.fit_transform(np.random.random((1000, 2)))