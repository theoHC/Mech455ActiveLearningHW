#!/usr/bin/env python
# coding: utf-8

# ### ME455: Introduction to the ergodic metric
# 
# Date: May 14, 2025
# 
# Author: Max Muchen Sun
# 
# References:
# 
# [1] *Mathew, G. and Mezić, I.*, 2011. **Metrics for ergodicity and design of ergodic dynamics for multi-agent systems**. Physica D: Nonlinear Phenomena, 240(4-5), pp.432-442. [[**Link**](https://www.sciencedirect.com/science/article/pii/S016727891000285X)]

# In[1]:


import numpy as np
np.set_printoptions(precision=4)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlepad'] = 8.0
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 3
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 3
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['legend.fontsize'] = 15


# #### Fourier basis function
# 
# We start with defining of the exploration space $\mathcal{X}$, which is a $d$-dimensional rectangular Euclidean space. Without loss of generality, we define the search space as:
# 
# $$\mathcal{X}=[0,L_1]\times\cdots\times[0,L_d], $$  
# 
# where $L_i$ is the boundary length of the $i$-th dimension. The *normalized* Fourier basis function is defined over the search space. Denote a state within the search space as $x = [x_1, \dots, x_d] \in \mathcal{X}$, the normalized Fourier basis function is characterized by a $d$-dimensional *index vector* $k = [k_1, \dots, k_d]$, where each entry of the vector is a whole number $k_i \in [0, 1, 2, \dots, K]$:
# 
# $$
# \begin{align}
# f_k(x) & = \frac{1}{h_k} \prod_{i=1}^{d} \cos\left( \frac{k_i \pi}{L_i} x_i \right),
# \end{align}
# $$
# 
# where $h_k$ is the normalization term, such that the function space norm of each basis function $\left[\int_{\mathcal{X}} f_k(x)^2 dx\right]^{\frac{1}{2}}$ is $1$.

# The normalized basis functions consist of a set of *orthonormal bases* in the infinite dimensional function space. Thus, we can transform any function with the same domain (the search space) into a set of new coordinates under these *orthonormal bases*.
# 
# Given a spatial probability density function $p(x)$, we can transform this function as follow:
# 
# $$
# \begin{align}
#     p(x) & = \sum_{k}^{\infty} \left( \underbrace{\int_{\mathcal{X}} p(x) f_k(x) dx}_{\phi_k} \right) \cdot f_k(x) \\
#         & = \sum_{k}^{\infty} \phi_k \cdot f_k(x)
# \end{align}
# $$ where all the coefficients $[c_k]$ consist the coordinate of the function $p(x)$ under the bases $[f_k(x)]$.
# 
# Though the equality only holds in theory when we have an infinite number of basis functions, in practice, we can choose a finite number of basis functions to approximate an infinite dimensional function $p(x)$ within a finite dimensional Euclidean space (a finite number of coefficients).
# 
# This transformation from an infinite dimensional function space to a finite dimensional vector space is crucial for the ergodic metric. We now verify this transformation below.

# In[2]:


### We define a Gaussian mixture model as the spatial probability density function
from scipy.stats import multivariate_normal as mvn

mean1 = np.array([0.35, 0.38])
cov1 = np.array([
    [0.01, 0.004],
    [0.004, 0.01]
])
w1 = 0.5

mean2 = np.array([0.68, 0.25])
cov2 = np.array([
    [0.005, -0.003],
    [-0.003, 0.005]
])
w2 = 0.2

mean3 = np.array([0.56, 0.64])
cov3 = np.array([
    [0.008, 0.0],
    [0.0, 0.004]
])
w3 = 0.3


# Define the Gaussian-mixture density function here
def pdf(x):
    return w1 * mvn.pdf(x, mean1, cov1) + \
           w2 * mvn.pdf(x, mean2, cov2) + \
           w3 * mvn.pdf(x, mean3, cov3)


# In[3]:


### We are going to use 10 coefficients per dimension --- so 100 index vectors in total

num_k_per_dim = 10
ks_dim1, ks_dim2 = np.meshgrid(
    np.arange(num_k_per_dim), np.arange(num_k_per_dim)
)
ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T  # this is the set of all index vectors
print('First 5 index vectors: ')
print(ks[:5, :])

# define a 1-by-1 2D search space
L_list = np.array([1.0, 1.0])  # boundaries for each dimension

# Discretize the search space into 100-by-100 mesh grids
grids_x, grids_y = np.meshgrid(
    np.linspace(0, L_list[0], 100),
    np.linspace(0, L_list[1], 100)
)
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
dx = 1.0 / 99
dy = 1.0 / 99  # the resolution of the grids

# Compute the coefficients
coefficients = np.zeros(ks.shape[0])  # number of coefficients matches the number of index vectors
for i, k_vec in enumerate(ks):
    # step 1: evaluate the fourier basis function over all the grid cells
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)  # we use NumPy's broadcasting feature to simplify computation
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)  # normalization term
    fk_vals /= hk

    # step 2: evaluate the spatial probabilty density function over all the grid cells
    pdf_vals = pdf(grids)  # this can computed ahead of the time

    # step 3: approximate the integral through the Riemann sum for the coefficient
    phik = np.sum(fk_vals * pdf_vals) * dx * dy
    coefficients[i] = phik

print('First 5 coefficients: ', coefficients[:5])


# In[4]:


### We can verify the correctness of the coefficients by reconstructing the probability
### density function through the coefficients

pdf_recon = np.zeros(grids.shape[0])
for i, (phik, k_vec) in enumerate(zip(coefficients, ks)):
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
    fk_vals /= hk

    pdf_recon += phik * fk_vals

pdf_gt = pdf(grids)  # ground truth density function

# visualize for comparison
fig, axes = plt.subplots(1, 2, figsize=(9,5), dpi=70, tight_layout=True)

ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(0.0, L_list[0])
ax.set_ylim(0.0, L_list[1])
ax.set_title('Original PDF')
ax.contourf(grids_x, grids_y, pdf_gt.reshape(grids_x.shape), cmap='Reds')

ax = axes[1]
ax.set_aspect('equal')
ax.set_xlim(0.0, L_list[0])
ax.set_ylim(0.0, L_list[1])
ax.set_title('Reconstructed PDF')
ax.contourf(grids_x, grids_y, pdf_recon.reshape(grids_x.shape), cmap='Blues')

plt.show()
plt.close()


# #### Ergodic metric: distance measure between a trajectory and a distribution
# 
# The intuition behind the ergodic metric is that we want to measure the difference between a spatial probability density function and the **empirical distribution** of a trajectory $s(t)$ within the search space. The empirical distribution is defined as:
# 
# $$
# \begin{align}
# \Phi_{s(t)}(x) = \frac{1}{T} \int_{0}^{T} \delta(x {-} s(t)) dt,
# \end{align}
# $$
# 
# where $\delta(\cdot)$ is a Dirac delta function.
# 
# The key insight here is that we can also transform the empirical distribution into a vector space through the Fourier basis functions. In fact, the Dirac delta function further simplifies the calculation:
# 
# $$
# \begin{align}
#     \Phi_{s(t)}(x) & = \sum_{k}^{\infty} \left[ \int_{\mathcal{X}} \left( \frac{1}{T} \int_{0}^{T} \delta(x {-} s(t)) dt \right) f_k(x) dx \right] \cdot f_k(x) \\
#     & = \sum_{k}^{\infty} \left(\underbrace{\frac{1}{T} \int_{0}^{T} f_k(s(t)) dt}_{c_k} \right) \cdot f_k(x) \\
#         & = \sum_{k}^{\infty} c_k \cdot f_k(x)
# \end{align}
# $$
# 
# This allows us to approximate the distance between a probability density function $p(x)$ and a trajectory's empirical distribution $\Phi_{s(t)}(x)$ in a finite dimensional vector space with the following formula, which is the mathematical formula of the ergodic metric:
# \begin{align}
#     \mathcal{E}(p(x), s(t)) & = \sum_{k} \lambda_k \cdot (c_k - \phi_k)^2, \\
#     \lambda_k & = (1 + \Vert k \Vert)^{-\frac{d+1}{2}}.
# \end{align}
# 
# This ergodic metric formula is equivalent to a distance metric given by the Sobolev space norm of negative index $-\frac{d+1}{2}$. [1]
# 
# **If the ergodic metric between a dynamic system's trajectory, over an infinitely long time horizon, and a spatial probability density function is zero, the system is ergodic with respect to the spatial probability distribution.**

# #### Ergodic control: connecting temporal statistics with spatial statistics
# 
# The goal of ergodic control is to design a dynamic system, through optimal control approaches, that is ergodic with respect to a target spatial probability distribution. In other words, to design a dynamic system with smallest possible ergodic metric value with respect a target spatial probability density function.
# 
# An ergodic system connects the temporal statistic of its behavior with the spatial statistic of the target distribution: an ergodic visits any region of the search space with the amount of time proportional to the integrated value of the spatial distribution over the same region. This property naturally introduces two advantages of using ergodic control as an exploration strategy:
# 
# (1) An ergodic system will eventually cover the entire search space, so long as the target spatial distribution is a non-zero probability density over any region of the space. This is the intrinsic coverage guarantee of ergodic control;
# 
# (2) An ergodic system automatically allocates the time spent on investigating each region of the search space based on the integrated density of the region. This is preferrable to standard coverage approach as an ergodic system covers a distribution instead of just the search space.
# 
# Here we give an example of calculating the ergodic metric given a trajectory. We will use the Gaussian-mixture distribution defined above as the target spatial distribution.
# 
# Assuming we have a dynamic system, where the state of the system at each time step is a sample from the Guassian-mixture distribution. Such a system is indeed ergodic, thus should lead to small value when we evaluate the ergodic metric.

# In[5]:


# define the dynamics
def dyn(st):
    mixture_category = np.where(np.random.multinomial(1, [w1, w2, w3])==1.0)[0][0]
    if mixture_category == 0:
        st_new = mvn.rvs(mean1, cov1)
    elif mixture_category == 1:
        st_new = mvn.rvs(mean2, cov2)
    else:
        st_new = mvn.rvs(mean3, cov3)
    return st_new


# simulate the trajectory
dt = 0.1
tsteps = 200
st = np.ones(2) * 0.5
s_traj = []
for t in range(tsteps):
    st_new = dyn(st)
    s_traj.append(st_new)
    st = st_new.copy()
s_traj = np.array(s_traj)

# compute the coefficient of the trajectory
traj_coefficients = np.zeros(ks.shape[0])
for i, k_vec in enumerate(ks):
    # step 1: evaluate the basis function over the trajectory
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * s_traj), axis=1)
    hk = np.sqrt(np.sum(np.square(np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1))) * dx * dy)
    fk_vals /= hk

    # step 2: approximate the integral through the Riemann sum for the coefficient
    ck = np.sum(fk_vals) * dt / (tsteps * dt)
    traj_coefficients[i] = ck

print('First 5 coefficients: ', traj_coefficients[:5])


# In[6]:


phi_recon = np.zeros(grids.shape[0])
for i, (ck, k_vec) in enumerate(zip(traj_coefficients, ks)):
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
    fk_vals /= hk

    phi_recon += ck * fk_vals

pdf_gt = pdf(grids)  # ground truth density function

# visualize for comparison
fig, axes = plt.subplots(1, 2, figsize=(9,5), dpi=70, tight_layout=True)

ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(0.0, L_list[0])
ax.set_ylim(0.0, L_list[1])
ax.set_title('Original PDF')
ax.contourf(grids_x, grids_y, pdf_gt.reshape(grids_x.shape), cmap='Reds')

ax = axes[1]
ax.set_aspect('equal')
ax.set_xlim(0.0, L_list[0])
ax.set_ylim(0.0, L_list[1])
ax.set_title('Empirical Distribution')
ax.contourf(grids_x, grids_y, phi_recon.reshape(grids_x.shape), cmap='Blues')
ax.plot(s_traj[:,0], s_traj[:,1], linestyle='', marker='o', color='k', alpha=0.2, label='Trajectory')
ax.legend(loc=1)

plt.show()
plt.close()


# In[7]:


# Finally, we compute the erogdic metric
lamk_list = np.power(1.0 + np.linalg.norm(ks, axis=1), -3/2.0)
erg_metric = np.sum(lamk_list * np.square(coefficients - traj_coefficients))
print('ergodic metric: ', erg_metric)


# #### (Bonus) Accelerated computation of ergodic metric
# 
# You may have noticed that most of the operations for computing the ergodic metric, mainly the calculation of the coefficients, can be done in parallel. In general, we need to evalulate **each** fourier basis function (with each index vector) over **each** state that we are interested in --- here the states can the discretized state space or the states within a trajectory.
# 
# Here we show can example of leveraging this structure to accelerate the computation of coefficients. By using NumPy's broadcasting feature, we can compute the coefficients without a single for loop!
# 
# Below is how we were computing the coefficients:
# 
# ```python
# coefficients = np.zeros(ks.shape[0])  
# for i, k_vec in enumerate(ks):
#     fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)
#     hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
#     fk_vals /= hk
# 
#     pdf_vals = pdf(grids)  
#     
#     phik = np.sum(fk_vals * pdf_vals) * dx * dy
#     coefficients[i] = phik
# ```
# 
# We were already using broadcasting to avoid the loop over each grid cells in the discretized search space, now we will further optimize to get rid of the loop over index vectors.

# In[8]:


fk_vals_tensor = np.cos(np.pi * ks[:,np.newaxis] / L_list * grids)
print(fk_vals_tensor.shape)


# This tensor `fk_vals_tensor` with the shape (100, 10000, 2) contains the evaluation of the fourier basis function at **each** index vector (100 in total), **each** grid cell in the discretized state space (100*100 in total), and **each** dimension of the state (2 in total). Based on this tensor, we can calculate the coefficients as follow:

# In[9]:


# compute the normalization terms for each index vector
hk_list = np.sqrt(np.sum(np.square(np.prod(fk_vals_tensor, axis=2)), axis=1) * dx * dy)

# compute the coefficients
coefficients_new = np.sum(pdf(grids) * np.prod(fk_vals_tensor, axis=2) / hk_list[:,np.newaxis], axis=1) * dx * dy

# verify that this optimized implementation produces the same results
print('difference: ', np.linalg.norm(coefficients - coefficients_new))


# In[ ]:




