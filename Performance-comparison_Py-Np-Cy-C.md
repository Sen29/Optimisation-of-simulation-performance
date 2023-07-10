<h1 style="text-align: center;">Performance comparison_Python-Numpy-Cython-C</h1>

The following program is a simulation of particle motion for the lenaard-jones potential, with n controlling the number of particles and iterations controlling the number of iterations.
The article concludes with a comparison of the performance of the Python version, the vectorisation optimised version using Numpy, the optimised version using Cython, and the version using C.

# Python


```python
import matplotlib.pyplot as plt
import timeit
import numpy as np


def particle_initial_position(n):
    np.random.seed(0)
    p = np.random.rand(n, 2)
    
    for i in range(n):
        while True:
            conflict = False
            for j in range(i):
                distance = np.linalg.norm(p[i] - p[j])
                if distance <= 0.1:
                    conflict = True
                    break
            if not conflict:
                break
            p[i] = np.random.rand(2)
    
    return p


def force_acting_on_i2(n, iterations, show_plot=False):
    p = particle_initial_position(n)
    for _ in range(iterations):
        fs = np.zeros(shape=(n, 2))  # forces from all other particles
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = p[j] - p[i]
                    dist = np.sqrt(np.sum(r**2))
                    
                    with np.errstate(invalid='ignore'):
                        unit_vector_nan = r / dist
                    unit_vector = np.nan_to_num(unit_vector_nan)
                    
                    epsilon = 1  # 势能参数
                    sigma = 0.1  # 势能参数
                    
                    with np.errstate(invalid='ignore'):
                        force_nan = 48 * epsilon * np.power(sigma, 12) / np.power(dist, 13) - 24 * epsilon * np.power(sigma, 6) / np.power(dist, 7)
                    force = np.nan_to_num(force_nan)
                    fs[i] += -force * unit_vector
        

        x_delta = fs / 1 * 0.00001
        p = update_position(p, x_delta)
        
        pos = p
        colors = ['red', 'green', 'blue', 'orange'] 
        if show_plot:
            if _ % 50 == 0:
                update_plot(pos,colors)
    # plot finally result
#    print("P({}): ".format(iterations), p)

    return p


def update_position(p, delta_r, min_x=0, max_x=1):
    
    new_pos = p + delta_r
    
    x_out_of_bounds = np.logical_or(new_pos[:,0] > max_x, new_pos[:,0] < min_x)
    y_out_of_bounds = np.logical_or(new_pos[:,1] > max_x, new_pos[:,1] < min_x)
    
    new_pos[x_out_of_bounds, 0] = np.clip(new_pos[x_out_of_bounds, 0], min_x, max_x)
    new_pos[y_out_of_bounds, 1] = np.clip(new_pos[y_out_of_bounds, 1], min_x, max_x)
    
    return new_pos


def update_plot(pos,color):

    plt.clf()

    xpos = pos[:, 0]
    ypos = pos[:, 1]

    N = len(pos)
    N_color = len(color)
    for i in range(N):
        plt.plot(xpos[i], ypos[i], "o", color=color[i % N_color])

    plt.xlim(left=-0.1, right=1.1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.grid()
    plt.draw()
    plt.pause(0.0001)
    

#force_acting_on_i2(50, 100, show_plot=True)

compute_time_py = timeit.timeit(lambda: force_acting_on_i2(50, 100,show_plot=False), number=1)

print("simulate_py execution time:", compute_time_py)
```

    simulate_py execution time: 23.236959100000007
    

# Numpy


```python
import matplotlib.pyplot as plt
import timeit
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def particle_initial_position(n):
    np.random.seed(0)
    p = np.random.rand(n, 2)
    
    for i in range(n):
        while True:
            conflict = False
            for j in range(i):
                distance = np.linalg.norm(p[i] - p[j])
                if distance <= 0.1:
                    conflict = True
                    break
            if not conflict:
                break
            p[i] = np.random.rand(2)
    
    return p


def force_acting_on_i2(n, iterations,show_plot=False):
    p = particle_initial_position(n)
    for _ in range(iterations):
        rvs = (p[:, np.newaxis, :] - p[np.newaxis, :, :])
        dist = np.sqrt(np.sum(rvs**2, axis=-1))
        fs = np.zeros(shape=(n, 2))  # forces from all other particles

        #for _ in range(iterations):
        dist_i = dist[:, :]
        rvs_i = rvs[:, :, :]

        with np.errstate(invalid='ignore'):
            unit_vectors_nan = rvs_i / dist_i[:, :, np.newaxis]
        unit_vectors = np.nan_to_num(unit_vectors_nan)

        dist_new = dist_i[:, :, np.newaxis]
        epsilon = 1  # 势能参数
        sigma = 0.1  # 势能参数
        with np.errstate(invalid='ignore'):
            fs_nan = 48 * epsilon * np.power(sigma, 12) / np.power(dist_new, 13)-24 * epsilon * np.power(sigma, 6) / np.power(dist_new, 7)
        fs = np.nan_to_num(fs_nan)*unit_vectors
                
        f_i = fs.sum(axis=1)
        x_delta = f_i / 1 * 0.00001

        p = update_position(p, x_delta)
        pos = p

        colors = ['red', 'green', 'blue', 'orange'] 
        if show_plot:
            if _ % 50 == 0:
                update_plot(pos,colors)
    # plot finally result
#    print("P({}): ".format(iterations), p)

    return p


def update_position(p, delta_r, min_x=0, max_x=1):
    
    new_pos = p + delta_r
    
    x_out_of_bounds = np.logical_or(new_pos[:,0] > max_x, new_pos[:,0] < min_x)
    y_out_of_bounds = np.logical_or(new_pos[:,1] > max_x, new_pos[:,1] < min_x)
    
    new_pos[x_out_of_bounds, 0] = np.clip(new_pos[x_out_of_bounds, 0], min_x, max_x)
    new_pos[y_out_of_bounds, 1] = np.clip(new_pos[y_out_of_bounds, 1], min_x, max_x)
    
    return new_pos

def update_plot(pos,color):

    plt.clf()

    xpos = pos[:, 0]
    ypos = pos[:, 1]

    N = len(pos)
    N_color = len(color)
    for i in range(N):
        plt.plot(xpos[i], ypos[i], "o", color=color[i % N_color])

    plt.xlim(left=-0.1, right=1.1)
    plt.ylim(bottom=-0.1, top=1.1)
    plt.grid()
    plt.draw()
    plt.pause(0.0001)
        

#force_acting_on_i2(50, 100, show_plot=True)

compute_time_np = timeit.timeit(lambda: force_acting_on_i2(50, 100,show_plot=False), number=1)

print("simulate_np execution time:", compute_time_np)
```

    simulate_np execution time: 0.0953713000000107
    

# Numba


```python
import matplotlib.pyplot as plt
import timeit
import numpy as np
import numba as nb


def particle_initial_position(n):
    np.random.seed(0)
    p = np.random.rand(n, 2)
    for i in range(n):
        while True:
            conflict = False
            for j in range(i):
                distance = np.linalg.norm(p[i] - p[j])
                if distance <= 0.1:
                    conflict = True
                    break
            if not conflict:
                break
            p[i] = np.random.rand(2)
    return p


@nb.jit
def simulate_nb(p, iterations):
    n = len(p)
    fs = np.zeros(shape=(n, 2))
    x_delta = np.zeros(shape=(n, 2))
    pos = p.copy()
    epsilon = 1  # 势能参数
    sigma = 0.1  # 势能参数
    
    for _ in range(iterations):
        fs[:, :] = 0.0

        for i in range(n):
            for j in range(n):
                if i != j:
                    x = pos[j, 0] - pos[i, 0]
                    y = pos[j, 1] - pos[i, 1]
                    dist = (x ** 2 + y ** 2) ** 0.5

                    ux = x / dist
                    uy = y / dist
                    force = 48 * epsilon * (sigma ** 12) / (dist ** 13) - 24 * epsilon * (sigma ** 6) / (dist ** 7)
                    factor = 0.00001
                    fs[i, 0] += -force * ux * factor
                    fs[i, 1] += -force * uy * factor

        x_delta[:, :] = 0.0
        for i in range(n):
            for j in range(2):
                x_delta[i, j] = fs[i, j] / 1.0

        pos = update_position(pos, x_delta)

    return pos


@nb.jit
def clip(a, min_value, max_value):
    return min(max(a, min_value), max_value)

@nb.jit
def update_position(p, delta_r, minimum=0, maximum=1):

    n = p.shape[0]

    new_pos = np.empty_like(p, dtype=np.float64)
    
    for i in range(n):
        x = p[i, 0] + delta_r[i, 0]
        y = p[i, 1] + delta_r[i, 1]

        if x > maximum or x < minimum:
            x = clip(x, minimum, maximum)

        if y > maximum or y < minimum:
            y = clip(y, minimum, maximum)

        new_pos[i, 0] = x
        new_pos[i, 1] = y

    return new_pos


compute_time_nb = timeit.timeit(lambda: simulate_nb(particle_initial_position(50), 100), number=1)

print("simulate_nb execution time:", compute_time_nb)
```

    simulate_nb execution time: 0.8365749999999821
    

# Cython


```python
%load_ext cython
```


```cython
%%cython

import pyximport

pyximport.install()

import matplotlib.pyplot as plt
import timeit
import numpy as np
cimport numpy as cnp
#import cython

def particle_initial_position(n):
    np.random.seed(0)
    p = np.random.rand(n, 2)
    for i in range(n):
        while True:
            conflict = False
            for j in range(i):
                distance = np.linalg.norm(p[i] - p[j])
                if distance <= 0.1:
                    conflict = True
                    break
            if not conflict:
                break
            p[i] = np.random.rand(2)
    return p


cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef simulate_cy( Py_ssize_t n, int iterations):
    cdef cnp.ndarray[double, ndim=2] p = particle_initial_position(n)
    cdef cnp.ndarray[double, ndim=2] fs
    cdef cnp.ndarray[double, ndim=1] r
    cdef double dist
    cdef cnp.ndarray[double, ndim=1] unit_vector
    cdef double epsilon = 1  # 势能参数
    cdef double sigma = 0.1  # 势能参数
    cdef double force
    cdef cnp.ndarray[double, ndim=2] x_delta
    cdef double[:,::1] pos
    cdef double x,y

    
    for _ in range(iterations):
        x_delta = fs = np.zeros(shape=(n, 2))  # forces from all other particles
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x = p[j,0] - p[i,0]
                    y = p[j,1] - p[i,1]
                    dist = (x**2+y**2)**0.5
                    
                    #unitvector is (ux,uy)
                    ux = x / dist
                    uy = y / dist
                    force = 48 * epsilon * (sigma**12) / (dist**13) - 24 * epsilon * (sigma**6) / (dist**7)
                    factor = 0.00001
                    fs[i,0] += -force * ux * factor
                    fs[i,1] += -force * uy * factor
        

            
        x_delta = np.zeros(shape=(n, 2))  # 创建与fs相同形状的数组
        for i in range(n):
            for j in range(2):
                x_delta[i, j] = fs[i, j] / 1
        
        
#        x_delta = fs / 1 * 0.00001
        p = update_position(p, x_delta)
        pos = p

    # plot finally result
#    print("P({}): ".format(iterations), p)
    
    return p



cpdef double clip(double a, double min_value, double max_value):
    return min(max(a, min_value), max_value)

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef update_position(cnp.ndarray[double, ndim=2] p, cnp.ndarray[double, ndim=2] delta_r, double minimum=0, double maximum=1):
#    print("p = \n", p)

    cdef Py_ssize_t i
    cdef cnp.ndarray[double, ndim=2] new_pos
    cdef double x, y
    cdef Py_ssize_t n = p.shape[0]

    new_pos = np.empty_like(p, dtype=np.float64)
    
    for i in range(n):
        x = p[i, 0] + delta_r[i, 0]
        y = p[i, 1] + delta_r[i, 1]

        if x > maximum or x < minimum:
            x = clip(x, minimum, maximum)

        if y > maximum or y < minimum:
            y = clip(y, minimum, maximum)

        new_pos[i, 0] = x
        new_pos[i, 1] = y

    return new_pos


compute_time_cy = timeit.timeit(lambda: simulate_cy(50, 100), number=1)

print("simulate_cy execution time:", compute_time_cy)
```

    simulate_cy execution time: 0.09075679999989461
    

# Cython: num_threads=4


```cython
%%cython --force -c=/openmp

import pyximport

pyximport.install()

import matplotlib.pyplot as plt
import timeit
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange

def particle_initial_position(n):
    np.random.seed(0)
    p = np.random.rand(n, 2)
    for i in range(n):
        while True:
            conflict = False
            for j in range(i):
                distance = np.linalg.norm(p[i] - p[j])
                if distance <= 0.1:
                    conflict = True
                    break
            if not conflict:
                break
            p[i] = np.random.rand(2)
    return p


cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef simulate_cy_threads_4(Py_ssize_t n, int iterations):
    cdef cnp.ndarray[double, ndim=2] p = particle_initial_position(n)
    cdef cnp.ndarray[double, ndim=2] fs = np.zeros(shape=(n, 2))
    cdef cnp.ndarray[double, ndim=2] x_delta = np.zeros(shape=(n, 2))
    cdef cnp.ndarray[double, ndim=2] pos = p.copy()
    cdef Py_ssize_t i, j
    cdef double x, y, dist, ux, uy, force, factor
    cdef double epsilon = 1  # 势能参数
    cdef double sigma = 0.1  # 势能参数
    
    for _ in range(iterations):
        fs[:, :] = 0.0

        for i in prange(n, num_threads=4, nogil=True):
            for j in range(n):
                if i != j:
                    x = pos[j, 0] - pos[i, 0]
                    y = pos[j, 1] - pos[i, 1]
                    dist = (x ** 2 + y ** 2) ** 0.5

                    ux = x / dist
                    uy = y / dist
                    force = 48 * epsilon * (sigma ** 12) / (dist ** 13) - 24 * epsilon * (sigma ** 6) / (dist ** 7)
                    factor = 0.00001
                    fs[i, 0] += -force * ux * factor
                    fs[i, 1] += -force * uy * factor

        x_delta[:, :] = 0.0
        for i in prange(n, num_threads=4, nogil=True):
            for j in range(2):
                x_delta[i, j] = fs[i, j] / 1.0

        pos = update_position(pos, x_delta)

    return pos



cpdef double clip(double a, double min_value, double max_value) nogil:
    return min(max(a, min_value), max_value)

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef update_position(cnp.ndarray[double, ndim=2] p, cnp.ndarray[double, ndim=2] delta_r, double minimum=0, double maximum=1):
#    print("p = \n", p)

    cdef Py_ssize_t i
    cdef cnp.ndarray[double, ndim=2] new_pos
    cdef double x, y
    cdef Py_ssize_t n = p.shape[0]

    new_pos = np.empty_like(p, dtype=np.float64)
    
    for i in prange(n, num_threads=4, nogil=True):
        x = p[i, 0] + delta_r[i, 0]
        y = p[i, 1] + delta_r[i, 1]

        if x > maximum or x < minimum:
            x = clip(x, minimum, maximum)

        if y > maximum or y < minimum:
            y = clip(y, minimum, maximum)

        new_pos[i, 0] = x
        new_pos[i, 1] = y

    return new_pos


compute_time_cy_threads_4 = timeit.timeit(lambda: simulate_cy_threads_4(50, 100), number=1)

print("simulate_cy execution time:", compute_time_cy_threads_4)
```

    simulate_cy execution time: 0.05473429999995005
    

# C


```python
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 50       //Number of particles
#define EPSILON 1.0
#define SIGMA 0.1
#define T_DELTA 0.00001
#define ITERATIONS 100

typedef struct {
    double x;
    double y;
} Vector;

void initialize_particles(Vector p[N]) {
    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        p[i].x = (double)rand() / RAND_MAX;
        p[i].y = (double)rand() / RAND_MAX;

        // Ensure that the spacing between particles is greater than 0.1
        for (int j = 0; j < i; j++) {
            double distance = sqrt(pow(p[j].x - p[i].x, 2) + pow(p[j].y - p[i].y, 2));
            if (distance <= 0.1) {
                i--;  // Regenerate coordinates
                break;
            }
        }
    }
}

void calculate_force(Vector p[N], Vector fs[N]) {
    for (int i = 0; i < N; i++) {
        fs[i].x = 0.0;
        fs[i].y = 0.0;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double r_x = p[j].x - p[i].x;
                double r_y = p[j].y - p[i].y;
                double dist = sqrt(r_x * r_x + r_y * r_y);

                double unit_vector_x = r_x / dist;
                double unit_vector_y = r_y / dist;

                double force = 48.0 * EPSILON * pow(SIGMA, 12) / pow(dist, 13) - 24.0 * EPSILON * pow(SIGMA, 6) / pow(dist, 7);

                fs[i].x += -force * unit_vector_x;
                fs[i].y += -force * unit_vector_y;
            }
        }
    }
}

void update_position(Vector p[N], Vector delta_r[N]) {
    for (int i = 0; i < N; i++) {
        p[i].x += delta_r[i].x;
        p[i].y += delta_r[i].y;
    }
}

void print_positions(Vector p[N]) {
    for (int i = 0; i < N; i++) {
        printf("P(%d): [%.8f, %.8f]\n", i, p[i].x, p[i].y);
    }
}

int main() {
    clock_t start_time, end_time;
    double total_time;

    start_time = clock();
    Vector p[N];
    Vector fs[N];
    Vector delta_r[N];

    initialize_particles(p);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        calculate_force(p, fs);

        for (int i = 0; i < N; i++) {
            delta_r[i].x = fs[i].x / 1 * T_DELTA;
            delta_r[i].y = fs[i].y / 1 * T_DELTA;
        }

        update_position(p, delta_r);
    }

    end_time = clock(); 

    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Code execution time：%f s\n", total_time);
//    print_positions(p);

    return 0;
}

```

simulate_C execution time: 0.058000

# Performance-comparison


```python
import pandas as pd
from IPython.display import HTML

data = {
    'Methods': ['Python', 'Numpy','Numba', 'Cython','Cython(threads=4)', 'C'],
    'Excution time(s)': [compute_time_py, compute_time_np, compute_time_nb, compute_time_cy, compute_time_cy_threads_4, 0.058000],
    'Speed up': [1, compute_time_py/compute_time_np, compute_time_py/compute_time_nb, compute_time_py/compute_time_cy, compute_time_py/compute_time_cy_threads_4, compute_time_py/0.058000]
}
df = pd.DataFrame(data)

# Creating style functions
def add_border(val):
    return 'border: 1px solid black'

# Applying style functions to data boxes
styled_df = df.style.applymap(add_border)

# Defining CSS styles
table_style = [
    {'selector': 'table', 'props': [('border-collapse', 'collapse')]},
    {'selector': 'th, td', 'props': [('border', '1px solid black')]}
]

# Adding styles to stylised data boxes
styled_df.set_table_styles(table_style)

# Displaying stylised data boxes in Jupyter Notebook
HTML(styled_df.to_html())
```




<style type="text/css">
#T_16c3a table {
  border-collapse: collapse;
}
#T_16c3a th {
  border: 1px solid black;
}
#T_16c3a  td {
  border: 1px solid black;
}
#T_16c3a_row0_col0, #T_16c3a_row0_col1, #T_16c3a_row0_col2, #T_16c3a_row1_col0, #T_16c3a_row1_col1, #T_16c3a_row1_col2, #T_16c3a_row2_col0, #T_16c3a_row2_col1, #T_16c3a_row2_col2, #T_16c3a_row3_col0, #T_16c3a_row3_col1, #T_16c3a_row3_col2, #T_16c3a_row4_col0, #T_16c3a_row4_col1, #T_16c3a_row4_col2, #T_16c3a_row5_col0, #T_16c3a_row5_col1, #T_16c3a_row5_col2 {
  border: 1px solid black;
}
</style>
<table id="T_16c3a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_16c3a_level0_col0" class="col_heading level0 col0" >Methods</th>
      <th id="T_16c3a_level0_col1" class="col_heading level0 col1" >Excution time(s)</th>
      <th id="T_16c3a_level0_col2" class="col_heading level0 col2" >Speed up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_16c3a_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_16c3a_row0_col0" class="data row0 col0" >Python</td>
      <td id="T_16c3a_row0_col1" class="data row0 col1" >23.236959</td>
      <td id="T_16c3a_row0_col2" class="data row0 col2" >1.000000</td>
    </tr>
    <tr>
      <th id="T_16c3a_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_16c3a_row1_col0" class="data row1 col0" >Numpy</td>
      <td id="T_16c3a_row1_col1" class="data row1 col1" >0.095371</td>
      <td id="T_16c3a_row1_col2" class="data row1 col2" >243.647293</td>
    </tr>
    <tr>
      <th id="T_16c3a_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_16c3a_row2_col0" class="data row2 col0" >Numba</td>
      <td id="T_16c3a_row2_col1" class="data row2 col1" >0.836575</td>
      <td id="T_16c3a_row2_col2" class="data row2 col2" >27.776301</td>
    </tr>
    <tr>
      <th id="T_16c3a_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_16c3a_row3_col0" class="data row3 col0" >Cython</td>
      <td id="T_16c3a_row3_col1" class="data row3 col1" >0.090757</td>
      <td id="T_16c3a_row3_col2" class="data row3 col2" >256.035461</td>
    </tr>
    <tr>
      <th id="T_16c3a_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_16c3a_row4_col0" class="data row4 col0" >Cython(threads=4)</td>
      <td id="T_16c3a_row4_col1" class="data row4 col1" >0.054734</td>
      <td id="T_16c3a_row4_col2" class="data row4 col2" >424.541085</td>
    </tr>
    <tr>
      <th id="T_16c3a_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_16c3a_row5_col0" class="data row5 col0" >C</td>
      <td id="T_16c3a_row5_col1" class="data row5 col1" >0.058000</td>
      <td id="T_16c3a_row5_col2" class="data row5 col2" >400.637226</td>
    </tr>
  </tbody>
</table>





```python

```
