import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from enum import Enum
from math import floor
from copy import deepcopy
from csv import writer

class Axis(Enum):
    x = 0
    y = 1

class BoundaryConditionType(Enum):
    abc1D = 0
    abc2D = 1

class CelerityMatrixType(Enum):
    m1 = 0
    m2 = 1
    m3 = 2
    m4 = 3
    m5 = 4
    m6 = 5

class TaskParameters():
    """Class to abstract parameters
    """
    def __init__(self, **kwargs) -> None:
        self.frequency: float = kwargs['frequency']
        self.x_length: int = kwargs['Lx']
        self.y_length: int = kwargs['Ly']
        self.delta_x: float = kwargs['dx']
        self.delta_y: float = kwargs['dy']
        self.source_x: int = kwargs['source_x']
        self.source_y: int = kwargs['source_y']
        self.t_max: float = kwargs['t_max']
        self.bc_type: BoundaryConditionType = kwargs['bc_type']
        
        self.delay: float = 1.0 / self.frequency
        self.x: np.array = np.arange(start=0, stop=self.x_length, step=self.delta_x)
        self.nx: int = len(self.x)
        self.y: np.array = np.arange(start=0, stop=self.y_length, step=self.delta_y)
        self.ny: int = len(self.y)

def sine_wave_source(amplitude: float, frequency: float, phase: float) -> float:
    return amplitude * np.sin(2 * np.pi * frequency * phase)

def get_multiplier(celerity: float, delta_t: float, delta_axis: float) -> float:
    """Get multiplier for the finite difference equation

    Args:
        celerity (float): value of celerity
        delta_t (float): time variation
        delta_axis (float): variation in x or y axis

    Returns:
        float: multiplier
    """
    return (pow(celerity, 2) * pow(delta_t, 2)) / pow(delta_axis, 2)

def abc_1d(i: int, j: int, celerities: np.ndarray, present_pressure: np.ndarray, delta_t: float, delta_axis: float, axis: Axis, is_limit: bool):
    """Absorbing boundary condition for one dimension

    Args:
        i (int): position in x axis
        j (int): position in y axis
        celerities (np.ndarray): celerity matrix
        present_pressure (np.ndarray): matrix of current values of pressure
        delta_t (float): time variation
        delta_axis (float): variation in x or y axis
        axis (Axis): evaluated axis
        is_limit (bool): True if on top or right limit, false otherwise

    Returns:
        float: pressure at (i, j)
    """
    limit_modifier: int = -1 if is_limit else 1
    pressure: float = (present_pressure[i][j] + limit_modifier * celerities[i][j] * delta_t) / delta_axis
    
    i2: int = i + 1 if axis == Axis.x and not is_limit else i
    j2: int = j + 1 if axis == Axis.y and not is_limit else j
    
    i3: int = i - 1 if axis == Axis.x and is_limit else i
    j3: int = j - 1 if axis == Axis.y and is_limit else j
    
    pressure = pressure * (present_pressure[i2][j2] - present_pressure[i3][j3])
    
    return pressure

def abc_2d(i: int, j: int, celerities: np.ndarray, present_pressure: np.ndarray, previous_pressure: np.ndarray, delta_t: float, delta_x: float, delta_y: float, axis: Axis, is_limit: bool) -> float:
    """Absorbing boundary condition for two dimensions

    Args:
        i (int): position in x axis
        j (int): position in y axis
        celerities (np.ndarray): celerity matrix
        present_pressure (np.ndarray): matrix of current values of pressure
        previous_pressure (np.ndarray): matrix of previous values of pressure
        delta_t (float): time variation
        delta_x (float): variation in x axis
        delta_y (_type_): variation in y axis
        axis (Axis): evaluated axis
        is_limit (bool): True if on top or right limit, false otherwise

    Returns:
        float: pressure at (i, j)
    """
    limit_modifier: int = -1 if is_limit else 1
    
    i2: int = i - 1 if axis == Axis.x and is_limit else i
    j2: int = j - 1 if axis == Axis.y and is_limit else j
    
    c: float = celerities[i2][j2]
    
    i3: int = i + 1 if axis == Axis.x and not is_limit else i
    j3: int = j + 1 if axis == Axis.y and not is_limit else j
    
    i4: int = i + 1 if axis == Axis.y else i
    j4: int = j + 1 if axis == Axis.x else j
    
    i5: int = i - 1 if axis == Axis.y else i
    j5: int = j - 1 if axis == Axis.x else j
    
    pressure: float = (present_pressure[i, j] - previous_pressure[i][j] + limit_modifier * c * delta_t) / delta_x
    pressure = pressure * (present_pressure[i3][j3] - previous_pressure[i3][j3] - present_pressure[i2][j2] - previous_pressure[i2][j2] - 2 * present_pressure[i][j])
    pressure = pressure + 0.5 * (delta_t / delta_y) * pow(c, 2) * (present_pressure[i4][j4] - 2 * present_pressure[i][j] + present_pressure[i5][j5])
    
    return pressure

def abc_2d_corner(i: int, j: int, celerities: np.ndarray, previous_pressure: np.ndarray, delta_t: float, delta_y: float, is_limit: bool) -> float:
    """Calculate value for the matrix's corner elements

    Args:
        i (int): position in x axis
        j (int): position in y axis
        celerities (np.ndarray): celerity matrix
        previous_pressure (np.ndarray): matrix of previous values of pressure
        delta_t (float): variation in time
        delta_y (float): variation in y axis
        is_limit (bool): True if on top limit, false otherwise

    Returns:
        float: pressure at (i, j)
    """
    limit_modifier: int = -1 if is_limit else 1
    
    j2: int = j - 1 if is_limit else j
    j3: int = j if is_limit else j + 1
    
    pressure: float = (previous_pressure[i][j] + limit_modifier * celerities[i][j2] * delta_t) / delta_y
    pressure = pressure * (previous_pressure[i][j3] - previous_pressure[i][j2])
    
    return pressure

def finite_difference(i: int, j: int, celerities: np.ndarray, delta_t: float, delta_x: float, delta_y: float, present_pressure: np.ndarray, previous_pressure: np.ndarray) -> float:
    """Calculate unknown pressure at (i, j)

    Args:
        i (int): position in x axis
        j (int): position in y axis
        celerities (np.ndarray): celerity matrix
        delta_t (float): time variation
        delta_x (float): variation in x axis
        delta_y (float): variation in y axis
        present_pressure (np.ndarray): matrix of current values of pressure
        previous_pressure (np.ndarray): matrix of previous values of pressure

    Returns:
        float: Pressure calculated at (i, j)
    """
    mult_1: float = get_multiplier(celerities[i][j], delta_t, delta_x)
    mult_2: float = get_multiplier(celerities[i - 1][j], delta_t, delta_x)
    mult_3: float = get_multiplier(celerities[i][j], delta_t, delta_y)
    mult_4: float = get_multiplier(celerities[i][j - 1], delta_t, delta_y)
    
    pressure_ij: float = present_pressure[i][j]
    
    return (
        2 * present_pressure[i][j] - previous_pressure[i][j] +
        mult_1 * (present_pressure[i + 1][j] - pressure_ij) -
        mult_2 * (pressure_ij - present_pressure[i - 1][j]) +
        mult_3 * (present_pressure[i][j + 1] - pressure_ij) -
        mult_4 * (pressure_ij - present_pressure[i - 1][j])
    )

def celerity_homogeneous(nx: int, ny: int, value: float) -> np.ndarray:
    """Build homogenous celerity matrix

    Args:
        nx (int): size of x axis
        ny (int): size of y axis
        value (float): value for all elements

    Returns:
        np.ndarray: celerity matrix
    """
    return np.full(shape=(ny, nx), fill_value=value, dtype=float)

def celerity_m1(nx: int, ny: int) -> np.ndarray:
    """Build celerity matrix m1

    Args:
        nx (int): size of x axis
        ny (int): size of y axis

    Returns:
        np.ndarray: celerity matrix
    """
    celerities: np.ndarray = np.full(shape=(ny, nx), fill_value=4000, dtype=float)
    
    for i in range(250):
        for j in range(nx):
            celerities[i][j] = 2000
    
    return celerities

def celerity_m2(nx: int, ny: int) -> np.ndarray:
    """Build celerity matrix m1

    Args:
        nx (int): size of x axis
        ny (int): size of y axis

    Returns:
        np.ndarray: celerity matrix
    """
    celerities: np.ndarray = np.full(shape=(ny, nx), fill_value=4000, dtype=float)
    
    for j in range(nx):
        for i in range(100):
            celerities[i][j] = 2000
        
        for i in range(100, 200):
            celerities[i][j] = 2500
        
        for i in range(200, 300):
            celerities[i][j] = 3000
        
        for i in range(300, 400):
            celerities[i][j] = 3250
        
        for i in range(400, 500):
            celerities[i][j] = 3500
    
        for i in range(500, 600):
            celerities[i][j] = 4000
    
    return celerities

def celerity_m3(nx: int, ny: int) -> np.ndarray:
    """Build celerity matrix m1

    Args:
        nx (int): size of x axis
        ny (int): size of y axis

    Returns:
        np.ndarray: celerity matrix
    """
    celerities: np.ndarray = np.full(shape=(ny, nx), fill_value=4000, dtype=float)
    
    for i in range(200):
        for j in range(350):
            celerities[i][j] = 2000
    
        for j in range(350, 700):
            celerities[i][j] = 2250
            
        for j in range(700, nx):
            celerities[i][j] = 2500
    
    for i in range(200, 400):
        for j in range(350):
            celerities[i][j] = 2750
    
        for j in range(350, 700):
            celerities[i][j] = 3000
            
        for j in range(700, nx):
            celerities[i][j] = 3250
    
    for i in range(400, ny):
        for j in range(350):
            celerities[i][j] = 3500
    
        for j in range(350, 700):
            celerities[i][j] = 3750
    
    return celerities

def celerity_m4(nx: int, ny: int) -> np.ndarray:
    """Build celerity matrix m1

    Args:
        nx (int): size of x axis
        ny (int): size of y axis

    Returns:
        np.ndarray: celerity matrix
    """
    celerities: np.ndarray = np.full(shape=(ny, nx), fill_value=4000, dtype=float)
    
    for j in range(nx):
        for i in range(115):
            celerities[i][j] = 2000
        
        for i in range(115, 231):
            celerities[i][j] = 2500
        
        for i in range(231, 324):
            celerities[i][j] = 3000
    
    for i in range(170, 286):
        for j in range(399, 690):
            celerities[i][j] = 4000
    
    return celerities

def celerity_m5(nx: int, ny: int) -> np.ndarray:
    """Build celerity matrix m1

    Args:
        nx (int): size of x axis
        ny (int): size of y axis

    Returns:
        np.ndarray: celerity matrix
    """
    celerities: np.ndarray = np.full(shape=(ny, nx), fill_value=4000, dtype=float)
    
    for i in range(ny):
        for j in range(nx):
            if i < 200 or j >= 549 and i < 300:
                celerities[i][j] = 2000
            elif (j < 450 and i < 400) or (j >= 449 and i < 400 and i >= 299) or (j >= 749 and i >= 399 and i < 500):
                celerities[i][j] = 3000
            
            if i >= 199 and i < 300 and j >= 499 and j < 550:
                celerities[i][j] = 2000 if j - 450 >= i - 200 else 3000
    
    ##for i=1:nx
    ##  for j=1:ny
    ##    if(y(j)<=200 || (x(i)>=550 && y(j)<=300))
    ##      cw(j,i) = 2000;
    ##    elseif((x(i)<=450 && y(j)<=400) || (x(i)>=450 && y(j)<=400 && y(j)>=300) || (x(i)>= 750 && y(j)>=400 && y(j)<=500))
    ##      cw(j,i) = 3000;
    ##    elseif((x(i)<=650 && y(j)>=400) || y(j)>=500)
    ##      cw(j,i) = 4000;
    ##    endif
    ##    
    ##    if(y(j)>=200 && y(j)<=300 && x(i)>=450 && x(i)<=550)
    ##      if(x(i)-450>=y(j)-200)
    ##        cw(j,i)=2000;
    ##      else
    ##        cw(j,i)=3000;
    ##      endif
    ##    endif
    ##    
    ##    if(y(j)>=400 && y(j)<=500 && x(i)>=650 && x(i)<=750)
    ##      if(x(i)-650>=y(j)-400)
    ##        cw(j,i)=3000;
    ##      else
    ##        cw(j,i)=4000;
    ##      endif
    ##    endif
    ##    
    ##  endfor
    ##endfor
    
    return celerities

def celerity_m6(nx: int, ny: int) -> np.ndarray:
    """Build celerity matrix m1

    Args:
        nx (int): size of x axis
        ny (int): size of y axis

    Returns:
        np.ndarray: celerity matrix
    """
    celerities: np.ndarray = np.full(shape=(ny, nx), fill_value=4000, dtype=float)
    
    for j in range(nx):
        for i in range(114):
            celerities[i][j] = 2000
    
    for i in range(114, 207):
        for j in range(173, 928):
            celerities[i][j] = 2000
    
    for i in range(114, 138):
        for j in range(115, 986):
            celerities[i][j] = 2000
    
    for i in range(207, 264):
        for j in range(231, 870):
            celerities[i][j] = 3000
    
    for i in range(264, 334):
        for j in range(289, 812):
            celerities[i][j] = 3000
    
    for i in range(334, 381):
        for j in range(347, 754):
            celerities[i][j] = 3000
    
    for i in range(207, 264):
        for j in range(231, 870):
            celerities[i][j] = 3000
    
    return celerities

def build_celerity_matrix(type: CelerityMatrixType, nx: int, ny: int) -> np.ndarray:
    """Build celerity matrix

    Args:
        type (CelerityMatrixType): type of the matrix
        nx (int): size of x axis
        ny (int): size of y axis

    Returns:
        np.ndarray: celerity matrix
    """
    if type == CelerityMatrixType.m1:
        return celerity_m1(nx, ny)
    elif type == CelerityMatrixType.m2:
        return celerity_m2(nx, ny)
    elif type == CelerityMatrixType.m3:
        return celerity_m3(nx, ny)
    elif type == CelerityMatrixType.m4:
        return celerity_m4(nx, ny)
    elif type == CelerityMatrixType.m5:
        return celerity_m5(nx, ny)
    
    return celerity_m6(nx, ny)

def get_delta_t(delta_x: float, delta_y: float, celerities: np.ndarray) -> float:
    """Calculate time variation

    Args:
        delta_x (float): variation in x axis
        delta_y (float): variation in y axis
        celerities (np.ndarray): celerity matrix

    Returns:
        float: time variation
    """
    delta_s: float = np.sqrt(pow(delta_x, 2) + pow(delta_y, 2))
    c_max: float = celerities.max()
    coeff: float = 1
    
    return coeff * (delta_x * delta_y) / (delta_s * c_max)

def compressibility_module() -> float:
    """Return compressibility module

    Returns:
        float: value of compressibility module
    """
    return 2.25 * pow(10, 9)

def validate_boundary_condition() -> None:
    params: TaskParameters = TaskParameters(
        frequency = 30,
        Lx = 500,
        Ly = 500,
        dx = 1.0,
        dy = 1.0,
        source_x = 0,
        source_y = 0,
        t_max = 0.8325,
        bc_type = BoundaryConditionType.abc1D
    )
    
    celerities: np.ndarray = celerity_homogeneous(params.nx, params.ny, 2000)
    sns.heatmap(celerities)
    
    delta_t: float = get_delta_t(params.delta_x, params.delta_y, celerities)
    max_iterations: int = floor(params.t_final / delta_t)
    
    previous_pressure: np.ndarray = np.zeros(shape=(params.ny, params.nx), dtype=float)
    present_pressure: np.ndarray = np.zeros(shape=(params.ny, params.nx), dtype=float)
    next_pressure: np.ndarray = np.zeros(shape=(params.ny, params.nx), dtype=float)
    
    compr_module: float = compressibility_module()
    
    with open('sismogram.csv', 'a') as file:
        sismogram_writer = writer(file)
        
        for iter in range(max_iterations):
            t: int = iter * delta_t
            
            if params.bc_type == BoundaryConditionType.abc1D:
                for i in range(params.ny):
                    next_pressure[i][0] = abc_1d(i, 0, celerities, present_pressure, delta_t, params.delta_x, Axis.x, False)
                    next_pressure[i][params.nx - 1] = abc_1d(i, params.nx - 1, celerities, present_pressure, delta_t, params.delta_x, Axis.x, True)
                
                for j in range(params.nx):
                    next_pressure[0][j] = abc_1d(0, j, celerities, present_pressure, delta_t, params.delta_y, Axis.y, False)
                    next_pressure[params.ny - 1][j] = abc_1d(params.ny - 1, j, celerities, present_pressure, delta_t, params.delta_y, Axis.y, True)
            else:
                present_pressure[0][0] = abc_2d_corner(0, 0, celerities, previous_pressure, delta_t, params.delta_y, False)
                present_pressure[0][params.nx - 1] = abc_2d_corner(0, params.nx - 1, celerities, previous_pressure, delta_t, params.delta_y, True)
                present_pressure[params.ny - 1][0] = abc_2d_corner(params.ny - 1, 0, celerities, previous_pressure, delta_t, params.delta_y, False)
                present_pressure[params.ny - 1][params.nx - 1] = abc_2d_corner(params.ny - 1, params.nx - 1, celerities, previous_pressure, delta_t, params.delta_y, True)
        
                for i in range(1, params.ny - 1):
                    next_pressure[i][0] = abc_2d(i, 0, celerities, present_pressure, previous_pressure, delta_t, params.delta_x, params.delta_y, Axis.x, False)
                    next_pressure[i][params.nx - 1] = abc_2d(i, params.nx - 1, celerities, present_pressure, previous_pressure, delta_t, params.delta_x, params.delta_y, Axis.x, True)
        
                for j in range(1, params.nx - 1):
                    next_pressure[0][j] = abc_2d(0, j, celerities, present_pressure, previous_pressure, delta_t, params.delta_x, params.delta_y, Axis.y, False)
                    next_pressure[params.ny - 1][j] = abc_2d(params.ny - 1, j, celerities, present_pressure, previous_pressure, delta_t, params.delta_x, params.delta_y, Axis.y, True)

            for i in range(1, params.ny - 1):
                for j in range(1, params.nx - 1):
                    next_pressure[i][j] = finite_difference(i, j, celerities, delta_t, params.delta_x, params.delta_y, present_pressure, previous_pressure)
            
            next_pressure[params.source_x][params.source_y] = next_pressure[params.source_x][params.source_y] + sine_wave_source(-pow(delta_t, 2), params.frequency, params.delay)
            
            previous_pressure = deepcopy(present_pressure)
            present_pressure = deepcopy(next_pressure)
            
            # sismogram_writer.writerow([compr_module * present_pressure[params.source_x][j] for j in range(params.ny)])
            
        xv, yv = np.meshgrid(params.x, params.y, indexing='ij')
        plt.surf
        
            

        

def main():
    # Parameters
    validate_boundary_condition()
    
    
    # previous_pressure: np.ndarray = np.zeros(shape=(nx, ny), dtype=float)
    # present_pressure: np.ndarray = np.zeros(shape=(nx, ny), dtype=float)
    # next_pressure: np.ndarray = np.zeros(shape=(nx, ny), dtype=float)
    
    # delta_t = 1
    
    
    
    # if bc_type == BoundaryConditionType.abc1D:
    #     for i in range(nx):
    #         next_pressure[i][0] = abc_1d(i, 0, celerities, present_pressure, delta_t, delta_x, Axis.x, False)
    #         next_pressure[i][ny - 1] = abc_1d(i, ny - 1, celerities, present_pressure, delta_t, delta_x, Axis.x, True)
        
    #     for j in range(ny):
    #         next_pressure[0][j] = abc_1d(0, j, celerities, present_pressure, delta_t, delta_y, Axis.y, False)
    #         next_pressure[nx - 1][j] = abc_1d(nx - 1, j, celerities, present_pressure, delta_t, delta_y, Axis.y, True)
    # else:
    #     present_pressure[0][0] = abc_2d_corner(0, 0, celerities, previous_pressure, delta_t, delta_y, False)
    #     present_pressure[0][ny - 1] = abc_2d_corner(0, ny - 1, celerities, previous_pressure, delta_t, delta_y, True)
    #     present_pressure[nx - 1][0] = abc_2d_corner(nx - 1, 0, celerities, previous_pressure, delta_t, delta_y, False)
    #     present_pressure[nx - 1][ny - 1] = abc_2d_corner(nx - 1, ny - 1, celerities, previous_pressure, delta_t, delta_y, True)
  
    #     for i in range(1, nx):
    #         next_pressure[i][0] = abc_2d(i, 0, celerities, present_pressure, previous_pressure, delta_t, delta_x, delta_y, Axis.x, False)
    #         next_pressure[i][ny - 1] = abc_2d(i, ny - 1, celerities, present_pressure, previous_pressure, delta_t, delta_x, delta_y, Axis.x, True)
  
    #     for j in range(1, ny):
    #         next_pressure[0][j] = abc_2d(0, j, celerities, present_pressure, previous_pressure, delta_t, delta_x, delta_y, Axis.y, False)
    #         next_pressure[nx - 1][j] = abc_2d(nx - 1, j, celerities, present_pressure, previous_pressure, delta_t, delta_x, delta_y, Axis.y, True)
        
          

main()