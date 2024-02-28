############################## Import Needed Dependencies ################################
import dolfin as df
import fenics as fe
from dolfin import (
    NonlinearProblem, UserExpression, MeshFunction, FunctionSpace, Function, MixedElement,
    TestFunctions, TrialFunction, split, derivative, NonlinearVariationalProblem,
    NonlinearVariationalSolver, cells, grad, project, refine, Point, RectangleMesh,
    as_vector, XDMFFile, DOLFIN_EPS, sqrt, conditional, Constant, inner, Dx, lt,
    set_log_level, LogLevel, MPI, UserExpression, LagrangeInterpolator
)

import numpy as np
import matplotlib.pyplot as plt
import time
from modadthermo import refine_mesh
from tqdm import tqdm


set_log_level(LogLevel.ERROR)

#################### Define Function For Lcocal Refine #################


def refine_mesh_local( mesh , rad , center , Max_level  ): 

    xc , yc = center

    mesh_itr = mesh

    for i in range(Max_level):

        mf = MeshFunction("bool", mesh_itr, mesh_itr.topology().dim() , False )

        cells_mesh = cells( mesh_itr )


        index = 0 

        for cell in cells_mesh :

            if ( cell.midpoint()[0] - xc ) **2  + ( cell.midpoint()[1] - yc ) **2  <   10 * rad**2 : 





                mf.array()[ index ] = True

            index = index + 1 


        mesh_r = refine( mesh_itr, mf )

        # Update for next loop
        mesh_itr = mesh_r


    return mesh_itr 


#############################  END  ################################


#################### Define Parallel Variables ####################

# Get the global communicator
comm = MPI.comm_world 

# Get the rank of the process
rank = MPI.rank(comm)

# Get the size of the communicator (total number of processes)
size = MPI.size(comm)

#############################  END  ################################


#################### Define Constants  #############################

parameters = {
    'dt': 0.0048,
    "dy": 0.4,     
    "Nx_aprox": 500,
    "Ny_aprox": 500,
    "max_level": 5, # Maximum level of Coarsening
    'w0': 1,
    'Tau_0': 1,
    'at': lambda: 1 / (2 * fe.sqrt(2.0)),
    'ep_4': 0.02,
    'k_eq': 0.15,
    'lamda': 12.7653,
    'Delta': 0.55,
    'm_c_inf': 0.5325,
    'opk': lambda k_eq: 1 + k_eq,
    'omk': lambda k_eq: 1 - k_eq,
    'a1': 0.8839,
    'a2': 0.6267,
    'le': 1,
    'd': lambda a2, lamda: a2 * lamda,
    'alpha': lambda d, le: d * le,
    'd0': lambda a1, lamda: a1 / lamda,
    'Initial_Circle_radius': lambda d0: 65 * d0,
    'rad': lambda Initial_Circle_radius: Initial_Circle_radius,
    'center': [0.0, 0.0],
    "abs_tol": 1e-5,
    "rel_tol": 1e-4,
}

# To access and compute values with dependencies, call the lambda functions with necessary arguments
parameters['at'] = parameters['at']()  # No dependencies
parameters['opk'] = parameters['opk'](parameters['k_eq'])
parameters['omk'] = parameters['omk'](parameters['k_eq'])
parameters['d'] = parameters['d'](parameters['a2'], parameters['lamda'])
parameters['alpha'] = parameters['alpha'](parameters['d'], parameters['le'])
parameters['d0'] = parameters['d0'](parameters['a1'], parameters['lamda'])
parameters['Initial_Circle_radius'] = parameters['Initial_Circle_radius'](parameters['d0'])
parameters['rad'] = parameters['rad'](parameters['Initial_Circle_radius'])


# Retrieve parameters individually from the dictionary
w0 = parameters['w0']
Tau_0 = parameters['Tau_0']
at = parameters['at']
ep_4 = parameters['ep_4']
k_eq = parameters['k_eq']
lamda = parameters['lamda']
Delta = parameters['Delta']
m_c_inf = parameters['m_c_inf']
opk = parameters['opk']
omk = parameters['omk']
a1 = parameters['a1']
a2 = parameters['a2']
le = parameters['le']
d = parameters['d']  # Assuming 'd' is calculated as a2 * lamda
alpha = parameters['alpha']  # Assuming 'alpha' is calculated as d * le
d0 = parameters['d0']  # Assuming 'd0' is calculated as a1 / lamda
Initial_Circle_radius = parameters['Initial_Circle_radius']  # Assuming calculated from 'd0'
rad = parameters['rad']  # Assuming 'rad' equals 'Initial_Circle_radius'
center = parameters['center']
rel_tol = parameters['rel_tol']
abs_tol = parameters['abs_tol']
dt = parameters['dt']
Nx_aprox = parameters['Nx_aprox']
Ny_aprox = parameters['Ny_aprox']
dy = parameters['dy']
max_level = parameters['max_level']

#############################  END  ################################


#################### Define Mesh Domain Parameters ############################

dy_coarse = 2**( max_level ) * dy

nx = (int)(Nx_aprox / dy_coarse ) + 1
ny = (int)(Ny_aprox / dy_coarse ) + 1

nx = nx + 1
ny = ny + 1 

Nx = nx * dy_coarse
Ny = ny * dy_coarse

nx = (int)(Nx / dy_coarse )
ny = (int)(Ny / dy_coarse )




#############################  END  ################################

########################## Define Mesh  ##################################

coarse_mesh = fe.RectangleMesh( Point(0, 0), Point(Nx, Ny), nx, ny)

mesh = refine_mesh_local( coarse_mesh , rad , center , max_level  )

# Printing Initial Mesh Informations 

# Calculate the number of cells in each mesh across all processes
number_of_coarse_mesh_cells = df.MPI.sum(comm, coarse_mesh.num_cells())
number_of_small_mesh_cells = df.MPI.sum(comm, mesh.num_cells())

if rank == 0:
    # Calculate and print details about the mesh sizes and number of cells
    min_dx_coarse = coarse_mesh.hmin() / df.sqrt(2)
    min_dx_small = mesh.hmin() / df.sqrt(2)

    print(f"Minimum Δx of Coarse Mesh = {min_dx_coarse}")
    print(f"Number Of Coarse Mesh Cells: {number_of_coarse_mesh_cells}")
    print(f"Minimum Δx of Small Mesh = {min_dx_small}")
    print(f"Number Of Small Mesh Cells: {number_of_small_mesh_cells}")


#############################  END  ####################################

#################### Define Initial Condition  ####################

# Initial Condition :
class InitialConditions(UserExpression):
    rad = Initial_Circle_radius

    def eval(self, values, x):
        xc = x[0]
        yc = x[1]
        dist = xc**2 + yc**2

        values[0] = -np.tanh((fe.sqrt(dist) - self.rad) / (fe.sqrt(2.0)))

        values[1] = 0

        values[2] = - Delta

    def value_shape(self):
        return (3,)
    
def Initial_Interpolate(Phi_U, Phi_U_0):
    initial_v = InitialConditions(degree=2)

    Phi_U.interpolate(initial_v)

    Phi_U_0.interpolate(initial_v)


#############################  END  ###############################


#################### Define Variables  ################################


def define_variables(mesh):
    # Define finite elements for each variable in the system
    P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Order parameter Phi
    P2 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # U: dimensionless solute supersaturation
    P3 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Theta: dimensionless Temperature

    # Create a mixed element and define the function space
    element = fe.MixedElement([P1, P2, P3])
    function_space = fe.FunctionSpace(mesh, element)

    # Define test functions
    test_1, test_2, test_3 = fe.TestFunctions(function_space)

    # Define functions for the current and previous solutions
    solution_vector = fe.Function(function_space)    # Current solution
    solution_vector_0 = fe.Function(function_space)  # Previous solution


    # Split functions to access individual components
    Phi_answer, U_answer, Theta_answer = fe.split(solution_vector)    # Current solution
    Phi_prev, U_prev, Theta_prev = fe.split(solution_vector_0)        # Last step solution

    # Collapse function spaces to individual subspaces
    num_subs = function_space.num_sub_spaces()
    spaces, maps = [], []
    for i in range(num_subs):
        space_i, map_i = function_space.sub(i).collapse(collapsed_dofs=True)
        spaces.append(space_i)
        maps.append(map_i)

    # Return all the variables
    return {
        'Theta_answer': Theta_answer, 'Theta_prev': Theta_prev,
        'Phi_answer': Phi_answer, 'U_answer': U_answer,
        'Phi_prev': Phi_prev, 'U_prev': U_prev,
        'solution_vector': solution_vector, 'solution_vector_0': solution_vector_0,
        'test_2': test_2, 'test_1': test_1, 'test_3': test_3,
        'spaces': spaces, 'function_space': function_space
    }


def calculate_dependent_variables(variables_dict, parameters):

    # Retrieve the values from the dictionary
    w0 = parameters['w0']
    ep_4 = parameters['ep_4']
    # Retrieve the values from the dictionary
    phi_answer = variables_dict['Phi_answer']

    # Define tolerance for avoiding division by zero errors
    tolerance_d = fe.sqrt(DOLFIN_EPS)  # sqrt(1e-15)

    # Calculate gradient and derivatives for anisotropy function
    grad_phi = fe.grad(phi_answer)
    mgphi = fe.inner(grad_phi, grad_phi)
    dpx = fe.Dx(phi_answer, 0)
    dpy = fe.Dx(phi_answer, 1)
    dpx = fe.variable(dpx)
    dpy = fe.variable(dpy)

    # Normalized derivatives
    nmx = -dpx / fe.sqrt(mgphi)
    nmy = -dpy / fe.sqrt(mgphi)
    norm_phi_4 = nmx**4 + nmy**4

    # Anisotropy function
    a_n = fe.conditional(
        fe.lt(fe.sqrt(mgphi), fe.sqrt(DOLFIN_EPS)),
        fe.Constant(1 - 3 * ep_4),
        1 - 3 * ep_4 + 4 * ep_4 * norm_phi_4
    )

    # Weight function based on anisotropy
    W_n = w0 * a_n

    # Derivatives of weight function w.r.t x and y
    D_w_n_x = fe.conditional(fe.lt(fe.sqrt(mgphi), tolerance_d), 0, fe.diff(W_n, dpx))
    D_w_n_y = fe.conditional(fe.lt(fe.sqrt(mgphi), tolerance_d), 0, fe.diff(W_n, dpy))

    return  {
        'D_w_n_x': D_w_n_x,
        'D_w_n_y': D_w_n_y,
        'mgphi': mgphi,
        'W_n': W_n
    }


def calculate_equation_1(variables_dict, dep_var_dict, parameters):

    # Retrieve parameters individually from the dictionary
    w0 = parameters['w0']
    k_eq = parameters['k_eq']
    lamda = parameters['lamda']
    m_c_inf = parameters['m_c_inf']
    le = parameters['le']
    d = parameters['d'] 
    dt = parameters['dt']
    # Retrieve the values from the dictionary
    theta_answer = variables_dict['Theta_answer']
    theta_prev = variables_dict['Theta_prev']
    phi_answer = variables_dict['Phi_answer']
    u_answer = variables_dict['U_answer']
    phi_prev = variables_dict['Phi_prev']
    v_test = variables_dict['test_1']
    # retrive dependent variables
    d_w_n_x = dep_var_dict['D_w_n_x']
    d_w_n_y = dep_var_dict['D_w_n_y']
    mgphi = dep_var_dict['mgphi']
    w_n = dep_var_dict['W_n']


    term4_in = mgphi * w_n * d_w_n_x
    term5_in = mgphi * w_n * d_w_n_y

    term4 = -fe.inner(term4_in, v_test.dx(0)) * fe.dx
    term5 = -fe.inner(term5_in, v_test.dx(1)) * fe.dx

    term3 = -(w_n**2 * fe.inner(fe.grad(phi_answer), fe.grad(v_test))) * fe.dx

    term2 = (
        fe.inner(
            (phi_answer - phi_answer**3) - lamda * (u_answer * m_c_inf + theta_answer) * (1 - phi_answer**2) ** 2,
            v_test,
        ) * fe.dx
    )

    tau_n = (w_n / w0) ** 2 * (1/le + m_c_inf * (1 + (1 - k_eq) * u_answer))

    term1 = -fe.inner((tau_n) * (phi_answer - phi_prev) / dt, v_test) * fe.dx

    eq1 = term1 + term2 + term3 + term4 + term5

    return eq1


def calculate_equation_2(variables_dict, parameters):
    
    # Retrieve the values from the dictionary
    at = parameters['at']
    opk = parameters['opk']
    omk = parameters['omk']
    d = parameters['d']  
    dt = parameters['dt']
    # Retrieve the values from the dictionary
    phi_answer = variables_dict['Phi_answer']
    u_answer = variables_dict['U_answer']
    phi_prev = variables_dict['Phi_prev']
    u_prev = variables_dict['U_prev']
    q_test = variables_dict['test_2']


    tolerance_d = fe.sqrt(DOLFIN_EPS)  # sqrt(1e-15)

    grad_phi = fe.grad(phi_answer)
    abs_grad = fe.sqrt(fe.inner(grad_phi, grad_phi))

    norm = fe.conditional(
        fe.lt(abs_grad, tolerance_d), fe.as_vector([0, 0]), grad_phi / abs_grad
    )

    dphidt = (phi_answer - phi_prev) / dt

    term6 = -fe.inner(((opk) / 2 - (omk) * phi_answer / 2) * (u_answer - u_prev) / dt, q_test) * fe.dx
    term7 = -fe.inner(d * (1 - phi_answer) / 2 * fe.grad(u_answer), fe.grad(q_test)) * fe.dx
    term8 = -at * (1 + (omk) * u_answer) * dphidt * fe.inner(norm, fe.grad(q_test)) * fe.dx
    term9 = (1 + (omk) * u_answer) * dphidt / 2 * q_test * fe.dx

    eq2 = term6 + term7 + term8 + term9

    return eq2


def calculate_equation_3(variables_dict, parameters):

    # Retrieve the values from the dictionary
    alpha = parameters['alpha']  
    dt = parameters['dt']
    # Retrieve the values from the dictionary
    theta_answer = variables_dict['Theta_answer']
    theta_prev = variables_dict['Theta_prev']
    phi_answer = variables_dict['Phi_answer']
    phi_prev = variables_dict['Phi_prev']
    w_test = variables_dict['test_3']


    term = (
        fe.inner((theta_answer - theta_prev) / dt, w_test) +
        fe.inner(fe.grad(theta_answer), fe.grad(w_test)) * alpha +
        - 0.5 * fe.inner((phi_answer - phi_prev) / dt, w_test)
    ) * fe.dx

    return term



def define_problem(eq1, eq2, eq3, phi_u, parameters):
    
    rel_tol = parameters['rel_tol']
    abs_tol = parameters['abs_tol']


    L = eq1 + eq2 + eq3  # Define the Lagrangian
    J = derivative(L, phi_u)  # Compute the Jacobian

    # Define the problem
    problem = NonlinearVariationalProblem(L, phi_u, J=J)

    # Create and configure the solver
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["relative_tolerance"] = rel_tol
    prm["newton_solver"]["absolute_tolerance"] = abs_tol
    # prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = nonzero_initial_guess

    return solver



def update_solver_on_new_mesh(mesh_new, parameters, old_solution_vector= None, old_solution_vector_0=None):

    # Define variables on the new mesh
    variables_dict = define_variables(mesh_new)
    # Extract each variable from the dictionary

    solution_vector = variables_dict['solution_vector']
    solution_vector_0 = variables_dict['solution_vector_0']
    spaces = variables_dict['spaces']

    # Interpolate previous step solutions from old mesh functions else interpolate initial condition
    if old_solution_vector is not None and old_solution_vector_0 is not None:

        LagrangeInterpolator.interpolate(solution_vector, old_solution_vector)
        LagrangeInterpolator.interpolate(solution_vector_0, old_solution_vector_0)

    else:
        # Interpolate initial condition
        Initial_Interpolate(solution_vector, solution_vector_0)


    # Calculate dependent variables
    dep_var_dict = calculate_dependent_variables(variables_dict, parameters)


    # Define equations
    eq1 = calculate_equation_1(variables_dict, dep_var_dict, parameters) 
        
    eq2 = calculate_equation_2(variables_dict, parameters)

    eq3 = calculate_equation_3(variables_dict, parameters)

    # Define problem
    solver = define_problem(eq1, eq2, eq3, solution_vector, parameters)

    # Return whatever is appropriate, such as the solver or a status message
    return solver, solution_vector, solution_vector_0, spaces

#############################  END  #############################


#################### Define Step 1 For Solving  ####################

solver, solution_vector, solution_vector_0, spaces = update_solver_on_new_mesh(mesh, parameters)


#############################  END  ###############################



############################ File Section #########################


file = fe.XDMFFile("Thermoslutal_Beluga_04_lamda_127653.xdmf" ) # File Name To Save #


def write_simulation_data(Sol_Func, time, file, variable_names ):

    
    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    # Split the combined function into its components
    functions = Sol_Func.split(deepcopy=True)

    # Check if the number of variable names matches the number of functions
    if variable_names and len(variable_names) != len(functions):
        raise ValueError("The number of variable names must match the number of functions.")

    # Rename and write each function to the file
    for i, func in enumerate(functions):
        name = variable_names[i] if variable_names else f"Variable_{i}"
        func.rename(name, "solution")
        file.write(func, time)

    file.close()



T = 0

variable_names = [  "Phi", "U", "Theta" ]  # Adjust as needed


write_simulation_data( solution_vector_0, T, file , variable_names=variable_names )


#############################  END  ###############################


############ Initialize for Adaptive Mesh #########################




for it in tqdm(range(0, 1000000000)):

    T = T + dt

    solver.solve()

    solution_vector_0.vector()[:] = solution_vector.vector()  # update the solution


    # Refining mesh
    if it == 20 or it % 30 == 25 :

        start = time.perf_counter()
        mesh_new, mesh_info = refine_mesh(coarse_mesh, solution_vector_0, spaces, max_level, comm )
        # Update the solver and solution on the new mesh
        solver, solution_vector, solution_vector_0, spaces = update_solver_on_new_mesh(mesh_new, parameters, solution_vector, solution_vector_0)
        end = time.perf_counter()
        Time_Lentgh_of_refinment = end - start


    if it % 500 == 0 :

        write_simulation_data( solution_vector_0,  T , file , variable_names )


    # print information of simulation
    if it % 1000 == 500 :
        n_cells = mesh_info["n_cells"]
        dx_min = mesh_info["dx_min"]
        dx_max = mesh_info["dx_max"]
        


        simulation_status_message = (
        f"Simulation Status:\n"
        f"├─ Iteration: {it}\n"
        f"├─ Simulation Time: {T:.2f} scaled unit \n"
        f"└─ Mesh Refinement Details:\n"
        f"   ├─ Number of cells: {n_cells}\n"
        f"   ├─ Minimum ΔX: {dx_min:.4f}\n"
        f"   ├─ Maximum ΔX: {dx_max:.4f}\n"
        f"   └─ Refinement Computation Time: {Time_Lentgh_of_refinment:.2f} seconds\n"
        )

        print(simulation_status_message, flush=True)

        



#############################  END  ###############################
