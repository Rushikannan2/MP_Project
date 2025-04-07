import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import re
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, LpInteger, lpSum

# Graphical Method Views
def home(request):
    return render(request, 'graphapp/home.html')

def graphical_view(request):
    return render(request, 'graphapp/graphical.html')

def solve(request):
    if request.method == 'POST':
        try:
            c_input = request.POST.get('objective_function')
            constraints_input = request.POST.get('constraints')

            # Here we assume objective coefficients are comma separated.
            c = list(map(float, c_input.split(',')))
            A, b = [], []

            for constraint in constraints_input.split(';'):
                constraint = constraint.strip()
                if not constraint:
                    continue
                
                # Handle different constraint types
                match = re.match(r"^(.*?)(<=|>=)(.*?)$", constraint)
                if not match:
                    continue  # Skip invalid constraints
                
                lhs_str, op, rhs_str = match.groups()
                coeffs = list(map(float, lhs_str.strip().split()))
                rhs = float(rhs_str.strip())
                
                if op == '>=':
                    coeffs = [-x for x in coeffs]
                    rhs = -rhs
                
                A.append(coeffs)
                b.append(rhs)

            graph_url, result = call_solver(c, A, b)
            return render(request, 'graphapp/graphical.html', {
                'result': result,
                'graph_url': graph_url,
            })
        except Exception as e:
            return render(request, 'graphapp/graphical.html', {
                'error': f"Error: {str(e)}"
            })
    return redirect('graphapp:graphical')

# Transportation Problem Views
def transportation_view(request):
    return render(request, 'graphapp/transportation.html')

def solve_transportation(request):
    if request.method == 'POST':
        try:
            # Get the expression and size from the form
            expression = request.POST.get('expression', '').strip()
            size = int(request.POST.get('size', 3))
            
            if not expression:
                raise ValueError("Please enter the transportation problem data")
            
            # Split and convert input values to floats
            numbers = []
            for num in expression.split():
                if num.strip():
                    try:
                        numbers.append(float(num.strip()))
                    except ValueError:
                        raise ValueError(f"Invalid number: {num}")
            
            # Validate input size
            expected_values = size * size + size * 2
            if len(numbers) != expected_values:
                raise ValueError(f"Expected {expected_values} values for a {size}x{size} problem, but got {len(numbers)}")
            
            # Extract the components
            matrix_elements = size * size
            costs = np.array(numbers[:matrix_elements]).reshape(size, size)
            supply = np.array(numbers[matrix_elements:matrix_elements + size])
            demand = np.array(numbers[matrix_elements + size:])
            
            # Validate supply and demand balance
            total_supply = sum(supply)
            total_demand = sum(demand)
            if abs(total_supply - total_demand) > 1e-10:
                raise ValueError(f"Total supply ({total_supply}) must equal total demand ({total_demand})")
            
            # Solve the transportation problem
            result = solve_transportation_problem(costs, supply, demand)
            
            if result["solution"] is not None:
                solution_matrix = result["solution"]
                total_cost = 0
                solution_steps = []
                
                # Format the input problem
                solution_steps.append("Input Problem:")
                solution_steps.append("\nCost Matrix:")
                solution_steps.append(str(np.array2string(costs, precision=2, separator=' ')))
                
                solution_steps.append("\nSupply Values:")
                solution_steps.append(str(np.array2string(supply, precision=2, separator=' ')))
                
                solution_steps.append("\nDemand Values:")
                solution_steps.append(str(np.array2string(demand, precision=2, separator=' ')))
                
                # Format the solution
                solution_steps.append("\nOptimal Transportation Plan:")
                solution_steps.append(str(np.array2string(solution_matrix, precision=2, separator=' ')))
                
                # Calculate and show route details
                solution_steps.append("\nDetailed Route Information:")
                for i in range(size):
                    for j in range(size):
                        if solution_matrix[i][j] > 0:
                            route_cost = solution_matrix[i][j] * costs[i][j]
                            total_cost += route_cost
                            solution_steps.append(
                                f"Route: Source {i+1} → Destination {j+1}"
                                f"\n  • Units: {solution_matrix[i][j]:.2f}"
                                f"\n  • Cost per unit: ${costs[i][j]:.2f}"
                                f"\n  • Total route cost: ${route_cost:.2f}"
                            )
                
                # Create the solution dictionary
                solution = {
                    'steps': solution_steps,
                    'total_cost': total_cost,
                    'explanation': [
                        "Transportation Problem Solution:",
                        f"• Total Transportation Cost: ${total_cost:.2f}",
                        f"• Status: {result['status']}",
                        "• All supply and demand requirements are satisfied"
                    ]
                }
                
                return render(request, 'graphapp/transportation.html', {
                    'result': solution
                })
            else:
                return render(request, 'graphapp/transportation.html', {
                    'error': f"No feasible solution found. Status: {result['status']}"
                })

        except Exception as e:
            return render(request, 'graphapp/transportation.html', {
                'error': f"Error: {str(e)}"
            })
    
    # For GET requests, just show the empty form
    return render(request, 'graphapp/transportation.html')

# Simplex Method Views
def simplex_view(request):
    return render(request, 'graphapp/simplex.html')

def solve_simplex(request):
    if request.method == 'POST':
        try:
            # Parse objective function (expects comma-separated values)
            objective_type = request.POST.get('objective_type', 'max')
            c_input = request.POST['objective_function'].strip()
            c = list(map(float, c_input.split(',')))
            num_vars = len(c)

            # Initialize constraint matrices
            A_ub, b_ub = [], []
            A_eq, b_eq = [], []

            # Allow constraints to be separated by semicolon OR newline.
            raw_constraints = re.split(r';|\n', request.POST['constraints'])
            for constraint in raw_constraints:
                constraint = constraint.strip()
                if not constraint:
                    continue

                # Expect a format like "1 1 1 <= 10"
                match = re.match(r"^(.*?)(<=|>=|=)(.*?)$", constraint)
                if not match:
                    raise ValueError(f"Invalid constraint format: '{constraint}'")

                lhs_str, op, rhs_str = match.groups()
                
                # Parse coefficients (space-separated)
                coeffs = list(map(float, lhs_str.strip().split()))
                if len(coeffs) != num_vars:
                    raise ValueError(f"Constraint '{constraint}' has {len(coeffs)} coefficients, expected {num_vars}")

                # Parse RHS value
                rhs = float(rhs_str.strip())

                # Handle constraint types
                if op == '<=':
                    A_ub.append(coeffs)
                    b_ub.append(rhs)
                elif op == '>=':
                    A_ub.append([-x for x in coeffs])
                    b_ub.append(-rhs)
                elif op == '=':
                    A_eq.append(coeffs)
                    b_eq.append(rhs)

            # Convert maximization to minimization (if needed)
            if objective_type == 'max':
                c = [-x for x in c]

            # Solve using linprog
            result = linprog(
                c=c,
                A_ub=A_ub if A_ub else None,
                b_ub=b_ub if b_ub else None,
                A_eq=A_eq if A_eq else None,
                b_eq=b_eq if b_eq else None,
                bounds=(0, None),
                method='highs'
            )

            # Prepare solution details
            solution = {
                'optimal_value': None,
                'variables': None,
                'status': result.message,
                'iterations': result.nit
            }

            if result.success:
                solution['optimal_value'] = abs(result.fun) if objective_type == 'max' else result.fun
                solution['variables'] = [round(x, 4) for x in result.x] if result.x is not None else []
                solution['status'] = 'Optimal solution found'

            return render(request, 'graphapp/simplex.html', {
                'result': solution,
                'objective_type': objective_type,
                'num_vars': num_vars,
                'constraint_count': len(A_ub) + len(A_eq)
            })

        except Exception as e:
            return render(request, 'graphapp/simplex.html', {
                'error': f"Input error: {str(e)}",
                'preserve_input': True
            })
    return redirect('graphapp:simplex')

# Helper Functions
def plot_matrix(matrix, title=""):
    if not matrix:
        return None
        
    buf = io.BytesIO()
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    table = ax.table(
        cellText=matrix,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def solve_transportation_problem(cost_matrix, supply, demand):
    """
    Solves the transportation problem using linear programming.
    """
    try:
        # Convert inputs to numpy arrays and ensure they are 2D/1D as needed
        cost_matrix = np.asarray(cost_matrix, dtype=float)
        supply = np.asarray(supply, dtype=float).flatten()
        demand = np.asarray(demand, dtype=float).flatten()

        # Get dimensions from the cost matrix shape
        m, n = cost_matrix.shape if isinstance(cost_matrix, np.ndarray) else (len(supply), len(demand))
        
        if m * n == 0:
            raise ValueError("Invalid dimensions: Cost matrix cannot be empty")

        # Flatten cost matrix for linprog
        c = cost_matrix.flatten()

        # Create equality constraints matrix
        A_eq = []
        b_eq = []

        # Supply constraints
        for i in range(m):
            row = np.zeros(m * n)
            row[i * n:(i + 1) * n] = 1
            A_eq.append(row)
            b_eq.append(supply[i])

        # Demand constraints
        for j in range(n):
            row = np.zeros(m * n)
            row[j::n] = 1
            A_eq.append(row)
            b_eq.append(demand[j])

        # Convert to numpy arrays
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        # Solve using linprog
        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            method='highs',
            bounds=(0, None)
        )

        if result.success:
            # Ensure the solution is properly reshaped
            solution_matrix = result.x.reshape((m, n))
            return {
                "solution": solution_matrix,
                "total_cost": result.fun,
                "status": "Optimal solution found"
            }
        else:
            return {
                "solution": None,
                "total_cost": None,
                "status": result.message
            }
    except Exception as e:
        return {
            "solution": None,
            "total_cost": None,
            "status": str(e)
        }

# Graphical Method Helpers
def call_solver(c, A, b):
    buf = io.BytesIO()
    try:
        optimal_vertex, optimal_value = solve_linear_program(c, A, b, buf)
        buf.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        return graph_url, f"Optimal solution: {np.round(optimal_vertex, 4)}, Z = {round(optimal_value, 4)}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def solve_linear_program(c, A, b, buf):
    bounds = [0, max(max(b), 10)]
    constraints = list(zip(A, b))
    vertices = []
    num_constraints = len(A)
    
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            A_ = np.array([A[i], A[j]])
            b_ = np.array([b[i], b[j]])
            try:
                vertex = np.linalg.solve(A_, b_)
                if np.all(np.dot(A, vertex) <= b) and np.all(vertex >= 0):
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue

    feasible_vertices = np.unique(vertices, axis=0)

    if len(feasible_vertices) > 0:
        z_values = [np.dot(c, v) for v in feasible_vertices]
        optimal_value = max(z_values)
        optimal_vertex = feasible_vertices[np.argmax(z_values)]
        plot_constraints(constraints, bounds, feasible_vertices, optimal_vertex, buf)
        return optimal_vertex, optimal_value
    else:
        raise ValueError("No feasible region found.")

def plot_constraints(constraints, bounds, feasible_region=None, optimal_vertex=None, buf=None):
    x = np.linspace(bounds[0], bounds[1], 400)
    plt.figure(figsize=(10, 8))

    for coeff, b_val in constraints:
        if coeff[1] != 0:
            y = (b_val - coeff[0] * x) / coeff[1]
            plt.plot(x, y, label=f"{coeff[0]}x1 + {coeff[1]}x2 ≤ {b_val}")
        else:
            x_val = b_val / coeff[0]
            plt.axvline(x_val, color='r', linestyle='--', label=f"x1 = {x_val}")

    if feasible_region is not None and len(feasible_region) > 0:
        hull = ConvexHull(feasible_region)
        polygon = Polygon(feasible_region[hull.vertices], closed=True, 
                         color='lightgreen', alpha=0.5, label='Feasible Region')
        plt.gca().add_patch(polygon)

    if feasible_region is not None:
        for point in feasible_region:
            plt.plot(point[0], point[1], 'bo')

    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro', label='Optimal Solution')

    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linear Programming: Graphical Method")
    plt.legend()
    plt.grid()

    if buf:
        plt.savefig(buf, format='png')
        plt.close()

def applications(request):
    return render(request, 'graphapp/applications.html')

def integer_programming(request):
    result = None
    if request.method == 'POST':
        try:
            # Get number of variables and constraints
            num_vars = int(request.POST.get('num_vars'))
            num_constraints = int(request.POST.get('num_constraints'))
            opt_type = request.POST.get('opt_type')

            # Create decision variables
            variables = {}
            for i in range(num_vars):
                var_name = f"x{i+1}"
                variables[var_name] = LpVariable(name=var_name, lowBound=0, cat=LpInteger)

            # Create the model
            if opt_type == "max":
                model = LpProblem(name="Integer_Programming", sense=LpMaximize)
            else:
                model = LpProblem(name="Integer_Programming", sense=LpMinimize)

            # Add objective function
            obj_coeffs = [float(request.POST.get(f'obj_coeff_{i}')) for i in range(num_vars)]
            model += lpSum(obj_coeffs[i] * variables[f"x{i+1}"] for i in range(num_vars)), "Objective_Function"

            # Add constraints
            for j in range(num_constraints):
                constraint_coeffs = [float(request.POST.get(f'constraint_{j}_coeff_{i}')) for i in range(num_vars)]
                operator = request.POST.get(f'constraint_{j}_operator')
                rhs = float(request.POST.get(f'constraint_{j}_rhs'))

                if operator == "<=":
                    model += lpSum(constraint_coeffs[i] * variables[f"x{i+1}"] for i in range(num_vars)) <= rhs, f"Constraint_{j+1}"
                elif operator == ">=":
                    model += lpSum(constraint_coeffs[i] * variables[f"x{i+1}"] for i in range(num_vars)) >= rhs, f"Constraint_{j+1}"
                else:
                    model += lpSum(constraint_coeffs[i] * variables[f"x{i+1}"] for i in range(num_vars)) == rhs, f"Constraint_{j+1}"

            # Solve the problem
            model.solve()

            # Prepare results
            result = {
                'variables': {var.name: var.varValue for var in variables.values()},
                'objective_value': model.objective.value()
            }

        except Exception as e:
            messages.error(request, f"Error solving the problem: {str(e)}")

    return render(request, 'graphapp/IntegerProgramming.html', {'result': result})
