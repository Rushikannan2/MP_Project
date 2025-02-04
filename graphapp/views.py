import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import re
from django.shortcuts import render, redirect
from django.http import HttpResponse
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

# Graphical Method Views
def home(request):
    return render(request, 'graphapp/home.html')

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
            return render(request, 'graphapp/home.html', {
                'result': result,
                'graph_url': graph_url,
            })
        except Exception as e:
            return render(request, 'graphapp/home.html', {
                'error': f"Error: {str(e)}"
            })
    return redirect('graphapp:home')

# Transportation Problem Views
def transportation_view(request):
    return render(request, 'graphapp/transportation.html')

def solve_transportation(request):
    if request.method == 'POST':
        try:
            # Parse the cost matrix, supply and demand from POST data.
            cost_matrix = [
                list(map(float, row.split()))
                for row in request.POST['cost_matrix'].split(';')
                if row.strip()
            ]
            supply = list(map(float, request.POST['supply'].split()))
            demand = list(map(float, request.POST['demand'].split()))

            result = solve_transportation_problem(cost_matrix, supply, demand)
            cost_matrix_img = plot_matrix(cost_matrix, "Cost Matrix")
            solution_matrix_img = plot_matrix(
                result['transport_plan'].tolist() if result['transport_plan'] is not None else [],
                "Solution Matrix"
            )

            return render(request, 'graphapp/transportation.html', {
                'result': result,
                'cost_matrix_url': cost_matrix_img,
                'solution_matrix_url': solution_matrix_img
            })

        except Exception as e:
            return render(request, 'graphapp/transportation.html', {
                'error': f"Error: {str(e)}"
            })
    return redirect('graphapp:transportation')

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
    cost_matrix = np.array(cost_matrix)
    supply = np.array(supply)
    demand = np.array(demand)

    m, n = cost_matrix.shape
    c = cost_matrix.flatten()

    # Build equality constraints for supply and demand.
    A_eq = []
    # For supply constraints: Each row's sum equals supply.
    for i in range(m):
        row = [1 if (i * n) <= j < ((i + 1) * n) else 0 for j in range(m * n)]
        A_eq.append(row)
    # For demand constraints: Each column's sum equals demand.
    for j in range(n):
        col = [1 if (k % n) == j else 0 for k in range(m * n)]
        A_eq.append(col)

    b_eq = np.concatenate([supply, demand])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    if result.success:
        solution_matrix = result.x.reshape(m, n).round(4)
        return {
            "transport_plan": solution_matrix,
            "optimal_cost": round(result.fun, 4),
            "status": "Optimal solution found",
        }
    else:
        return {
            "transport_plan": None,
            "optimal_cost": None,
            "status": result.message,
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
