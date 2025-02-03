import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from django.shortcuts import render, redirect
from django.http import HttpResponse
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from matplotlib import gridspec

# Existing Graphical Method Views
def home(request):
    return render(request, 'graphapp/home.html')

def solve(request):
    if request.method == 'POST':
        try:
            c_input = request.POST.get('objective_function')
            constraints_input = request.POST.get('constraints')

            c = list(map(float, c_input.split(',')))
            A, b = [], []

            for constraint in constraints_input.split(';'):
                if '<=' in constraint:
                    parts = constraint.split('<=')
                    coeffs = list(map(float, parts[0].split()))
                    rhs = float(parts[1])
                elif '>=' in constraint:
                    parts = constraint.split('>=')
                    coeffs = list(map(float, parts[0].split()))
                    rhs = float(parts[1])
                    coeffs = [-x for x in coeffs]
                    rhs = -rhs
                else:
                    continue

                A.append(coeffs)
                b.append(rhs)

            graph_url, result = call_solver(c, A, b)
            return render(request, 'graphapp/home.html', {
                'result': result,
                'graph_url': graph_url,
            })
        except Exception as e:
            return HttpResponse(f"Error: {e}")
    return HttpResponse("Invalid request")

# Transportation Problem Views
def transportation_view(request):
    return render(request, 'graphapp/transportation.html')

def solve_transportation(request):
    if request.method == 'POST':
        try:
            cost_matrix = [
                list(map(float, row.split()))
                for row in request.POST['cost_matrix'].split(';')
            ]
            supply = list(map(float, request.POST['supply'].split()))
            demand = list(map(float, request.POST['demand'].split()))

            result = solve_transportation_problem(cost_matrix, supply, demand)
            cost_matrix_img = plot_matrix(cost_matrix, "Cost Matrix")
            solution_matrix_img = plot_matrix(
                result['solution'].tolist() if result['solution'] is not None else [],
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
            # Parse input data
            objective_type = request.POST.get('objective_type', 'max')
            c_input = list(map(float, request.POST['objective_function'].split(',')))
            constraints = []
            
            # Parse constraints
            for constraint in request.POST['constraints'].split(';'):
                parts = constraint.strip().split()
                coeffs = list(map(float, parts[:-2]))
                inequality = parts[-2]
                rhs = float(parts[-1])
                
                # Convert >= constraints to <= form
                if inequality == '>=':
                    coeffs = [-x for x in coeffs]
                    rhs = -rhs
                    inequality = '<='
                
                constraints.append({
                    'coeffs': coeffs,
                    'rhs': rhs,
                    'inequality': inequality
                })

            # Set up problem for linprog
            A = [con['coeffs'] for con in constraints]
            b = [con['rhs'] for con in constraints]

            # Convert maximization to minimization
            if objective_type == 'max':
                c = [-x for x in c_input]
            else:
                c = c_input

            # Solve using simplex
            result = linprog(c=c, A_ub=A, b_ub=b, method='highs')

            # Process results
            if result.success:
                solution = {
                    'optimal_value': -result.fun if objective_type == 'max' else result.fun,
                    'variables': result.x.tolist(),
                    'status': 'Optimal solution found',
                    'iterations': result.nit
                }
            else:
                solution = {
                    'optimal_value': None,
                    'variables': None,
                    'status': result.message,
                    'iterations': 0
                }

            return render(request, 'graphapp/simplex.html', {
                'result': solution,
                'objective_type': objective_type,
                'num_vars': len(c_input)
            })

        except Exception as e:
            return render(request, 'graphapp/simplex.html', {
                'error': f"Error processing input: {str(e)}"
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

    # Build equality constraints
    A_eq = []
    for i in range(m):
        A_eq.append([1 if (i * n) <= j < ((i + 1) * n) else 0 for j in range(m * n)])
    for j in range(n):
        A_eq.append([1 if (k % n) == j else 0 for k in range(m * n)])

    b_eq = np.concatenate([supply, demand])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    if result.success:
        solution_matrix = result.x.reshape(m, n)
        return {
            "solution": solution_matrix,
            "total_cost": result.fun,
            "status": "Optimal solution found",
        }
    else:
        return {
            "solution": None,
            "total_cost": None,
            "status": result.message,
        }

# Graphical Method Helpers
def call_solver(c, A, b):
    buf = io.BytesIO()
    try:
        optimal_vertex, optimal_value = solve_linear_program(c, A, b, buf)
        buf.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        return graph_url, f"Optimal solution: {optimal_vertex}, Z = {optimal_value}"
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