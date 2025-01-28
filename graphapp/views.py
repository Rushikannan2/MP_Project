import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

def home(request):
    return render(request, 'graphapp/home.html')

def solve(request):
    if request.method == 'POST':
        try:
            # Retrieve inputs from the form
            c_input = request.POST.get('objective_function')  # e.g., "50,18"
            constraints_input = request.POST.get('constraints')  # e.g., "2 1<=100;1 1<=80;1 0>=0;0 1>=0"

            # Parse objective function
            c = list(map(float, c_input.split(',')))

            # Parse constraints
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
                    coeffs = [-x for x in coeffs]  # Negate coefficients
                    rhs = -rhs  # Negate RHS
                else:
                    continue  # Ignore invalid constraints

                A.append(coeffs)
                b.append(rhs)

            # Solve and generate graph
            graph_url, result = call_solver(c, A, b)

            # Pass results to the template
            return render(request, 'graphapp/home.html', {
                'result': result,
                'graph_url': graph_url,
            })
        except Exception as e:
            return HttpResponse(f"Error: {e}")

    return HttpResponse("Invalid request")

def call_solver(c, A, b):
    """Wrapper to solve the linear program and capture the graph."""
    buf = io.BytesIO()
    try:
        optimal_vertex, optimal_value = solve_linear_program(c, A, b, buf)
        buf.seek(0)
        graph_url = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        return graph_url, f"Optimal solution: {optimal_vertex}, Z = {optimal_value}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def solve_linear_program(c, A, b, buf):
    """Solve the linear programming problem and plot."""
    bounds = [0, max(max(b), 10)]  # Adjust bounds dynamically
    constraints = list(zip(A, b))

    # Solve using vertices of the feasible region
    vertices = []
    num_constraints = len(A)
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            A_ = np.array([A[i], A[j]])
            b_ = np.array([b[i], b[j]])
            try:
                vertex = np.linalg.solve(A_, b_)
                if np.all(np.dot(A, vertex) <= b) and np.all(vertex >= 0):  # Feasibility check
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue

    # Filter unique vertices
    feasible_vertices = np.unique(vertices, axis=0)

    # Evaluate the objective function at each vertex
    if len(feasible_vertices) > 0:
        z_values = [np.dot(c, v) for v in feasible_vertices]
        optimal_value = max(z_values)
        optimal_vertex = feasible_vertices[np.argmax(z_values)]

        # Plot constraints and feasible region
        plot_constraints(constraints, bounds, feasible_region=feasible_vertices, optimal_vertex=optimal_vertex, buf=buf)
        return optimal_vertex, optimal_value
    else:
        raise ValueError("No feasible region found.")

def plot_constraints(constraints, bounds, feasible_region=None, optimal_vertex=None, buf=None):
    """Plots constraints, feasible region, and optimal solution."""
    x = np.linspace(bounds[0], bounds[1], 400)
    plt.figure(figsize=(10, 8))

    # Plot constraints
    for coeff, b in constraints:
        if coeff[1] != 0:  # Line with slope
            y = (b - coeff[0] * x) / coeff[1]
            plt.plot(x, y, label=f"{coeff[0]}x1 + {coeff[1]}x2 ≤ {b}")
        else:  # Vertical line
            x_val = b / coeff[0]
            plt.axvline(x_val, color='r', linestyle='--', label=f"x1 = {x_val}")

    # Plot feasible region
    if feasible_region is not None and len(feasible_region) > 0:
        hull = ConvexHull(feasible_region)
        polygon = Polygon(feasible_region[hull.vertices], closed=True, color='lightgreen', alpha=0.5, label='Feasible Region')
        plt.gca().add_patch(polygon)

    # Mark feasible region vertices
    if feasible_region is not None:
        for point in feasible_region:
            plt.plot(point[0], point[1], 'bo')  # Mark corners

    # Highlight the optimal solution
    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro', label='Optimal Solution')

    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linear Programming: Graphical Method")
    plt.legend()
    plt.grid()

    # Save plot to buffer
    if buf:
        plt.savefig(buf, format='png')
        plt.close()
