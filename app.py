import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(page_title="Gauss-Jordan Calculator", layout="wide")

# Custom CSS for matrix styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .center-title {
        text-align: center;
        font-size: 70px !important;
        margin-bottom: 15px;
    }
    
    /* Parameters section */
    .parameters-container {
        max-width: 600px !important;
        margin: 10px auto;
        display: flex;
        flex-direction: column;
        gap: 20px;
        padding: 10px;
        background-color: transparent;
        align-items: center;
    }
    
    /* Parameter labels */
    .param-label {
        font-family: 'Roboto', sans-serif;
        font-size: 15px;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 10px;
        font-weight: 500;
        width: 100%;
    }
    
    /* All parameter controls (sliders and select) */
    .parameters-container > div {
        width: 500px !important;
        margin: 0 auto !important;
    }
    
    /* Slider tracks */
    .parameters-container [data-testid="stSlider"] > div > div > div {
        width: 500px !important;
    }
    
    /* Select slider */
    .parameters-container div[data-testid="stSelectSlider"] {
        width: 500px !important;
    }
    
    /* Hide default labels */
    .parameters-container [data-testid="stSlider"] > div:first-child,
    .parameters-container div[data-testid="stSelectSlider"] label {
        display: none !important;
    }
    
    /* Matrix input styling */
    .matrix-input {
        position: relative;
        max-width: 600px;
        margin: 40px 0px;
        padding: 10px;
        background-color: transparent;
    }
    
    .matrix-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        margin: 4px 0;
    }
    
    .matrix-container {
        position: relative;
        margin: 0 auto;
        padding: 10px 0;
    }
    
    /* Matrix sliders */
    .matrix-row .stSlider {
        width: 80px !important;
        min-width: 80px !important;
        margin: 0 !important;
    }
    
    .matrix-row [data-testid="stThumbValue"] {
        display: none !important;
    }
    
    /* Show value above matrix slider */
    .matrix-row .stSlider::before {
        content: attr(data-value);
        position: absolute;
        top: -15px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 12px;
        font-family: 'Roboto', sans-serif;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .matrix-row .stSlider [data-testid="stTickBar"] {
        display: none !important;
    }
    
    /* Matrix separator */
    .matrix-separator {
        width: 4px;
    }
    
    /* Matrix element container */
    .matrix-element {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 80px !important;
        min-width: 80px !important;
        max-width: 80px !important;
    }
    
    /* Matrix element sliders */
    .matrix-element .stSlider {
        width: 80px !important;
        min-width: 80px !important;
        margin: 0 !important;
    }
    
    /* Matrix element number inputs */
    .matrix-element .stNumberInput {
        width: 80px !important;
    }
    
    .matrix-element .stNumberInput input {
        width: 80px !important;
        text-align: center;
        padding: 0 4px !important;
        background: transparent !important;
        border: none !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-family: 'Roboto', sans-serif;
        font-size: 12px !important;
    }
    
    /* Hide slider thumb value */
    .matrix-element [data-testid="stThumbValue"] {
        display: none !important;
    }
    
    /* Matrix row styling */
    .matrix-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        margin: 15px 0;
    }
    
    /* Matrix container */
    .matrix-container {
        margin: 0 auto;
        padding: 10px 0;
    }
    
    /* Matrix labels */
    .matrix-labels {
        display: flex;
        justify-content: center;
        gap: 350px;
        margin-bottom: 10px;
        font-family: 'Roboto', sans-serif;
        font-size: 15px;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .matrix-a-label, .vector-b-label {
        font-weight: 500;
    }
    
    /* Variable labels */
    
</style>
""", unsafe_allow_html=True)

# Application title and description
st.markdown('<h1 class="center-title">Gauss-Jordan Calculator</h1>', unsafe_allow_html=True)
# Input parameters section - all in one column
st.markdown('<div class="parameters-container">', unsafe_allow_html=True)

# Matrix size with custom label
st.markdown('<div class="param-label">Matrix Size</div>', unsafe_allow_html=True)
size = st.select_slider("", options=[2, 3, 4], value=2, label_visibility="collapsed")

# Tolerance with custom label - using logarithmic scale
st.markdown('<div class="param-label">Tolerance (ε)</div>', unsafe_allow_html=True)
tolerance_exp = st.slider(
    "",
    min_value=-15,
    max_value=-1,
    value=-10,
    step=1,
    label_visibility="collapsed"
)
tolerance = 10.0 ** tolerance_exp

# Maximum iterations with custom label
st.markdown('<div class="param-label">Maximum Iterations</div>', unsafe_allow_html=True)
max_iterations = st.slider(
    "",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# Matrix input interface
st.markdown('<div class="matrix-input">', unsafe_allow_html=True)
st.markdown('<div class="matrix-container">', unsafe_allow_html=True)

# Initialize session state for matrix values
if 'matrix_values' not in st.session_state:
    st.session_state.matrix_values = {}

# Create the augmented matrix input [A|b]
matrix_input = []
for i in range(size):
    st.markdown(f'<div class="matrix-row">', unsafe_allow_html=True)
    row = []
    
    # Create columns with specific widths
    column_widths = [1] * size + [0.2] + [1]
    cols = st.columns(column_widths)
    
    # Input fields for matrix A
    for j in range(size):
        with cols[j]:
            st.markdown('<div class="matrix-element">', unsafe_allow_html=True)
                        # Variable labels removed
            
            # Initialize value if not exists
            key = f"{i}{j}"
            if key not in st.session_state:
                st.session_state[key] = 1.0 if i==j else 0.0
            
            # Number input for precise value
            number_val = st.number_input(
                "",
                min_value=-10.0,
                max_value=10.0,
                value=st.session_state[key],
                step=0.01,
                format="%.2f",
                key=f"number_{key}",
                label_visibility="collapsed"
            )
            
            if number_val != st.session_state[key]:
                st.session_state[key] = number_val
            
            # Slider for value
            slider_val = st.slider(
                "",
                min_value=-10.0,
                max_value=10.0,
                value=st.session_state[key],
                step=0.01,
                key=f"slider_{key}",
                label_visibility="collapsed"
            )
            
            if slider_val != st.session_state[key]:
                st.session_state[key] = slider_val
            
            row.append(st.session_state[key])
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add separator between A and b
    with cols[size]:
        st.markdown('<div class="matrix-separator"></div>', unsafe_allow_html=True)
    
    # Input field for vector b
    with cols[size + 1]:
        st.markdown('<div class="matrix-element">', unsafe_allow_html=True)
        # Removed b label
        
        # Initialize value if not exists
        key = f"b{i}"
        if key not in st.session_state:
            st.session_state[key] = 0.0
        
        # Number input for precise value
        number_val = st.number_input(
            "",
            min_value=-10.0,
            max_value=10.0,
            value=st.session_state[key],
            step=0.01,
            format="%.2f",
            key=f"number_{key}",
            label_visibility="collapsed"
        )
        
        if number_val != st.session_state[key]:
            st.session_state[key] = number_val
        
        # Slider for value
        slider_val = st.slider(
            "",
            min_value=-10.0,
            max_value=10.0,
            value=st.session_state[key],
            step=0.01,
            key=f"slider_{key}",
            label_visibility="collapsed"
        )
        
        if slider_val != st.session_state[key]:
            st.session_state[key] = slider_val
        
        row.append(st.session_state[key])
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    matrix_input.append(row)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

def format_number(num):
    """
    Format numbers with proper mathematical notation.
    
    Args:
        num (float): Number to format
        
    Returns:
        str: Formatted number string
    """
    if abs(num) < 1e-10:
        return "0"
    elif abs(abs(num) - 1) < 1e-10:
        return "-1" if num < 0 else "1"
    elif abs(num - round(num)) < 1e-10:
        return str(int(round(num)))
    else:
        return f"{num:.6f}".rstrip('0').rstrip('.')

def format_solution(var_idx, value):
    """
    Format solution with mathematical notation.
    
    Args:
        var_idx (int): Variable index (0-based)
        value (float): Variable value
        
    Returns:
        str: HTML formatted solution string
    """
    return f"{chr(97 + var_idx)}1 = {format_number(value)}"

def gauss_jordan_elimination(matrix, tolerance, max_iterations):
    """
    Solve system of linear equations using Gauss-Jordan elimination.
    
    Args:
        matrix (numpy.ndarray): Augmented matrix [A|b]
        tolerance (float): Error tolerance for convergence
        max_iterations (int): Maximum number of iterations
        
    Returns:
        tuple: (result_matrix, message, iteration_history, error_history)
            - result_matrix: Final augmented matrix or None if no solution
            - message: Status message
            - iteration_history: List of matrices for each iteration
            - error_history: List of errors for each iteration
    """
    try:
        matrix = matrix.astype(float)
        n = len(matrix)
        iteration_history = []
        error_history = []
        
        for iteration in range(max_iterations):
            old_matrix = matrix.copy()
            
            # Perform Gauss-Jordan elimination for each pivot
            for i in range(n):
                pivot = matrix[i][i]
                
                # Find maximum pivot if current pivot is too small
                if abs(pivot) < tolerance:
                    max_row = i
                    max_val = abs(pivot)
                    for j in range(i + 1, n):
                        if abs(matrix[j][i]) > max_val:
                            max_val = abs(matrix[j][i])
                            max_row = j
                    
                    if max_val < tolerance:
                        return None, "No unique solution exists", None, None
                    
                    # Swap rows to get better pivot
                    matrix[i], matrix[max_row] = matrix[max_row].copy(), matrix[i].copy()
                    pivot = matrix[i][i]
                
                # Normalize the pivot row
                matrix[i] = matrix[i] / pivot
                
                # Eliminate the pivot element from other rows
                for j in range(n):
                    if i != j:
                        factor = matrix[j][i]
                        matrix[j] = matrix[j] - factor * matrix[i]
            
            # Calculate error and store history
            error = np.max(np.abs(matrix - old_matrix))
            iteration_history.append(matrix.copy())
            error_history.append(error)
            
            # Check for convergence
            if error < tolerance:
                break
        
        # Verify solution
        left_side = matrix[:, :n]
        if not np.allclose(left_side, np.eye(n), rtol=tolerance):
            return None, "System is inconsistent", None, None
        
        return matrix, "Solution found", iteration_history, error_history
    except Exception as e:
        return None, f"Error: {str(e)}", None, None

# Solve button and results display
if st.button("Solve System", type="primary"):
    matrix = np.array(matrix_input)
    result, message, iteration_history, error_history = gauss_jordan_elimination(matrix, tolerance, max_iterations)
    
    st.write("---")
    st.markdown(f"<p class='math-symbol'>{message}</p>", unsafe_allow_html=True)
    
    if result is not None:
        # Solution found message (large)
        st.markdown("<div style='text-align: center; font-size: 70px; color: rgba(255, 255, 255, 0.9); margin: 40px 0; font-family: Roboto, sans-serif; font-weight: 300;'>Solution found</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='text-align: center; font-family: Roboto, sans-serif; max-width: 800px; margin: 0 auto;'>", unsafe_allow_html=True)
        
        # Original Matrix Section
        st.markdown("<div style='margin: 40px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 24px; color: rgba(255, 255, 255, 0.9); margin-bottom: 25px; font-weight: 500; font-family: Roboto, sans-serif;'>Original Augmented Matrix</div>", unsafe_allow_html=True)
        
        # Format original matrix
        original_matrix_html = "<div style='font-family: Roboto Mono, monospace; font-size: 20px; color: rgba(255, 255, 255, 0.9); margin: 20px 0; line-height: 1.5;'>"
        original_matrix_html += "⎡"
        for i in range(size):
            if i > 0:
                original_matrix_html += "⎢" if i < size - 1 else "⎣"
            for j in range(size):
                val = matrix_input[i][j]
                original_matrix_html += f" {val:6.2f} "
                if j == size - 1:
                    original_matrix_html += "│"
            original_matrix_html += f" {matrix_input[i][-1]:6.2f} "
            if i == 0:
                original_matrix_html += "⎤<br>"
            elif i < size - 1:
                original_matrix_html += "⎥<br>"
            else:
                original_matrix_html += "⎦"
        original_matrix_html += "</div>"
        st.markdown(original_matrix_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # RREF Section
        st.markdown("<div style='margin: 40px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 24px; color: rgba(255, 255, 255, 0.9); margin-bottom: 25px; font-weight: 500; font-family: Roboto, sans-serif;'>Reduced Row Echelon Form (RREF)</div>", unsafe_allow_html=True)
        
        # Format RREF matrix
        rref_matrix_html = "<div style='font-family: Roboto Mono, monospace; font-size: 20px; color: rgba(255, 255, 255, 0.9); margin: 20px 0; line-height: 1.5;'>"
        rref_matrix_html += "⎡"
        for i in range(size):
            if i > 0:
                rref_matrix_html += "⎢" if i < size - 1 else "⎣"
            for j in range(size):
                val = result[i,j]
                rref_matrix_html += f" {val:6.2f} "
                if j == size - 1:
                    rref_matrix_html += "│"
            rref_matrix_html += f" {result[i,-1]:6.2f} "
            if i == 0:
                rref_matrix_html += "⎤<br>"
            elif i < size - 1:
                rref_matrix_html += "⎥<br>"
            else:
                rref_matrix_html += "⎦"
        rref_matrix_html += "</div>"
        st.markdown(rref_matrix_html, unsafe_allow_html=True)
        
        # Solution Type and Values
        is_unique = all(abs(result[i,i] - 1.0) < 1e-10 for i in range(size))
        has_solution = not any(all(abs(result[i,j]) < 1e-10 for j in range(size)) and abs(result[i,-1]) > 1e-10 for i in range(size))
        
        if has_solution:
            if is_unique:
                solution_type = "Unique Solution"
                st.markdown("<div style='font-size: 24px; color: rgba(255, 255, 255, 0.9); margin: 30px 0 20px; font-weight: 500; font-family: Roboto, sans-serif;'>" + solution_type + "</div>", unsafe_allow_html=True)
                
                solution_html = "<div style='font-size: 20px; color: rgba(255, 255, 255, 0.9); margin: 20px 0; line-height: 1.5; font-family: Roboto, sans-serif;'>"
                for i in range(size):
                    solution_html += f"x<sub>{i+1}</sub> = {result[i,-1]:.4f}<br>"
                solution_html += "</div>"
                st.markdown(solution_html, unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 24px; color: rgba(255, 255, 255, 0.9); margin: 30px 0 20px; font-weight: 500; font-family: Roboto, sans-serif;'>∞ Infinite Solutions</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size: 24px; color: rgba(255, 255, 255, 0.9); margin: 30px 0 20px; font-weight: 500; font-family: Roboto, sans-serif;'>✗ No Solution</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)