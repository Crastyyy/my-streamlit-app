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
        text-align: side;
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
        margin: 15px auto;
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
</style>
""", unsafe_allow_html=True)

# Application title and description
st.markdown('<h1 class="center-title">Gauss-Jordan Calculator</h1>', unsafe_allow_html=True)
st.markdown('This is a tool that automatically solves systems of linear equations of the form Ax = b using the Gauss-Jordan elimination method. Simply choose your matrix size, set tolerance and maximum iterations if needed, and the calculator will transform your matrix into reduced row echelon form (RREF) to find the solution quickly and accurately, with no manual steps required.', unsafe_allow_html=True)

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

# Create the augmented matrix input [A|b]
matrix_input = []
for i in range(size):
    st.markdown(f'<div class="matrix-row">', unsafe_allow_html=True)
    row = []
    
    # Create columns for matrix A and vector b
    cols = st.columns(size + 2)  # +2 for separator and b vector
    
    # Input fields for matrix A
    for j in range(size):
        with cols[j]:
            val = st.slider(
                "",
                min_value=-10.0,
                max_value=10.0,
                value=1.0 if i==j else 0.0,  # Initialize as identity matrix
                step=0.1,
                key=f"a{i}{j}",
                label_visibility="collapsed"
            )
            row.append(val)
    
    # Add separator between A and b
    with cols[size]:
        st.markdown('<div class="matrix-separator"></div>', unsafe_allow_html=True)
    
    # Input field for vector b
    with cols[size + 1]:
        val = st.slider(
            "",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            key=f"b{i}",
            label_visibility="collapsed"
        )
        row.append(val)
    
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
    return f"x<sub>{var_idx + 1}</sub> = {format_number(value)}"

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
        # Display solution and final matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Solution:")
            solution_html = "<div class='math-symbol'>"
            for i in range(size):
                solution_html += format_solution(i, result[i,-1]) + "<br>"
            solution_html += "</div>"
            st.markdown(solution_html, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Final Matrix:")
            st.write(result)
        
        # Display convergence plot
        if error_history:
            st.markdown("#### Convergence")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=error_history,
                mode='lines+markers',
                name='Error',
                line=dict(color='#1f77b4')
            ))
            fig.update_layout(
                title='Error vs Iteration',
                xaxis_title='Iteration (n)',
                yaxis_title='Error (ε)',
                yaxis_type='log',
                showlegend=False
            )
            st.plotly_chart(fig)
        
        # Display iteration details
        if iteration_history:
            st.markdown("#### Iteration Details")
            iterations_df = pd.DataFrame({
                'n': range(len(error_history)),
                'ε': [f"{e:.2e}" for e in error_history]
            })
            st.dataframe(iterations_df, hide_index=True)