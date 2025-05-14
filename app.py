import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sympy as sp

# Configure the page
st.set_page_config(page_title="Gauss-Jordan Calculator", layout="wide")

# Custom styling
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"], 
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stSidebar"] {
            background-color: white;
        }
        
        .step-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #dee2e6;
        }
        
        .matrix-display {
            font-family: 'Computer Modern', serif;
            font-size: 1.2em;
            margin: 20px 0;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .solution-header {
            background-color:rgb(0, 0, 0);
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: bold;
        }
        
        /* Target all Streamlit text elements */
        .st-emotion-cache-1y4p8pa,
        .st-emotion-cache-183lzff,
        .st-emotion-cache-1inwz65,
        .st-emotion-cache-10trblm,
        .st-emotion-cache-1gulkj5,
        .st-emotion-cache-1n76uvr {
            color: black !important;
        }
        
        /* Style buttons */
        .stButton > button {
            color: white !important;
            background-color: black !important;
            border: none !important;
            padding: 8px 16px !important;
            border-radius: 4px !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background-color: #333 !important;
            transform: translateY(-1px);
        }
    </style>
""", unsafe_allow_html=True)

def format_matrix(matrix: np.ndarray) -> str:
    """Convert matrix to LaTeX string."""
    matrix = np.round(matrix, decimals=4)
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    latex = "\\begin{bmatrix} "
    for i in range(rows):
        for j in range(cols):
            if j == cols - 1:
                latex += f"{matrix[i,j]:.2f}"
            else:
                latex += f"{matrix[i,j]:.2f} & "
        if i != rows - 1:
            latex += " \\\\ "
    latex += " \\end{bmatrix}"
    return latex

def gauss_jordan_elimination_steps(matrix: np.ndarray, vector: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """Perform Gauss-Jordan elimination with steps."""
    augmented = np.hstack((matrix, vector.reshape(-1, 1)))
    n = len(matrix)
    steps = []
    
    # Store initial matrix
    steps.append((augmented.copy(), "Initial augmented matrix"))
    
    try:
        for i in range(n):
            # Make the diagonal element 1
            pivot = augmented[i][i]
            if abs(pivot) < 1e-10:
                raise np.linalg.LinAlgError("Zero pivot encountered")
            
            if pivot != 1:
                augmented[i] = augmented[i] / pivot
                steps.append((augmented.copy(), f"R_{i+1} = R_{i+1} ÷ {pivot:.2f}"))
            
            # Make other elements in column i equal to 0
            for j in range(n):
                if i != j:
                    factor = augmented[j][i]
                    if abs(factor) > 1e-10:
                        augmented[j] = augmented[j] - factor * augmented[i]
                        steps.append((augmented.copy(), 
                            f"R_{j+1} = R_{j+1} - {factor:.2f}R_{i+1}"))
        
        return steps, True
    except Exception as e:
        return steps, False

st.title("Gauss-Jordan Elimination Calculator")
st.markdown("---")

# Create two columns for input
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### Matrix Input")
    size = st.selectbox("Matrix Size", [2, 3], index=0)
    
    st.markdown("#### Enter Matrix Coefficients")
    matrix = []
    for i in range(size):
        row = []
        cols = st.columns(size)
        for j in range(size):
            row.append(cols[j].number_input(
                f"a_{i+1}{j+1}",
                value=1.0 if i == j else 0.0,
                key=f"matrix_{i}_{j}",
                format="%.2f"
            ))
        matrix.append(row)
    
    st.markdown("#### Enter Constants")
    vector = []
    cols = st.columns(size)
    for i in range(size):
        vector.append(cols[i].number_input(
            f"b_{i+1}",
            value=1.0,
            key=f"vector_{i}",
            format="%.2f"
        ))

    solve_button = st.button("Solve System")

with right_col:
    if solve_button:
        st.markdown("### Solution Steps")
        matrix_np = np.array(matrix, dtype=float)
        vector_np = np.array(vector, dtype=float)
        
        steps, success = gauss_jordan_elimination_steps(matrix_np, vector_np)
        
        if success:
            for idx, (step_matrix, description) in enumerate(steps):
                with st.container():
                    st.markdown(f"""
                    <div class="step-container">
                        <div class="solution-header">Step {idx + 1}: {description}</div>
                        <div class="matrix-display">
                            ${format_matrix(step_matrix)}$
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display final solution
            solution = steps[-1][0][:, -1]
            st.markdown("### Final Solution")
            for i, val in enumerate(solution):
                st.markdown(f"""
                <div style="font-size: 1.2em; margin: 10px 0;">
                    x<sub>{i+1}</sub> = {val:.4f}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("The system may not have a unique solution. Please check your input matrix.")

st.markdown("---")
st.markdown("""
### How to Use:
1. Select the matrix size (2×2 or 3×3)
2. Enter the coefficients of your system of equations
3. Enter the constants (right-hand side values)
4. Click "Solve System" to see the step-by-step solution
""")