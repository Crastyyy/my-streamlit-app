import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set page config and style
st.set_page_config(page_title="Gauss-Jordan Calculator", layout="wide")

# Custom CSS for white background and styling
st.markdown("""
    <style>
        .stApp {
            background-color: white;
        }
        .stButton>button {
            width: 100%;
        }
        .plot-container>div {
            background-color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Gauss-Jordan Elimination Calculator")

def gauss_jordan_elimination(matrix, vector):
    try:
        # Combine matrix with vector
        augmented = np.hstack((matrix, vector.reshape(-1, 1)))
        n = len(matrix)
        
        # Perform Gauss-Jordan elimination
        for i in range(n):
            # Make the diagonal element 1
            pivot = augmented[i][i]
            if pivot == 0:
                raise np.linalg.LinAlgError("Zero pivot encountered")
            augmented[i] = augmented[i] / pivot
            
            # Make other elements in column i equal to 0
            for j in range(n):
                if i != j:
                    augmented[j] = augmented[j] - augmented[i] * augmented[j][i]
        
        return augmented[:, -1], True
    except Exception as e:
        return None, False

def plot_system(matrix, vector, solution=None):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Create points for plotting
    x = np.linspace(-10, 10, 1000)
    
    # Plot each equation
    colors = ['#2ecc71', '#e74c3c', '#3498db']  # Green, Red, Blue
    for i in range(len(matrix)):
        if matrix[i][1] != 0:  # Avoid division by zero
            y = (-matrix[i][0] * x - vector[i]) / matrix[i][1]
            ax.plot(x, y, label=f'Equation {i+1}', color=colors[i], linewidth=2)
    
    # Plot solution point if available
    if solution is not None and len(matrix) >= 2:
        ax.plot(solution[0], solution[1], 'ko', label='Solution', markersize=10)
        ax.plot(solution[0], solution[1], 'wo', markersize=6)
    
    # Styling
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title('System of Linear Equations Visualization', pad=20, fontsize=14)
    
    # Set reasonable axis limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    # Make plot background white
    ax.set_facecolor('white')
    
    return fig

# Create two columns for input and visualization
left_col, right_col = st.columns([1, 1])

with left_col:
    # User input for matrix size
    size = st.number_input("Enter the size of the matrix (2 or 3):", min_value=2, max_value=3, value=2)

    # Create input fields for matrix
    st.subheader("Enter Matrix Coefficients:")
    matrix = []
    for i in range(size):
        row = []
        cols = st.columns(size)
        for j in range(size):
            row.append(cols[j].number_input(f"Matrix[{i}][{j}]", value=1.0 if i == j else 0.0, 
                                          key=f"matrix_{i}_{j}", step=0.1))
        matrix.append(row)

    # Create input fields for vector
    st.subheader("Enter Constants:")
    vector = []
    cols = st.columns(size)
    for i in range(size):
        vector.append(cols[i].number_input(f"b[{i}]", value=1.0, key=f"vector_{i}", step=0.1))

    matrix_np = np.array(matrix, dtype=float)
    vector_np = np.array(vector, dtype=float)

with right_col:
    st.subheader("System Visualization and Solution")
    
    # Try to solve and update visualization in real-time
    solution, success = gauss_jordan_elimination(matrix_np.copy(), vector_np.copy())
    
    if size == 2:  # Only show graph for 2x2 systems
        fig = plot_system(matrix_np, vector_np, solution if success else None)
        st.pyplot(fig)
        
        if success:
            st.success("Solution:")
            for i, val in enumerate(solution):
                st.write(f"x{i+1} = {val:.4f}")
        else:
            st.warning("The system may not have a unique solution. Try adjusting the coefficients.")
    else:
        st.info("3D visualization is not supported. Solution will be shown below.")
        if success:
            st.success("Solution:")
            for i, val in enumerate(solution):
                st.write(f"x{i+1} = {val:.4f}")
        else:
            st.warning("The system may not have a unique solution. Try adjusting the coefficients.")