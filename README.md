# Gauss-Jordan Calculator

The Gauss-Jordan method is a variation of Gauss elimination. The major difference is that when an unknown is eliminated in the Gauss-Jordan method, it is eliminated from all other equations rather than just the subsequent ones. In addition, all rows are normalized by dividing them by their pivot elements. Thus, the elimination step results in an identity matrix rather than a triangular matrix.

This modern web application that solves systems of linear equations using the Gauss-Jordan elimination method. Built with Streamlit, this calculator provides an intuitive interface for solving linear systems of various sizes.

## Method Explanation

### Gauss-Jordan Elimination Method

The application implements the Gauss-Jordan elimination method, which is a systematic procedure for solving systems of linear equations. The key aspects of the implementation include:

1. **Matrix Representation**: The system Ax = b is represented as an augmented matrix [A|b].

2. **Algorithm Steps**:
   - Convert the augmented matrix to row echelon form
   - Transform to reduced row echelon form (RREF)
   - Each pivot is reduced to 1
   - All other entries in pivot columns are reduced to 0

3. **Key Features**:
   - Automatic pivot selection for numerical stability
   - Error tracking for convergence monitoring
   - Handles 2×2, 3×3, and 4×4 systems
   - Detects unique, infinite, and no solutions


### Installation Steps
1. Clone the repository or download the source code
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
1. Navigate to the project directory
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
3. The app will open in your default web browser

## Usage Guide

1. **Matrix Size Selection**:
   - Use the slider to select matrix size (2×2 to 4×4)

2. **Input Values**:
   - Enter matrix coefficients using sliders or direct input
   - Values range from -50 to 50
   - Matrix A represents coefficients
   - Vector b represents constants

3. **Solution**:
   - Click "Solve" to compute the solution
   - Results show original matrix and RREF form
   - Solution type is automatically determined

## Sample Examples

### Example 1: Unique Solution
Input:
```
Matrix A:
[1.00  0.00]
[0.00  1.00]

Vector b:
[2.00]
[3.00]
```
Expected Output:
```
Solution:
w = 2.00
x = 3.00
```

### Example 2: No Solution
Input:
```
Matrix A:
[1.00  1.00]
[1.00  1.00]

Vector b:
[1.00]
[2.00]
```
Expected Output:
```
✗ No Solution
```

### Example 3: Infinite Solutions
Input:
```
Matrix A:
[1.00  1.00]
[2.00  2.00]

Vector b:
[2.00]
[4.00]
```
Expected Output:
```
∞ Infinite Solutions
```

## Features

- **Modern UI**: Clean, intuitive interface with real-time updates
- **Input Flexibility**: Both slider and direct numerical input
- **Visual Feedback**: Clear presentation of matrices and solutions
- **Error Handling**: Robust detection of solution types
- **Precision Control**: Adjustable tolerance for numerical calculations

## Technical Details

- **Tolerance**: Configurable numerical tolerance for solution convergence
- **Maximum Iterations**: Adjustable iteration limit for convergence
- **Input Range**: Matrix elements can range from -50 to 50
- **Precision**: Results displayed with appropriate decimal places 
