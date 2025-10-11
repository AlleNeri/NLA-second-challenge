#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

int main() {
    std::cout << "--- Task 1 ---\n" << std::endl;

    const int num_nodes = 9;
    typedef Eigen::SparseMatrix<double> SpMat;  // declares a column-major sparse matrix type of double
    typedef Eigen::Triplet<int> T;
    std::vector<T> tripletsVector;

    // Define edges (undirected)
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {0, 3}, {2, 3}, {2, 1}, {2, 4}, {4, 8},
        {4, 7}, {4, 5}, {8, 6}, {8, 7}, {6, 7}, {6, 5}
    };

    // Each edge adds two symmetric entries (i,j) and (j,i)
    for (auto& edge : edges) {
        int i = edge.first;
        int j = edge.second;
        tripletsVector.emplace_back(T(i, j, 1));
        tripletsVector.emplace_back(T(j, i, 1));
    }

    // Initialize and fill the sparse matrix
    SpMat Ag(num_nodes, num_nodes);
    Ag.setFromTriplets(tripletsVector.begin(), tripletsVector.end());

    // Print the matrix (non-zero entries)
    std::cout << "Sparse matrix contents:\n";
    for (int k = 0; k < Ag.outerSize(); ++k) {
        for (SpMat::InnerIterator it(Ag, k); it; ++it) {
            std::cout << "(" << it.row() << ", " << it.col() << ") = "
                    << it.value() << "\n";
        }
    }

    // Optionally, print the full dense matrix for verification
    std::cout << "\nFull dense matrix:\n" << Eigen::MatrixXd(Ag) << std::endl;

    // Report the Frobenius norm of Ag
    std::cout << "\nFrobenius norm of Ag: " << Ag.norm() << std::endl; 


    std::cout << "\n--- Task 2 ---\n" << std::endl;

    // Construct the vector vg such that each component vi is the sum of the entries in the i-th row of matrix Ag.
    Eigen::VectorXd vg(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        vg(i) = Ag.row(i).sum();
    }

    // Construct the degree matrix Dg
    Eigen::SparseMatrix<double> Dg(num_nodes, num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        Dg.insert(i, i) = vg(i);
    }

    // Compute the graph Laplacian Lg = Dg - Ag
    Eigen::SparseMatrix<double> Lg = Dg - Ag;

    std::cout << "Matrix Lg:\n" << Eigen::MatrixXd(Lg) << std::endl;

    // Compute the matrix-vector product y = Lg * x, with x = [1, 1, ..., 1]^T
    Eigen::VectorXd x = Eigen::VectorXd::Ones(num_nodes);
    Eigen::VectorXd y = Lg * x;

    std::cout << "\nResult y = Lg * x:\n" << y << std::endl;

    std::cout << "\nEuclidean norm of y: " << y.norm() << std::endl;


    std::cout << "\n--- Task 3 ---\n" << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> eigensolver(Lg);
    if (eigensolver.info() != Eigen::Success) abort();
    std::cout << "Eigenvalues of Lg:\n" << eigensolver.eigenvalues() << std::endl;
    std::cout << "\nMatrix of eigenvectors (columns) of Lg:\n" << eigensolver.eigenvectors() << std::endl;
    std::cout << "\nSmallest eigenvalue: " << eigensolver.eigenvalues()[0] << std::endl;
    std::cout << "Largest eigenvalue: " << eigensolver.eigenvalues()[num_nodes - 1] << std::endl;


    std::cout << "\n--- Task 4 ---\n" << std::endl;

    std::cout << "Second smallest eigenvalue: " << eigensolver.eigenvalues()[1] << std::endl;
    std::cout << "Corresponding eigenvector: " << eigensolver.eigenvectors().col(1).transpose() << std::endl;

    
    std::cout << "\n--- Task 5 ---\n" << std::endl;

    Eigen::SparseMatrix<double> As;
    Eigen::loadMarket(As, "social.mtx");
    std::cout << "Size of Matrix As: " << As.rows() << " x " << As.cols() << std::endl;
    std::cout << "Non-zeros of Matrix As: " << As.nonZeros() << std::endl;

    // Report the Frobenius norm of A
    std::cout << "Frobenius norm of As: " << As.norm() << std::endl; 


    std::cout << "\n--- Task 6 ---\n" << std::endl;

    // Construct the vector vs such that each component vi is the sum of the entries in the i-th row of matrix As.
    Eigen::VectorXd vs(As.rows());
    for (int i = 0; i < As.rows(); ++i) {
        vs(i) = As.row(i).sum();
    }

    // Construct the degree matrix Ds
    Eigen::SparseMatrix<double> Ds(As.rows(), As.cols());
    for (int i = 0; i < As.rows(); ++i) {
        Ds.insert(i, i) = vs(i);
    }

    // Compute the graph Laplacian Ls = Ds - As
    Eigen::SparseMatrix<double> Ls(As.rows(), As.cols());
    Ls = Ds - As;

    std::cout << "Size of Matrix Ls: " << Ls.rows() << " x " << Ls.cols() << std::endl;
    std::cout << "Non-zeros of Matrix Ls: " << Ls.nonZeros() << std::endl;
    std::cout << "Matrix Ls is symmetric? " << std::boolalpha << Ls.isApprox(Ls.transpose()) << std::endl;


    std::cout << "\n--- Task 7 ---\n" << std::endl;

    // In order to make the graph laplacian matrix invertible,
    // add a small perturbation to the first diagonal entry of Ls
    std::cout << "Value Ls(0,0): " << Ls.coeff(0, 0) << std::endl;
    Ls.coeffRef(0, 0) += 0.2;
    std::cout << "Modified Ls(0,0): " << Ls.coeff(0, 0) << std::endl;
    
    // Export Ls in the .mtx format
    std::string matrixFileOut("./Ls.mtx");
    Eigen::saveMarket(Ls, matrixFileOut);
    // Move it to the lis-2.1.10/test folder.
    // Using the proper iterative solver available in the LIS library
    // compute the largest eigenvalue of Ls up to a tolerance of 10−8.

    // ../../lis-2.1.10/test/eigen1 Ls.mtx eigvec.txt hist.txt -e pi -etol 1.0e-8
    std::cout << "Using: ./../../lis-2.1.10/test/eigen1 Ls.mtx eigvec.txt hist.txt -e pi -emaxiter 3000 -etol 1.0e-8\n"<< std::endl;
    std::cout << "Max eigenvalue: 6.013370e+01,  number of iterations: 2007\n"<< std::endl;
    
    
    std::cout << "\n--- Task 8 ---\n" << std::endl;
    
    // ../../lis-2.1.10/test/eigen1 Ls.mtx eigvec.txt hist.txt -e pi -etol 1.0e-8 -shift 60.
    std::cout << "Using: ./../../lis-2.1.10/test/eigen1 Ls.mtx eigvec.txt hist.txt -e ii -etol 1.0e-8 -shift 60.\n"<< std::endl;
    std::cout << "Max eigenvalue: 6.013370e+01, number of iterations: 25\n"<< std::endl;
    return 0;
}

