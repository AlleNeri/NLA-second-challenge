#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra> // for saving in .mtx format
#include <random> // Add this if not already included

using namespace Eigen;
using namespace std;


bool isSymmetric(const MatrixXd& mat) {
  if (mat.rows() != mat.cols()) {
	return false; // A non-square matrix cannot be symmetric
  }
  return mat.isApprox(mat.transpose());	// Check if the matrix is approximately equal to its transpose, considering numerical precision
}

//TO CHECK IF POSITIVE DEFINITE DEFINITION IS RIGHT
bool isPositiveDefinite(const VectorXcd& eigval){
    bool result = true;
    for(int i=0; i<eigval.size() && result == true; ++i){
        if(eigval(i).real() < 0 || eigval(i).imag() != 0){
            result = false;
        }
    }
    return result;
}

int findPos(const VectorXcd& eigval, double tofind){
    for(int i=0; i<eigval.size(); ++i){
        if(eigval(i) == tofind){
            return i;
        }
    }
    cout << "not found" << endl;
    return -1;
}

int main(int argc, char* argv[]) {
    cout << "------------Task 1------------" << endl; 
    MatrixXd Ag(9,9);
    Ag <<    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	        1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	        0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0;
    float frobenius_norm = Ag.norm(); //ARE WE SURE THIS IS RIGHT?
    cout << "frobenius_norm: " << frobenius_norm << endl;
    VectorXd vg(9);
    for(int i=0;i<9;++i){
        vg(i)=0;
        for(int j=0;j<9;++j){
            vg(i) = vg(i) + Ag(i,j);
        }
    }
    MatrixXd Lg(9,9);
    MatrixXd Dg(9,9);
    Dg.diagonal() = vg;
    Lg = Dg - Ag;
    VectorXd x(9);
    cout << "------------Task 2------------" << endl; 
    x = VectorXd::Ones(9);
    VectorXd y = Lg*x;
    cout << "y: " << y << endl;

    EigenSolver<Eigen::MatrixXd> es(Lg);
std::cout << "The eigenvalues of A are:\n" << es.eigenvalues() << "\n\n";

    // Eigenvectors are returned as a matrix of complex numbers, 
    // where each column is an eigenvector.
std::cout << "The eigenvectors of A are:\n" << es.eigenvectors() << "\n\n";

if(isSymmetric(Lg) && isPositiveDefinite(es.eigenvalues())){
    cout << "symmetric positive definite" << endl;
}

cout << "------------Task 3------------" << endl; 
cout << "min eigenvalue: " << es.eigenvalues().real().minCoeff() << endl;
cout << "max eigenvalue: " << es.eigenvalues().real().maxCoeff() << endl;
cout << "------------Task 4------------" << endl;
VectorXd eigval_sorted = es.eigenvalues().real();
sort(eigval_sorted.data(), eigval_sorted.data()+eigval_sorted.size());
cout << "smallest non 0 eigen value: " << eigval_sorted(1) << endl;
cout << "corresponding eigen vector: " << es.eigenvectors().col(findPos(es.eigenvalues().real(), eigval_sorted(1))) << endl;
}