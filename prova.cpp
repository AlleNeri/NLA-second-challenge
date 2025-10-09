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

//to be completed, is not really efficient
bool isPositiveDefinite(const MatrixXd& mat){
    EigenSolver<Eigen::MatrixXd> es(mat);
    VectorXd eigval = es.eigenvalues();
    bool result = true;
    for(double i : eigval){
        if(i < 0){
            result = false;
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
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
    cout << "frobenius_norm" << frobenius_norm << endl;
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
    x = VectorXd::Ones(9);
    VectorXd y = Lg*x;
    cout << "y: " << y << endl;
    /*MatrixXd Hsh1(9, 9);
    Hsh1 << 2.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	        -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	        0.0, -1.0, 3.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0, 4.0, -1.0, 0.0, -1.0, -1.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 3.0, -1.0, -1.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.0, -1.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, -1.0, 3.0;
    MatrixXd asd(3,3);
    asd << 1.0, -1.0, 0.0,
            -1.0, 2.0, -1.0,
            0.0, -1.0, 1.0;*/
saveMarket(Lg, "mia_prova.mtx");
EigenSolver<Eigen::MatrixXd> es(Lg);
std::cout << "The eigenvalues of A are:\n" << es.eigenvalues() << "\n\n";

    // Eigenvectors are returned as a matrix of complex numbers, 
    // where each column is an eigenvector.
std::cout << "The eigenvectors of A are:\n" << es.eigenvectors() << "\n\n";

/*if(isSymmetric(Lg) && isPositiveDefinite(Lg)){
    cout << "symmetric positive definite" << endl;
}*/

}