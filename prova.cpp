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
 
int findSecondSmallest(const VectorXcd& eigval){
    double lower = eigval.real()(0);
    int index=0;
    for(int i=0; i<eigval.size(); ++i){
        if(lower>eigval.real()(i) && eigval.real()(i)!=eigval.real().minCoeff()){
            lower = eigval.real()(i);
            index = i;
        }
    }
    return index;
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
        vg(i) = Ag.row(i).sum();
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
    cout << "x^t*Lg*x=0 (with x=[1, ..., 1]^t) so Lg is not positive definite (but it is symmetric)" << endl;

    cout << "------------Task 3------------" << endl; 
    cout << "min eigenvalue: " << es.eigenvalues().real().minCoeff() << endl;
    cout << "max eigenvalue: " << es.eigenvalues().real().maxCoeff() << endl;
    cout << "------------Task 4------------" << endl;
    int pos = findSecondSmallest(es.eigenvalues());
    cout << "smallest non 0 eigen value: " << es.eigenvalues()(pos) << endl;
    cout << "corresponding eigen vector: \n" << es.eigenvectors().col(pos) << endl;
    cout << "------------Task 5------------" << endl;
    SparseMatrix<double, RowMajor> As;
    loadMarket(As, "social.mtx");
    cout << "frobenius_norm: " << As.norm() << endl;
    cout << "------------Task 6------------" << endl;
    VectorXd vs(351);
    SparseMatrix<double, RowMajor> Ds(351,351);
    for(int i=0;i<351;++i){
        vs(i)=0;
        vs(i) = As.row(i).sum();
        if(vs(i) != 0){
            Ds.coeffRef(i,i) = vs(i);
        }
    }
    SparseMatrix<double, RowMajor> Ls(351,351);
    Ls = Ds - As;
    cout << "Ls is symmetric? " << isSymmetric(Ls) << endl;
    cout << "Non-zero entries: " << Ls.nonZeros() << endl;
    cout << "------------Task 7------------" << endl;
    Ls.coeffRef(1,1) = Ls.coeff(1,1)+0.2;
    saveMarket(Ls, "Ls.mtx");
    cout << "using: mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e pi -tol 10e-8" << endl;
    cout << "max eigenvalue: 7.837972e+00, iterations: 960" << endl;
    cout << "------------Task 8------------" << endl;

}