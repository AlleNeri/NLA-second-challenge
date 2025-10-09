#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

int main() {
	cout << "--- Task 1 ---" << endl;
	// Hard-coded adjacency matrix of the graph
	const int DIM = 9;
	MatrixXi Ag_dense(DIM, DIM);
	Ag_dense << 0, 1, 0, 1, 0, 0, 0, 0, 0,
				1, 0, 1, 0, 0, 0, 0, 0, 0,
				0, 1, 0, 1, 1, 0, 0, 0, 0,
				1, 0, 1, 0, 0, 0, 0, 0, 0,
				0, 0, 1, 0, 0, 1, 0, 1, 1,
				0, 0, 0, 0, 1, 0, 1, 0, 0,
				0, 0, 0, 0, 0, 1, 0, 1, 1,
				0, 0, 0, 0, 1, 0, 1, 0, 1,
				0, 0, 0, 0, 1, 0, 1, 1, 0;
	// Convert to sparse matrix
	SparseMatrix<int> Ag = Ag_dense.sparseView();
	Ag_dense.resize(0, 0);	// free memory

	cout << "Frobenius norm of Ag: " << Ag.cast<double>().norm() << endl;

	return 0;
}
