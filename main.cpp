#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

constexpr double TOL = 1e-8;

/** Check if a matrix is Symmetric
 * @tparam T Type of the matrix elements (e.g., double, int)
 * @param mat The input sparse matrix to check
 * @return true if the matrix is symmetric, false otherwise
 */
template<typename T>
bool isSymmetric(const SparseMatrix<T> &mat, const double tol = TOL) {
	if(mat.rows() != mat.cols())
		return false; // A non-square matrix cannot be symmetric
	return mat.isApprox(mat.transpose(), tol);	// Check if the matrix is approximately equal to its transpose
}

/** Check if a matrix is Positive Definite (PD) using its eigenvalues
 * @tparam T Type of the matrix elements (e.g., double, int)
 * @param mat The input sparse matrix to check
 * @param eigenvalues The eigenvalues of the matrix
 * @param tol Tolerance to consider an eigenvalue as positive; useful since they can be computed by numerical methods
 * @return true if the matrix is SPD, false otherwise
 */
bool isPD(const SparseMatrix<double> &mat, const VectorXd &eigenvalues, const double tol = 1e-10) {
	for(int i=0; i<eigenvalues.size(); ++i)
		if(eigenvalues(i) <= tol)
			return false; // If any eigenvalue is non-positive, the matrix is not PD
	return true; // All eigenvalues are positive, so the matrix is PD
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " <image_path>" << endl;
		return 1;
	}

	cout << endl << "--- Task 1 ---" << endl;
	// Hard-coded adjacency matrix of the graph
	constexpr int DIM = 9;
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

	cout << endl << "--- Task 2 ---" << endl;
	// Get the vector representing the by row sum of Ag
	VectorXi vg = Ag * VectorXi::Ones(DIM);
	// Compute the graph Laplacian Lg = Dg - Ag
	SparseMatrix<int> Dg = vg.asDiagonal().toDenseMatrix().sparseView();
	SparseMatrix<int> Lg = Dg - Ag;
	// Compute Lg * [1, 1, ..., 1]^T
	VectorXi y = Lg * VectorXi::Ones(DIM);

	cout << "Euclidean norm of y: " << y.norm() << endl;

	SelfAdjointEigenSolver<MatrixXd> es;
	VectorXd Lg_eigenvalues;
	// Check if Lg is SPD
	if(!isSymmetric(Lg)) {
		cout << "Lg isn't SPD." << endl;
	} else {
		// Since we know that Lg is symmetric and has only real values (so its hermitian is equal to its transpose),
		// we can compute its eigenvalues with SelfAdjointEigenSolver
		es.compute(Lg.cast<double>().toDense());
		Lg_eigenvalues = es.eigenvalues().real();
		// In order to determine if Lg is PD we can now use its eigenvalues
		cout << "Lg " << (isPD(Lg.cast<double>(), Lg_eigenvalues) ? "is" : "isn't") << " SPD." << endl;
	}

	cout << endl << "--- Task 3 ---" << endl;
	cout << "Smallest eigenvalue of Lg: " << Lg_eigenvalues.minCoeff() << endl;
	cout << "Largest eigenvalue of Lg: " << Lg_eigenvalues.maxCoeff() << endl;
	// Note: from the request we know that the smallest eigenvalue should be 0, even if it isn't due to numerical method approximation.

	cout << endl << "--- Task 4 ---" << endl;
	// Get all the eigenvectors
	MatrixXd Lg_eigenvectors = es.eigenvectors().real();
	// Since SelfAdjointEigenSolver already returns the eigenvalues sorted in ascending order,
	// we can directly search the first strictly positive one
	for(int i=0; i<Lg_eigenvalues.size(); ++i) {
		if(Lg_eigenvalues(i) > TOL) {
			cout << "Smallest strictly positive eigenvalue:" << Lg_eigenvalues(i) << endl
				 << "Corresponding eigenvector:" << endl;
			// print also the index of the eigenvector elements
			for(int j=0; j<Lg_eigenvectors.rows();)
				// put 8 blanks before the index with a format
				cout << setw(9) << ++j << " ";
			cout << endl << Lg_eigenvectors.col(i).transpose() << endl;
			break;
		}
	}
	cout << "Notice that the eigenvector values correspond to the intuitive clustering {1, 2, 3, 4} and {5, 6, 7, 8, 9}." << endl;

	cout << endl << "--- Task 5 ---" << endl;
	// Load the social adjacency matrix
	SparseMatrix<int> As;
	const char* input_matrix_path = argv[1];
	loadMarket(As, input_matrix_path);
	cout << "Frobenius norm of As: " << As.cast<double>().norm() << endl;

	cout << endl << "--- Task 6 ---" << endl;
	// Compute the graph Laplacian Ls = Ds - As
	VectorXi vs = As * VectorXi::Ones(As.cols());
	SparseMatrix<int> Ds = vs.asDiagonal().toDenseMatrix().sparseView();
	SparseMatrix<int> Ls = Ds - As;
	// Report Ls symmetry and the number of non-zeros entries
	cout << "Ls is " << (isSymmetric(Ls) ? "" : "not ") << "symmetric." << endl;
	cout << "Number of non-zero entries in Ls: " << Ls.nonZeros() << endl;

	cout << endl << "--- Task 7 ---" << endl;
	// Perform the perturbation Ls'[1, 1] = Ls[1, 1] + 0.2
	Ls.coeffRef(0, 0) += 0.2;
	// Export the perturbed matrix to a file
	const string Ls_perturbed_path = "Ls_perturbed.mtx";
	if(!saveMarket(Ls, Ls_perturbed_path)) {
		cerr << "Error saving the perturbed matrix Ls to file." << endl;
		exit(1);
	}

	cout << "Perturbed matrix Ls saved to " << Ls_perturbed_path << endl;

	cout << "Execute the following command using LIS library example to compute the largest eigenvalue of Ls inside the container." << endl
		 << "source /u/sw/etc/bash.bashrc" << endl
		 << "module load gcc-glibc lis" << endl
		 << "<path-to-lis-test>/eigen " << Ls_perturbed_path << " eigenvec.txt hist.txt -e pi -etol 1e-8" << endl;	// TODO: tune the parameters
	cout << "Check in the command output the computed eigenvalue and the iterations count." << endl;
	return 0;
}
