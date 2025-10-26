#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

constexpr double TOL = 1e-10;

//Needed due to incomatibility between Eigen and LIS
VectorXd loadVectorFromMtx(string filename){
	FILE* in = fopen(filename.c_str(),"r");
	if(in == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}
	char line_buffer[100];
	fscanf(in, "%99[^\n]%*c", line_buffer);
	int len;
	fscanf(in, "%d", &len);
	VectorXd y(len);
	int pos;
	double value;
	for(int i=0; i<len; ++i) {
		fscanf(in, "%d", &pos);
		fscanf(in, "%lf", &value);
		y(pos-1)=value;
	}
	fclose(in);
	return y;
}

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
bool isPD(const SparseMatrix<double> &mat, const VectorXd &eigenvalues, const double tol = TOL) {
	for(int i=0; i<eigenvalues.size(); ++i)
		if(eigenvalues(i) <= tol)
			return false; // If any eigenvalue is non-positive, the matrix is not PD
	return true; // All eigenvalues are positive, so the matrix is PD
}

/** Compute the permutation matrix P that sorts the input vector v (in ascending order)
 * @param v The input vector to sort
 * @return The permutation matrix P such that P * v is sorted in ascending order
 */
SparseMatrix<int> computePermutationMatrix(const VectorXd &v) {
	// Create a vector of indices
	vector<int> indices(v.size());
	for(int i = 0; i < v.size(); ++i)
		indices[i] = i;

	// Sort the indices based on the values in v
	sort(indices.begin(), indices.end(), [&v](int a, int b) {
		return v(a) > v(b);
	});

	// Create the permutation matrix P according to the indices sorting found
	SparseMatrix<int> P(v.size(), v.size());
	vector<Triplet<int>> tripletList;
	tripletList.reserve(v.size());
	for(int i = 0; i < v.size(); ++i)
		tripletList.emplace_back(i, indices[i], 1);
	P.setFromTriplets(tripletList.begin(), tripletList.end());

	return P;
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
			cout << Lg_eigenvectors.col(i).transpose() << endl;
			// print also the index of the eigenvector elements
			for(int j=0; j<Lg_eigenvectors.rows();)
				// put 8 blanks before the index with a format
				cout << setw(5) << ++j << setw(5) << "";
			cout << "(related note number)" << endl;
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
	SparseMatrix<double> Ls = (Ds - As).cast<double>();	// It's an integer matrix for now, but it will be intentionally perturbated with a double value later
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

	const string start_container_env =
		"source /u/sw/etc/bash.bashrc\n"s +
		"module load gcc-glibc lis"s;

	cout << "Perturbed matrix Ls saved to " << Ls_perturbed_path << endl;

	cout << "Execute the following command using LIS library example to compute the largest eigenvalue of Ls inside the container." << endl
		 << start_container_env << endl
		 << "<path-to-lis-test>/eigen1 " << Ls_perturbed_path << " eigenvec.mtx hist.txt -e pi -etol 1e-8 -emaxiter 2500" << endl;
	cout << endl << "Check in the command output the computed eigenvalue and the iterations count." << endl;

	cout << endl << "--- Task 8 ---" << endl;

	cout << "The eigenvalue computed with LIS is around 60; in order to compute again the largest eigenvalue by employing a shift execute the following command inside the container." << endl
		 << start_container_env << endl
		 << "<path-to-lis-test>/eigen1 " << Ls_perturbed_path << " eigenvec.mtx hist.txt -e ii -etol 1e-8 -shift 60" << endl;
	cout << endl << "Check in the command output the iterations count and report the shift." << endl;

	cout << endl << "--- Task 9 ---" << endl;
	cout << "In order to compute the second smallest eigenvalue execute the following command inside the container." << endl
		 << start_container_env << endl
		 << "# pay attention to the tighter tolerance!" << endl
		 << "<path-to-lis-test>/eigen1 " << Ls_perturbed_path << " eigenvec.mtx hist.txt -e si -ie ii -ss 2 -etol 1e-10" << endl; // using Lanzos and Arnoldi methods the eigenvalues aren't the same (TODO: check it)
																																  // EDIT: also the eigenvector doesn't returns the same result... do we want to investigate it? (maybe we could just use the shift method directly, the one reported below)
	cout << endl << "Or equivalently, using shift:" << endl
		 << start_container_env << endl
		 << "# pay attention to the tighter tolerance!" << endl
		 << "<path-to-lis-test>/eigen1 " << Ls_perturbed_path << " eigenvec.mtx hist.txt -e ii -shift 1.7 -etol 1e-10" << endl;	// is it cheating? :)
	cout << endl << "Check in the command output the iterations count and report the second smallest eigenvalue." << endl;

	cout << endl << "--- Task 10 ---" << endl;

	// Get the filename from user input
	string filename_eigenvec, default_filename = "eigenvec.mtx";
	cout << "Enter the name of the eigenvector file [default: " << default_filename << "]: ";
	getline(cin, filename_eigenvec);
	if(filename_eigenvec.empty())
		filename_eigenvec = default_filename;
	// Check if the file exists
	ifstream file_check(filename_eigenvec);
	if(!file_check) {
		cerr << "Error: File " << filename_eigenvec << " does not exist." << endl;
		return 1;
	}

	// Load the eigenvector from the .mtx file
	VectorXd eigenvector;
	eigenvector = loadVectorFromMtx(filename_eigenvec);
	if(eigenvector.size() != Ls.rows())
		cerr << "Error: The size of the eigenvector (" << eigenvector.size() << ") does not match the size of the matrix Ls (" << Ls.rows() << "x" << Ls.cols() << ")." << endl;
	cout << endl;

	// Order the eigenvector entries
	SparseMatrix<int> P = computePermutationMatrix(eigenvector);
	VectorXd sorted_eigenvector = P.cast<double>().toDense() * eigenvector;

	// Report the number of positive and negative entries in the eigenvector
	int n_n = 0;
	for(int i = 0; i < eigenvector.size(); ++i)
		if(eigenvector(i) < TOL)
			++n_n;
	int n_p = eigenvector.size() - n_n;
	cout << "Number of positive entries in the eigenvector (n_p): " << n_p << endl;
	cout << "Number of negative entries in the eigenvector (n_n): " << n_n << endl;

	cout << endl << "--- Task 11 ---" << endl;

	// Compute the reordered adjacency matrix Aord = P * As * P^T
	SparseMatrix<int> Aord = P * As * P.transpose();

	// Report the number of non-zero entries in the block Aord[0:n_p-1, n_p:n_p+n_n-1]
	int row_start = 0,
		row_end = n_p,
		col_start = n_p,
		col_end = n_p + n_n;

	SparseMatrix<int> Aord_submatrix = Aord.block(row_start, col_start, row_end - row_start, col_end - col_start);
	cout << "Number of non-zero entries in the block Aord[" << row_start << ":" << row_end-1 << ", " << col_start << ":" << col_end-1 << "]: " << Aord_submatrix.nonZeros() << endl;

	// Report the number of non-zero entries in the block As[0:n_p-1, n_p:n_p+n_n-1]
	SparseMatrix<int> As_submatrix = As.block(row_start, col_start, row_end - row_start, col_end - col_start);
	cout << "Number of non-zero entries in the block As[" << row_start << ":" << row_end-1 << ", " << col_start << ":" << col_end-1 << "]: " << As_submatrix.nonZeros() << endl;

	return 0;
}
