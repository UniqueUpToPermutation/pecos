#include <iostream>

#include <ctime>
#include <set>
#include <filesystem>

#include "pecos/core/xmc/inference.hpp"
#include "pecos/core/utils/scipy_loader.hpp"

struct Prediction {
    int label;
    double value; // labels's value/probability/loss
    Prediction(){ label = 0; value = 0; }
    Prediction(int label, double value): label(label), value(value) {}

    bool operator<(const Prediction& r) const { return value < r.value; }

    friend std::ostream& operator<<(std::ostream& os, const Prediction& p) {
        os << p.label << ":" << p.value;
        return os;
    }
};

void ComputeRecallPrecision(
	const std::vector<std::vector<Prediction>>& ground_truth, 
	const std::vector<std::vector<Prediction>>& predictions,
	int topK,
	std::vector<double>& recall,
	std::vector<double>& precision) {

	recall.resize(topK);
	precision.resize(topK);

	std::fill(recall.begin(), recall.end(), 0.0);
	std::fill(precision.begin(), precision.end(), 0.0);

	for (int i = 0; i < ground_truth.size(); ++i) {
		auto& truth = ground_truth[i];
		auto& prediction = predictions[i];

		std::set<int> truth_labels;

		for (auto& t : truth) {
			truth_labels.emplace(t.label);
		}

		for (int k = 1; k <= topK; ++k) {
			std::set<int> pred_labels;
			std::set<int> pred_truth_labels;

			for (int j = 0; j < k; ++j) {
				pred_labels.emplace(prediction[j].label);
			}

			std::set_intersection(truth_labels.begin(), truth_labels.end(), 
				pred_labels.begin(), pred_labels.end(),
				std::inserter(pred_truth_labels, pred_truth_labels.begin()));

			recall[k-1] += (double)pred_truth_labels.size() / (double)truth_labels.size();
			precision[k-1] += (double)pred_truth_labels.size() / (double)pred_labels.size();
		}
	}

	for (auto& r : recall) {
		r /= (double)ground_truth.size();
	}

	for (auto& p : precision) {
		p /= (double)ground_truth.size();
	}
}

std::vector<std::vector<Prediction>> ProcessPredictions(const pecos::csr_t& mat) {
	std::vector<std::vector<Prediction>> result;
	result.reserve(mat.rows);
	
	for (int row = 0; row < mat.rows; ++row) {

		auto start = mat.indptr[row];
		auto end = mat.indptr[row+1];
		auto len = end - start;

		std::vector<Prediction> row_vec;
		row_vec.reserve(len);

		for (auto i = start; i < end; ++i) {
			Prediction p;
			p.label = mat.indices[i];
			p.value = mat.val[i];
			row_vec.emplace_back(p);
		}

		result.emplace_back(std::move(row_vec));
	}

	return result;
}

void TestDataSet(const std::filesystem::path& path) {
	auto pecos_path = path / "model";

	if (!std::filesystem::exists(pecos_path) || !std::filesystem::is_directory(pecos_path)) {
		std::cout << path << " does not have a PECOS model. Skipping..." << std::endl;
		return;
	}

	pecos::csr_t X;
	pecos::csr_t Y;

	std::vector<std::vector<Prediction>> pecos_hash_csc_predictions;
	std::vector<std::vector<Prediction>> pecos_hash_chunked_predictions;

	int top_k = 10;
	int beam_size = 20; 

	{
		std::cout << "Loading " << path / "X.tst.tfidf.npz" << "..." << std::endl;
		pecos::ScipyCsrF32Npz X_npz(path / "X.tst.tfidf.npz");
		std::cout << "Loading " << path / "Y.tst.npz" << "..." << std::endl;
		pecos::ScipyCsrF32Npz Y_npz(path / "Y.tst.npz");
		X = pecos::csr_npz_to_csr_t_deep_copy(X_npz);
		Y = pecos::csr_npz_to_csr_t_deep_copy(Y_npz);
	}

	{
		std::cout << "Loading PECOS model (HASH CSC)" << pecos_path << "..." << std::endl;
		pecos::HierarchicalMLModel model(pecos_path, pecos::LAYER_TYPE_HASH_CSC);

		std::cout << "Running PECOS Prediction..." << std::endl;
		pecos::csr_t Y_pred;

		std::clock_t c_start = std::clock();
		model.predict<pecos::csr_t, pecos::csr_t>(X, Y_pred, beam_size, "sigmoid", top_k, 1);
		std::clock_t c_end = std::clock();

		long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
		std::cout << "CPU time per query: " << time_elapsed_ms / (double)X.rows << " ms\n";

		pecos_hash_csc_predictions = ProcessPredictions(Y_pred);
		Y_pred.free_underlying_memory();

		std::cout << std::endl;
	}

	{
		std::cout << "Loading PECOS model (HASH CHUNCKED)" << pecos_path << "..." << std::endl;
		pecos::HierarchicalMLModel model(pecos_path, pecos::LAYER_TYPE_HASH_CHUNKED);

		std::cout << "Running PECOS Prediction..." << std::endl;
		pecos::csr_t Y_pred;

		std::clock_t c_start = std::clock();
		model.predict<pecos::csr_t, pecos::csr_t>(X, Y_pred, beam_size, "sigmoid", top_k, 1);
		std::clock_t c_end = std::clock();

		long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
		std::cout << "CPU time per query: " << time_elapsed_ms / (double)X.rows << " ms\n";

		pecos_hash_chunked_predictions = ProcessPredictions(Y_pred);
		Y_pred.free_underlying_memory();

		std::cout << std::endl;
	}

	std::vector<double> hash_csc_recall;
	std::vector<double> hash_csc_precision;

	std::vector<double> hash_chunked_recall;
	std::vector<double> hash_chunked_precision;

	auto truth = ProcessPredictions(Y);

	ComputeRecallPrecision(truth, pecos_hash_csc_predictions, top_k, hash_csc_recall, hash_csc_precision);
	ComputeRecallPrecision(truth, pecos_hash_chunked_predictions, top_k, hash_chunked_recall, hash_chunked_precision);

	auto printPrecisionRecall = [top_k](const std::string& header, 
		const std::vector<double>& precision,
		const std::vector<double>& recall) {
		std::cout << "=========== " << header << " =============" << std::endl;
		std::cout << std::setw(10) << "prec@k";
		for (int i = 0; i < top_k; ++i) {
			std::cout << std::setw(10) << precision[i];
		}
		std::cout << std::endl;
		std::cout << std::setw(10) << "recall@k";
		for (int i = 0; i < top_k; ++i) {
			std::cout << std::setw(10) << recall[i];
		}
		std::cout << std::endl << std::endl;
	};

	printPrecisionRecall("PECOS (HASH CSC)", hash_csc_precision, hash_csc_recall);
	printPrecisionRecall("PECOS (HASH CHUNKED)", hash_chunked_precision, hash_chunked_recall);

	X.free_underlying_memory();
	Y.free_underlying_memory();
}

int main(int argc, char *argv[]) {

	std::vector<std::filesystem::path> data_dirs;

	if (argc < 2) {
		auto path = std::filesystem::path(DATA_DIR);

		for (auto entry : std::filesystem::directory_iterator(path)) {
			if (entry.is_directory()) {
				data_dirs.emplace_back(entry.path());
			}
		}
	} else {
		for (int i = 1; i < argc; ++i) {
			data_dirs.emplace_back(argv[i]);
		}
	}

	for (auto dir : data_dirs) {
		if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
			TestDataSet(dir);
		}
	}
}