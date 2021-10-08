#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <map>
#include <set>

#include <xmc/inference.hpp>
#include "json.hpp"

#define FREE_MAT(x) x.free_underlying_memory()
#define QUERY_STATS(x) pecos::query_statistics_t::compute(x)

#ifdef _GAUNTLET_PROFILER_
#include <gperftools/profiler.h>
#define PROFILER_START(MACRO, DATASET) ProfilerStart( \
    (string("profiles/") + MACRO + "@" + DATASET + ".prof").c_str())
#define PROFILER_STOP ProfilerStop()
#else
#define PROFILER_START(MACRO, DATASET)
#define PROFILER_STOP
#endif

#define INCLUDE_DENSE_LOOKUP
#define INCLUDE_HASH_CSC
#define INCLUDE_MARCHING_POINTERS

using namespace std;
using namespace nlohmann;
using namespace pecos;

using duration_t = std::chrono::duration<double>;
using time_point_t = std::chrono::high_resolution_clock::time_point;

csr_t load_matrix_csr_from_npz(const std::string& path) {
    ScipyCsrF32Npz mat_npz;
    mat_npz.load(path);
    return csr_npz_to_csr_t_deep_copy(mat_npz);
}

struct metrics_t {
    uint32_t num;
    double precision;
    double recall;
};

class PerformanceTimer : public IPerformanceTimer {
private:
    std::vector<time_point_t> startTimes;
    std::vector<time_point_t> endTimes;
    std::vector<duration_t> durations;
    std::vector<duration_t> cummulativeDurations;

    time_point_t totalStart;
    time_point_t totalEnd;
    duration_t totalDuration;

public:
    PerformanceTimer() {
        cummulativeDurations.emplace_back(0);
    }

    void Begin() {
        totalStart = std::chrono::high_resolution_clock::now();
    }

    void End() {
        totalEnd = std::chrono::high_resolution_clock::now();
        totalDuration = totalEnd - totalStart;
    }

    void BeginLayer() override {
        startTimes.emplace_back(std::chrono::high_resolution_clock::now());
    }

    void EndLayer() override {
        endTimes.emplace_back(std::chrono::high_resolution_clock::now());
        durations.emplace_back(endTimes.back() - startTimes.back());
        cummulativeDurations.emplace_back(cummulativeDurations.back() + durations.back());
    }

    const auto& GetLayerDurations() const {
        return durations;
    }

    const auto& GetCummulativeLayerDurations() const {
        return cummulativeDurations;
    }

    auto GetTotalDuration() const {
        return totalDuration;
    }
};

struct gauntlet_run_params_t {
    std::string mode;
    std::string dataset;
    std::string macro;
    int threads;
    uint32_t beam_size;
    uint32_t only_top_k;
    layer_type_t layer_type;
    json config;
};

typedef struct {
    unsigned long size,resident,share,text,lib,data,dt;
} statm_t;

string get_mem_str(unsigned long size) {
    string markers[] = { "B", "KB", "MB", "GB", "TB" };
    uint32_t indx = 0;

    double sz = size;
    while (sz > 1000.0) {
        sz /= 1024.0;
        ++indx;
    }

    stringstream s;
    s << setprecision(3) << sz << " " << markers[indx];
    return s.str();
}

void print_mem_stat(const statm_t& stat, const size_t page_size) {
    cout << left << setw(10) << "size:" << setw(10)
        << get_mem_str(stat.size * page_size) << endl;
    cout << left << setw(10) << "resident:" << setw(10)
        << get_mem_str(stat.resident * page_size) << endl;
    cout << left << setw(10) << "share:" << setw(10)
        << get_mem_str(stat.share * page_size) << endl;
    cout << left << setw(10) << "text:" << setw(10)
        << get_mem_str(stat.text * page_size) << endl;
    cout << left << setw(10) << "lib:" << setw(10)
        << get_mem_str(stat.lib * page_size) << endl;
    cout << left << setw(10) << "data:" << setw(10)
        << get_mem_str(stat.data * page_size) << endl;
    cout << left << setw(10) << "dt:" << setw(10)
        << get_mem_str(stat.dt * page_size) << endl;
}

statm_t read_off_memory_status()
{
    statm_t result;
    unsigned long dummy;
    const char* statm_path = "/proc/self/statm";

    FILE *f = fopen(statm_path,"r");
    if (!f){
        perror(statm_path);
        abort();
    }
    if (7 != fscanf(f,"%ld %ld %ld %ld %ld %ld %ld",
        &result.size,&result.resident,&result.share,&result.text,&result.lib,&result.data,&result.dt))
    {
        perror(statm_path);
        abort();
    }
    fclose(f);
    return result;
}

string get_layer_type_str(const layer_type_t type) {
    switch (type) {
    case LAYER_TYPE_CSC:
        return "csc";
    case LAYER_TYPE_HASH_CHUNKED:
        return "hash-chunked";
    case LAYER_TYPE_BINARY_SEARCH_CHUNKED:
        return "binary-search-chunked";
#ifdef INCLUDE_HASH_CSC
    case LAYER_TYPE_HASH_CSC:
        return "hash-csc";
#endif
#ifdef INCLUDE_DENSE_LOOKUP
    case LAYER_TYPE_DENSE_LOOKUP_CHUNKED:
        return "dense-lookup-chunked";
    case LAYER_TYPE_DENSE_LOOKUP_CSC:
        return "dense-lookup-csc";
#endif
#ifdef INCLUDE_MARCHING_POINTERS
    case LAYER_TYPE_MARCH_CSC:
        return "march-csc";
    case LAYER_TYPE_MARCH_CHUNKED:
        return "march-chunked";
#endif
    default:
        return "unknown";
    }
}

// Used to bucket benchmark runs that use the same model to prevent unnecessary
// model reloading.
string to_model_cmp_str(const gauntlet_run_params_t& params) {
    std::stringstream s;
    s << params.dataset;
    s << get_layer_type_str(params.layer_type);
    return s.str();
}

string to_dataset_comp_str(const gauntlet_run_params_t& params) {
    return params.dataset;
}

struct gauntlet_run_results_t {
    std::vector<metrics_t> metrics;
    gauntlet_run_params_t params;
    query_statistics_t input_statistics;
    std::vector<layer_statistics_t> layer_statistics;
    std::vector<duration_t> layer_durations;
    duration_t total_duration;
    duration_t total_duration_incl_overhead;
    duration_t query_preprocess_time;
    duration_t query_postprocess_time;
    statm_t memory_stat;
    bool b_realtime;
};

struct gauntlet_results_t {
    std::vector<gauntlet_run_results_t> runs;
};

gauntlet_run_params_t read_macro(const json& macro_json, const string& macro_name) {
   gauntlet_run_params_t params;
    macro_json["mode"].get_to(params.mode);
    macro_json["threads"].get_to(params.threads);
    macro_json["beamsize"].get_to(params.beam_size);
    macro_json["onlytopk"].get_to(params.only_top_k);

    // params.query_processor = nullptr;
    params.config = macro_json;
    params.macro = macro_name;

    // Read the layer type
    string layer_type_str;
    macro_json["layertype"].get_to(layer_type_str);
    if (layer_type_str == "csc") {
        params.layer_type = LAYER_TYPE_CSC;
    } else if (layer_type_str == "hash-chunked") {
        params.layer_type = LAYER_TYPE_HASH_CHUNKED;
    } else if (layer_type_str == "binary-search-chunked") {
        params.layer_type = LAYER_TYPE_BINARY_SEARCH_CHUNKED;
    }
#ifdef INCLUDE_HASH_CSC
    else if (layer_type_str == "hash-csc") {
        params.layer_type = LAYER_TYPE_HASH_CSC;
    }
#endif
#ifdef INCLUDE_DENSE_LOOKUP
    else if (layer_type_str == "dense-lookup-chunked") {
        params.layer_type = LAYER_TYPE_DENSE_LOOKUP_CHUNKED;
    } else if (layer_type_str == "dense-lookup-csc") {
        params.layer_type = LAYER_TYPE_DENSE_LOOKUP_CSC;
    }
#endif
#ifdef INCLUDE_MARCHING_POINTERS
    else if (layer_type_str == "march-chunked") {
        params.layer_type = LAYER_TYPE_MARCH_CHUNKED;
    } else if (layer_type_str == "march-csc") {
        params.layer_type = LAYER_TYPE_MARCH_CSC;
    }
#endif
    else {
        params.layer_type = LAYER_TYPE_HASH_CHUNKED;
        params.config["layertype"] = "hash-chunked";
    }

    return params;
}

// Read all configurations and queued benchmarks from a json file
pecos::unordered_map<string, gauntlet_run_params_t> read_gaunlet_macros(const json& j) {
    pecos::unordered_map<string, gauntlet_run_params_t> result;

    for (auto& it : j.items()) {
        auto& macro_json = it.value();
        auto params = read_macro(macro_json, it.key());
        result[params.macro] = params;
    }

    return result;
}

vector<gauntlet_run_params_t> read_guantlet_multi_runs(const json& j,
    pecos::unordered_map<string, gauntlet_run_params_t>& macro_map) {

    vector<gauntlet_run_params_t> result;

    for (auto& it : j.items()) {
        vector<string> macros;
        vector<string> datasets;

        auto multi_run = it.value();
        multi_run["macros"].get_to(macros);
        multi_run["datasets"].get_to(datasets);

        for (auto& macro : macros) {
            for (auto& dataset : datasets) {
                gauntlet_run_params_t run_params = macro_map[macro];
                run_params.dataset = dataset;
                result.push_back(run_params);
            }
        }
    }

    return result;
}

vector<gauntlet_run_params_t> read_guantlet_single_runs(const json& j,
    pecos::unordered_map<string, gauntlet_run_params_t>& macro_map) {
    vector<gauntlet_run_params_t> result;

    for (auto& it : j.items()) {
        string macro;
        string dataset;

        auto single_run = it.value();
        gauntlet_run_params_t run_params;

        // Copy parameters over from macro if the run has a macro defined
        if (single_run.contains("macro")) {
            single_run["macro"].get_to(macro);
            run_params = macro_map[macro];
        }
        else
            run_params = read_macro(single_run, "");

        single_run["dataset"].get_to(dataset);
        run_params.dataset = dataset;
        result.push_back(run_params);
    }

    return result;
}

// Evaluate precision and recall
vector<metrics_t> eval_metrics(const csr_t& groundtruth, const csr_t& predicted, const uint32_t topk) {
    assert(groundtruth.rows == predicted.rows);
    assert(groundtruth.cols == predicted.cols);

    csr_t::mem_index_type total_truth_items = 0;
    csr_t::mem_index_type total_predicted_items = 0;
    csr_t::mem_index_type total_correct_predicted_items = 0;
    vector<csr_t::mem_index_type> total_matched;
    vector<double> recall;
    total_matched.resize(topk);
    recall.resize(topk);
    for (auto& t : total_matched)
        t = 0;
    for (auto& r : recall)
        r = 0.0;

    std::vector<metrics_t> result;

    for (csr_t::index_type row = 0; row < predicted.rows; ++row) {

        vector<csr_t::mem_index_type> found_idx_vec;
        vector<csr_t::index_type> found_items;
        vector<csr_t::index_type> truth_items;
        vector<bool> matched;

        for (csr_t::mem_index_type i = groundtruth.row_ptr[row]; i < groundtruth.row_ptr[row+1]; ++i)
            truth_items.push_back(groundtruth.col_idx[i]);
        for (csr_t::mem_index_type i = predicted.row_ptr[row]; i < predicted.row_ptr[row+1]; ++i)
            found_idx_vec.push_back(i);

        std::sort(found_idx_vec.begin(), found_idx_vec.end(),
            [predicted](csr_t::mem_index_type a, csr_t::mem_index_type b) {
            return predicted.col_idx[a] < predicted.col_idx[b];
        });
        std::stable_sort(found_idx_vec.begin(), found_idx_vec.end(),
            [predicted](csr_t::mem_index_type a, csr_t::mem_index_type b) {
            return predicted.val[a] > predicted.val[b];
        });

        found_items.reserve(found_idx_vec.size());
        for (csr_t::index_type i = 0; i < found_idx_vec.size(); ++i)
            found_items.push_back(predicted.col_idx[found_idx_vec[i]]);

        size_t matched_size = std::min<size_t>(topk, found_items.size());
        matched.resize(matched_size);
        for (csr_t::index_type i = 0; i < matched_size; ++i) {
            auto item = found_items[i];
            matched[i] = (std::find(truth_items.begin(),
                truth_items.end(), item) != truth_items.end());
        }

        vector<csr_t::index_type> cum_matched;
        cum_matched.resize(matched_size);
        cum_matched[0] = matched[0];
        for (csr_t::index_type i = 1; i < matched_size; ++i)
            cum_matched[i] = cum_matched[i - 1] + matched[i];

        for (csr_t::index_type i = 0; i < matched_size; ++i) {
            total_matched[i] += cum_matched[i];
            recall[i] += (double)cum_matched[i] / (double)std::max<csr_t::index_type>(truth_items.size(), 1);
        }
        if (cum_matched.size() > 0) {
            for (csr_t::index_type i = matched_size; i < topk; ++i) {
                total_matched[i] += cum_matched[cum_matched.size() - 1];
                recall[i] += (double)cum_matched[cum_matched.size() - 1] / (double)std::max<csr_t::index_type>(truth_items.size(), 1);
            }
        }
    }

    for (csr_t::index_type i = 0; i < topk; ++i) {
        metrics_t m;
        m.num = i + 1;
        m.precision = (double)total_matched[i] / (double)predicted.rows / (double)(i + 1);
        m.recall = (double)recall[i] / (double)predicted.rows;
        result.push_back(m);
    }

    return result;
}

// Sorts each row entry by column
csr_t sort_csr(const csr_t& mat) {
    csr_t::mem_index_type nnz = mat.row_ptr[mat.rows];

    csr_t result;
    result.rows = mat.rows;
    result.cols = mat.cols;
    result.col_idx = new csr_t::index_type[nnz];
    result.row_ptr = new csr_t::mem_index_type[mat.rows + 1];
    result.val = new float[nnz];

    std::memcpy(result.row_ptr, mat.row_ptr, sizeof(csr_t::mem_index_type) * (mat.rows + 1));

    auto sort_idx = new csr_t::mem_index_type[nnz];
    for (csr_t::mem_index_type i = 0; i < nnz; ++i)
        sort_idx[i] = i;

    for (csr_t::index_type row = 0; row < mat.rows; ++row) {
        // Sort by col_idx
        std::sort(&sort_idx[mat.row_ptr[row]], &sort_idx[mat.row_ptr[row+1]],
            [mat](csr_t::mem_index_type a, csr_t::mem_index_type b) {
            return mat.col_idx[a] < mat.col_idx[b];
        });

        // Copy over by index
        for (csr_t::mem_index_type i = mat.row_ptr[row]; i < mat.row_ptr[row+1]; ++i) {
            result.val[i] = mat.val[sort_idx[i]];
            result.col_idx[i] = mat.col_idx[sort_idx[i]];
        }
    }

    delete[] sort_idx;

    return result;
}

void for_each_group(const std::vector<gauntlet_run_params_t>& run_params,
    const function<string(const gauntlet_run_params_t&)>& group_key_func,
    const function<void(const std::vector<gauntlet_run_params_t>&)>& group_do_func) {

    multimap<string, gauntlet_run_params_t> groupStrToParam;
    for (auto& param : run_params)
        groupStrToParam.emplace(group_key_func(param), param);

    // Foreach group
    for (auto it = groupStrToParam.begin(); it != groupStrToParam.end();) {
        std::vector<gauntlet_run_params_t> group;
        auto upperBound = groupStrToParam.upper_bound(it->first);
        for (; it != upperBound; ++it)
            group.push_back(it->second);
        group_do_func(group);
        it = upperBound;
    }
}

// Do a bunch of runs with the same data set and method
void run_model_group(const std::vector<gauntlet_run_params_t>& run_params,
    gauntlet_results_t* results, const string& model_folder_path, 
    csr_t& X, csr_t& Y) {

    cout << "Loading model " << run_params[0].macro << " for "
        << run_params[0].dataset << "..." << endl;
    // cout << "-----------------------------------------------" << endl;

    HierarchicalMLModel* mc = new HierarchicalMLModel(model_folder_path,
        run_params[0].layer_type);

    for (const auto& params : run_params) {
        gauntlet_run_results_t current_results;

        cout << "Run:" << endl;
        cout << std::setw(4) << params.config;
        cout << endl << endl;

        PerformanceTimer timer;

        auto full_time_incl_overhead = std::chrono::duration<double>::zero();

        // Run prediction
        if (params.mode == "batch") {
            PROFILER_START(params.macro, params.dataset);
            timer.Begin();

            csr_t result;
            mc->predict<csr_t, csr_t>(X, result, params.beam_size,
                nullptr, params.only_top_k, params.threads, 0, &timer);
            
            timer.End();
            PROFILER_STOP;

            // Evaluate predictions
            current_results.metrics = eval_metrics(Y, result, params.only_top_k);

            FREE_MAT(result);
        }
        else if (params.mode == "realtime") {
            // Make dummy csr matrix for each row
            csr_t dummy;
            dummy.cols = X.cols;
            dummy.rows = 1;
            dummy.row_ptr = new csr_t::mem_index_type[2];
            dummy.row_ptr[0] = 0;

            // Combined result for metric evaluation
            vector<csr_t::index_type> combined_res_col_idx;
            vector<csr_t::value_type> combined_res_val;
            vector<csr_t::mem_index_type> combined_res_row_ptr;
            combined_res_row_ptr.push_back(0);

            // Query each row individually
            PROFILER_START(params.macro, params.dataset);
            timer.Begin();

            for (csr_t::index_type row = 0; row < X.rows; ++row) {
                dummy.row_ptr[1] = X.row_ptr[row+1] - X.row_ptr[row];
                dummy.col_idx = &X.col_idx[X.row_ptr[row]];
                dummy.val = &X.val[X.row_ptr[row]];

                csr_t result;

                mc->predict(dummy, result, params.beam_size,
                    nullptr, params.only_top_k, params.threads);

                    // Combine results into a large matrix
                for (csr_t::index_type i = 0; i < result.row_ptr[1]; ++i) {
                    combined_res_col_idx.push_back(result.col_idx[i]);
                    combined_res_val.push_back(result.val[i]);
                }
                combined_res_row_ptr.push_back(
                    combined_res_row_ptr[combined_res_row_ptr.size() - 1]
                    + result.row_ptr[1]);

                FREE_MAT(result);
            }
            
            timer.End();
            PROFILER_STOP;

            csr_t combined_result;
            combined_result.rows = X.rows;
            combined_result.cols = Y.cols;
            combined_result.col_idx = combined_res_col_idx.data();
            combined_result.row_ptr = combined_res_row_ptr.data();
            combined_result.val = combined_res_val.data();

            // Evaluate predictions
            current_results.metrics = eval_metrics(Y, combined_result, params.only_top_k);

            delete[] dummy.row_ptr;
        }
        else
            cout << "Mode " << params.mode << " unrecognized!" << endl;

        current_results.input_statistics = QUERY_STATS(X);
        current_results.layer_durations = timer.GetLayerDurations();
        current_results.layer_statistics = mc->get_layer_statistics();
        current_results.b_realtime = params.mode == "realtime";
        current_results.total_duration_incl_overhead = timer.GetTotalDuration();
        current_results.params = params;
        current_results.memory_stat = read_off_memory_status();

        if (current_results.b_realtime)
            current_results.total_duration = timer.GetTotalDuration();
        else
            current_results.total_duration = timer.GetCummulativeLayerDurations().back();
        
        results->runs.push_back(current_results);

        cout << endl;
    }

    delete mc;
}

void run_dataset_group(const std::vector<gauntlet_run_params_t>& run_params,
    gauntlet_results_t* results, const json& datasets_config) {

    cout << "Loading dataset " << run_params[0].dataset << "..." << endl;
    // cout << "-----------------------------------------------" << endl;

    // Load dataset for this group of runs
    string data_set_str = run_params[0].dataset;
    // cout << "-----------------------------------------------" << endl;

    auto data_set_json_it = datasets_config.find(data_set_str);

    if (data_set_json_it == datasets_config.end()) {
        cout << "Failed to locate dataset " << run_params[0].dataset << "! Will Skip." << endl;
        return;
    }

    auto data_set_json = datasets_config[data_set_str];
    // cout << "-----------------------------------------------" << endl;
    string data_set_root;
    datasets_config["root"].get_to(data_set_root);
    // cout << "-----------------------------------------------" << endl;
    string X_path;
    string Y_path;
    string model_folder_path;
    data_set_json["queries"].get_to(X_path);
    // cout << "-----------------------------------------------" << endl;
    data_set_json["truth"].get_to(Y_path);
    // cout << "-----------------------------------------------" << endl;
    data_set_json["model"].get_to(model_folder_path);
    // cout << "-----------------------------------------------" << endl;
    X_path = data_set_root + "/" + X_path;
    Y_path = data_set_root + "/" + Y_path;
    model_folder_path = data_set_root + "/" + model_folder_path;

    // cout << "-----------------------------------------------" << endl;

    // Load query data and sort it
    cout << X_path << endl;
    csr_t X = load_matrix_csr_from_npz(X_path);
    // cout << "-----------------------------------------------" << endl;
    csr_t X_sorted = sort_csr(X);
    // cout << "-----------------------------------------------" << endl;
    FREE_MAT(X);
    X = X_sorted;
    // cout << "-----------------------------------------------" << endl;

    csr_t Y = load_matrix_csr_from_npz(Y_path);
    // cout << "-----------------------------------------------" << endl;

    for_each_group(run_params, &to_model_cmp_str,
        [&X, &Y, results, &model_folder_path]
        (const vector<gauntlet_run_params_t>& params_group) {
        run_model_group(params_group, results, model_folder_path, X, Y);
    });

    FREE_MAT(X);
    FREE_MAT(Y);
}

template <typename T>
void print_statistics(const statistics_t<T>& stats) {
    cout << std::left << std::setw(10) << "Q0:" << std::setw(10) << stats.q0 << endl;
    cout << std::left << std::setw(10) << "Q1:" << std::setw(10) << stats.q1 << endl;
    cout << std::left << std::setw(10) << "Q2:" << std::setw(10) << stats.q2 << endl;
    cout << std::left << std::setw(10) << "Q3:" << std::setw(10) << stats.q3 << endl;
    cout << std::left << std::setw(10) << "Q4:" << std::setw(10) << stats.q4 << endl;
    cout << std::left << std::setw(10) << "mean:" << std::setw(10) << stats.mean << endl;
}

// Print timings as well as precision and recall
void print_result_header(const gauntlet_run_results_t& result) {
    cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << std::left << std::setw(10) << "Macro: " << std::setw(20) << result.params.macro << endl;
    cout << std::left << std::setw(10) << "Dataset: " << std::setw(20) << result.params.dataset << endl;
    cout << std::setw(4) <<result.params.config << endl;
    cout << endl;
    stringstream s_time;
    s_time << std::setprecision(4) << result.total_duration.count() << " s";
    stringstream s_time_per_query;
    s_time_per_query << std::setprecision(4) << result.total_duration.count() / (double)result.input_statistics.rows * 1000.0 << " ms";
    cout << std::left << std::setw(20) << "Total Time:" << std::setw(20) << s_time.str() << endl;
    cout << std::left << std::setw(20) << "Time Per Query:" << std::setw(20) << s_time_per_query.str() << endl;
    cout << endl;
    if (result.metrics.size() > 0) {
        cout << std::left;
        for (const auto& metric : result.metrics) {
            stringstream s;
            s << "prec@" << metric.num;
            cout << setw(10) << s.str();
        }
        cout << endl;
        cout << std::left;
        for (const auto& metric : result.metrics) {
            cout << setw(10) << setprecision(4) << metric.precision * 100.0;
        }
        cout << endl;
        cout << std::left;
        for (const auto& metric : result.metrics) {
            stringstream s;
            s << "recall@" << metric.num;
            cout << setw(10) << s.str();
        }
        cout << endl;
        cout << std::left;
        for (const auto& metric : result.metrics) {
            cout << setw(10) << setprecision(4) << metric.recall * 100.0;
        }
        cout << endl;
        cout << endl;
    }
}

void print_results(const gauntlet_results_t& results, const json& config) {
    cout << endl << endl;
    cout << "##########################################################" << endl;
    cout << " Detailed Statistics" << endl;
    cout << "##########################################################" << endl;

    for (const auto& result : results.runs) {
        print_result_header(result);
        cout << "Input Statistics:" << endl;
        cout << "=================" << endl;
        cout << std::left << setw(10) << "nnz: " << setw(20) << result.input_statistics.nnz << endl;
        cout << std::left << setw(10) << "rows: " << setw(20) << result.input_statistics.rows << endl;
        cout << std::left << setw(10) << "cols: " << setw(20) << result.input_statistics.cols << endl;
        cout << "-----------------" << endl;
        cout << "nnz per row:" << endl;
        cout << "-----------------" << endl;
        print_statistics(result.input_statistics.nnz_per_row);
        cout << endl;
        if (result.layer_durations.size() > 0) {
            cout << "Layer Statistics:" << endl;
            cout << "=================" << endl;
            cout << endl;
            for (size_t layer_i = 0; layer_i < result.layer_durations.size(); ++layer_i) {
                const auto& stats = result.layer_statistics[layer_i];
                auto time = result.layer_durations[layer_i];
                cout << "Layer " << layer_i << endl;
                cout << "=================" << endl;
                cout << std::left << setw(10) << "nnz: " << setw(20) << stats.nnz << endl;
                cout << std::left << setw(10) << "parents: " << setw(20) << stats.num_parents << endl;
                cout << std::left << setw(10) << "children: " << setw(20) << stats.num_children << endl;
                cout << "-----------------" << endl;
                cout << "nnz per col:" << endl;
                cout << "-----------------" << endl;
                print_statistics(stats.nnz_per_col);
                cout << "-----------------" << endl;

                stringstream out_time;
                out_time << std::setprecision(4) << time.count() * 1000.0 << " ms";
                stringstream out_percent;
                out_percent <<  "(" << std::setprecision(3) << time.count() / result.total_duration.count() * 100.0 << " %)";
                cout << std::left << std::setw(10) << "time: " << std::setw(20) << out_time.str() <<
                    std::setw(20) << out_percent.str() << endl;
                cout << endl;
            }
        }
        cout << endl;
        cout << "Memory Statistics:" << endl;
        cout << "=================" << endl;
        size_t page_size;
        config["pagesize"].get_to(page_size);
        print_mem_stat(result.memory_stat, page_size);
        cout << endl;
    }

    cout << endl << endl;
    cout << "##########################################################" << endl;
    cout << " Performance Results" << endl;
    cout << "##########################################################" << endl;

    for (const auto& result : results.runs) {
        print_result_header(result);
    }
}

string get_macro_result(gauntlet_run_results_t& result, const string& data_type) {
    stringstream s;
    if (data_type == "time-per-query") {
        s << setprecision(3) << result.total_duration.count() * 1000.0 / result.input_statistics.rows << " ms";
    } else if (data_type == "last-layer-percent") {
        if (result.layer_durations.size() > 0 && !result.b_realtime) {
            double total = result.total_duration.count();
            double last = result.layer_durations[result.layer_durations.size() - 1].count();
            s << setprecision(3) << last / total * 100.0 << " %";
        }
        else
            s << "N/A";
    } else if (data_type == "last-layer-time") {
        if (result.layer_durations.size() > 0 && !result.b_realtime) {
            double last = result.layer_durations[result.layer_durations.size() - 1].count();
            s << setprecision(3) << last << " s";
        }
        else
            s << "N/A";
    } else if (data_type == "not-last-layer-time") {
        if (result.layer_durations.size() > 0 && !result.b_realtime) {
            double time = 0.0;
            for (uint32_t ilayer = 0; ilayer < result.layer_durations.size() - 1; ++ilayer)
                time += result.layer_durations[ilayer].count();
            s << setprecision(3) << time << " s";
        }
        else
            s << "N/A";
    }
    else if (data_type == "total-time") {
        s << setprecision(3) << result.total_duration.count() << " s";
    }
    else if (data_type == "total-time-incl-overhead") {
         s << setprecision(3) << result.total_duration_incl_overhead.count() << " s";
    }
    else if (data_type == "query-preprocess-time") {
        s << "N/A";
    }
    else if (data_type == "query-postprocess-time") {
        s << "N/A";
    } else if (data_type.length() > 5 && data_type.substr(0, 5) == "prec@") {
        auto num = std::stoi(data_type.substr(5));
        if (num - 1 > result.metrics.size()) {
            s << "N/A";
        } else {
            s << setprecision(4) << result.metrics[num - 1].precision * 100;
        }
    } else if (data_type.length() > 7 && data_type.substr(0, 7) == "recall@") {
        auto num = std::stoi(data_type.substr(7));
        if (num - 1 > result.metrics.size()) {
            s << "N/A";
        } else {
            s << setprecision(4) << result.metrics[num - 1].recall * 100;
        }
    }
    else {
        s << "N/A";
    }
    return s.str();
}

void print_macro_table(map<pair<string, string>, gauntlet_run_results_t>& resultMap,
    const set<string>& macros,
    const set<string>& datasets,
    const string& data_type) {

    uint32_t largest_size_str = 0;
    for (auto& data : datasets)
        largest_size_str = std::max<uint32_t>(largest_size_str, data.size());

    vector<uint32_t> columnWidths;
    columnWidths.push_back(30);
    for (const auto& data : datasets)
        columnWidths.push_back(std::max<uint32_t>(10, largest_size_str));
    uint32_t totalWidth = std::accumulate(columnWidths.begin(), columnWidths.end(), 0u);
    totalWidth += 3 * datasets.size();

    if (data_type == "time-per-query")
        cout << "Time Per Query:" << endl;
    else if (data_type == "last-layer-percent")
        cout << "Percent Time in Last Layer:" << endl;
    else if (data_type == "last-layer-time")
        cout << "Time in Last Layer:" << endl;
    else if (data_type == "total-time")
        cout << "Total Runtime:" << endl;
    else if (data_type == "total-time-incl-overhead")
        cout << "Total Runtime Including Overhead:" << endl;
    else if (data_type == "not-last-layer-time")
        cout << "Time Not in Last Layer:" << endl;
    else if (data_type == "query-preprocess-time")
        cout << "Query Preprocess Time:" << endl;
    else if (data_type == "query-postprocess-time")
        cout << "Query Postprocess Time:" << endl;
    else if (data_type.length() > 5 && data_type.substr(0, 5) == "prec@")
        cout << data_type << ":" << endl;
    else if (data_type.length() > 7 && data_type.substr(0, 7) == "recall@")
        cout << data_type << ":" << endl;
    else
        cout << "[Unknown Table Type]" << endl;

    for (uint32_t i = 0; i < totalWidth; ++i)
        cout << "-";
    cout << endl;

    cout << left << setw(columnWidths[0]) << "";
    uint32_t column = 1;
    for (const auto& data : datasets) {
        cout << " | " << left << setw(columnWidths[column]) << data;
        ++column;
    }
    cout << endl;
    for (uint32_t i = 0; i < totalWidth; ++i)
        cout << "-";
    cout << endl;

    // Print table contents
    for (const auto& macro : macros) {
        uint32_t column = 0;
        cout << left << setw(columnWidths[column]) << macro;
        ++column;

        for (const auto& data : datasets) {
            auto it = resultMap.find(make_pair(macro, data));
            if (it != resultMap.end())
                cout << " | " << left << setw(columnWidths[column])
                    << get_macro_result(it->second, data_type);
            else
                cout << " | " << left << setw(columnWidths[column]) << "?";
            ++column;
        }

        cout << endl;
    }
}

void print_macro_tables(const gauntlet_results_t& results, const json& macro_tables) {

    cout << "##########################################################" << endl;
    cout << " Macro Tables" << endl;
    cout << "##########################################################" << endl;
    cout << endl;

    // Make result map
    map<pair<string, string>, gauntlet_run_results_t> resultMap;
    set<string> macros;
    set<string> datasets;
    for (auto& result : results.runs) {
        if (result.params.macro != "")
            resultMap[make_pair(result.params.macro, result.params.dataset)] = result;
        macros.emplace(result.params.macro);
        datasets.emplace(result.params.dataset);
    }

    for (auto item : macro_tables.items()) {
        string data_type = item.value();
        print_macro_table(resultMap, macros, datasets, data_type);
        cout << endl;
    }
}

void read_json(string& path, json* out) {
    cout << path << endl;
    std::ifstream f(path);
    if (!f.is_open()) {
        cout << "Failed to open " << path << "!" << endl;
        return;
    }
    f >> *out;
    f.close();
}

int main(const int argc, const char* argv[]) {
    string config_path = "gauntlet.json";
    json config;

    if (argc > 1)
        config_path = argv[1];

    // Read configuration
    read_json(config_path, &config);

    // Read macro and dataset configurations
    string macro_config_path = "macros.json";
    string datasets_config_path = "datasets.json";
    // Overwrite defaults if necessary
    if (config.contains("macros-path"))
        config["macros-path"].get_to(macro_config_path);
    if (config.contains("datasets-path"))
        config["datasets-path"].get_to(datasets_config_path);

    json macro_config;
    json datasets_config;
    read_json(macro_config_path, &macro_config);
    read_json(datasets_config_path, &datasets_config);

    // Read macros from json. Macros correspond to different choices of parameters
    auto macros = read_gaunlet_macros(macro_config);
    vector<gauntlet_run_params_t> single_runs;
    vector<gauntlet_run_params_t> multi_runs;
    if (config.contains("single-runs"))
        single_runs = read_guantlet_single_runs(config["single-runs"], macros);
    if (config.contains("multi-runs"))
        multi_runs = read_guantlet_multi_runs(config["multi-runs"], macros);

    // Combine single runs and multi runs
    vector<gauntlet_run_params_t> run_params;
    run_params.insert(run_params.end(), single_runs.begin(), single_runs.end());
    run_params.insert(run_params.end(), multi_runs.begin(), multi_runs.end());

    // Run models in groups
    gauntlet_results_t results;
    for_each_group(run_params, &to_dataset_comp_str,
        [&results, &datasets_config](const std::vector<gauntlet_run_params_t>& params) {
            run_dataset_group(params, &results, datasets_config);
    });

    cout << endl << endl << endl << endl;

    // Print performance and per run statistics
    print_results(results, config);

    cout << endl << endl << endl << endl;
    if (config.contains("macro-tables"))
        print_macro_tables(results, config["macro-tables"]);
}
