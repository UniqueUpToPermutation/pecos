# Benchmarks for Enterprise-Scale Search [Etter, Zhong, et al. '22]

Contained in this branch of my fork of the mainline PECOS repository is the complete code to reproduce all of the benmarks in our paper Enterprise-Scale Search: Accelerating Inference for Sparse Extreme Multi-Label Ranking Trees.

## Compiling Gauntlet

The benchmarks use **gauntlet**, our internal script for scheduling benchmarks that ensures models and datasets are not reloaded if they don't need to be. Gauntlet spits out extensive performance metrics and statistics. To compile it, you must have cmake installed. You can use cmake to generate makefiles:

```[bash]
cmake [PATH_TO_REPOSITORY]
```

On Linux, cmake will spit out a Makefile that you can then use to compile the benchmarking tool.

## Running and Configuring Gauntlet

Gauntlet is configured via three json files:

-------------------------

* **datasets.json**: A JSON file that lists the relative directories of all available datasets and models. The root database folder is set at the top of the file:
```[json]
"root" : "datasets"
```
datasets are then listed in dictionary format, where the key is the name of the dataset. Every corresponding value holds a path to a input file of queries to test, an input file of ground truths to compare model results, and the path to the actual model itself. For example:
```
"eurlex-4k" : {
    "queries" : "eurlex-4k/X.tst.tfidf.npz",
    "truth": "eurlex-4k/Y.tst.npz", 
    "model": "eurlex-4k/model"
},
```
We have already configured the datasets JSON, you will just have to download the datasets and models we trained [LINK TO BE PROVIDED], and place them all in a folder named ``datasets`` relative to the binary directory.

-------------------------

* **gauntlet.json**: This configures which benchmarks Gauntlet will actually run. The user may change the JSON to run different benchmarks. The main section of the file to note is the ``multi-runs`` key:

```[json]
"multi-runs": [
{
    "macros": [
        "st-batch-csc",
        "st-batch-binary-search-ch",
        "st-batch-hash-csc",
        "st-batch-hash-ch",
        "st-batch-dense-lookup-ch",
        "st-batch-dense-lookup-csc",
        "st-batch-march-csc",
        "st-batch-march-ch",
        "st-realtime-csc",
        "st-realtime-hash-csc",
        "st-realtime-binary-search-ch",
        "st-realtime-hash-ch",
        "st-realtime-march-csc",
        "st-realtime-march-ch"
    ],

    "datasets": [
        "eurlex-4k",
        "amazoncat-13k",
        "wiki10-31k"
    ]
}]
```

``multi-runs`` provides a set of macros (i.e., configurations specifying matrix type, iteration method, thread count, etc.) for PECOS, as well as a set of datasets to run those macros on. All macros will be run for each dataset. The results are displayed in a table format at the end of Gauntlet's run.

Our macros are named as follows:

```
[threading]-[inputmode]-[iterationtype]-[matrixtype]
```

Each one of these name parts has a specific meaning:
* **threading**: Can either be **st** (single-threaded) or **mt** (multi-threaded).
* **inputmode**: Can either be **batch** or **realtime**.
* **iterationtype**: Denotes the iteration type that is used for computing vector x vector and vector x chunk products. Can be empty (in which case, binary search is used), **binary-search**, **hash** (hash tables), or **march** (marcing pointers).
* **matrixtype**: Denotes the storage format used for the weight matrices. Can be **csc** (Compressed-Column Format) or **ch** (Chunked Format, ours).

To duplicate all of our benchmarks will take some time, so we have included only the three smallest datasets under the ``datasets`` entry. However, the following are all the datasets that are used in our paper:
* ``eurlex-4k``
* ``amazoncat-13k``
* ``wiki10-31k``
* ``wiki-500k``
* ``amazon-670k``
* ``amazon-3m``

-------------------------

* **macros.json**: In this JSON file you can add new macros or change existing ones. Modify this if you would like to change things like thread count etc.

