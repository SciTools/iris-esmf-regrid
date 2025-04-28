# SciTools Performance Benchmarking

SciTools uses an [Airspeed Velocity](https://github.com/airspeed-velocity/asv)
(ASV) setup to benchmark performance. This is primarily designed to check for
performance shifts between commits using statistical analysis, but can also
be easily repurposed for manual comparative and scalability analyses.

The benchmarks are run as part of the CI (the `benchmark_task` in
[`benchmark.yml`](../.github/workflows/benchmark.yml)), with any notable shifts in performance
raising a ❌ failure.

## Running benchmarks

As mentioned, benchmarks are always run on GitHub as part of the CI.
To run locally: the **benchmark runner** provides conveniences for
common benchmark setup and run tasks, including replicating the benchmarking
performed by GitHub Actions workflows. This can be accessed by:

- `benchmarks/bm_runner.py` (use the `--help` argument for details).
- Directly running `asv` commands from the `benchmarks/` directory (check
  whether environment setup has any extra dependencies - see 
  [Benchmark environments](#benchmark-environments)).

### Reducing run time

A significant portion of benchmark run time is environment management. Run-time
can be reduced by co-locating the benchmark environment and your 
[Conda package cache](https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/custom-env-and-pkg-locations.html) 
on the same [file system](https://en.wikipedia.org/wiki/File_system), if they 
are not already. This can be done in several ways:

- Temporarily reconfiguring `env_parent` in
  [`_asv_delegated_abc`](_asv_delegated_abc.py) to reference a location on the same 
  file system as the Conda package cache.
- Using an alternative Conda package cache location during the benchmark run,
  e.g. via the `$CONDA_PKGS_DIRS` environment variable.
- Moving your repo checkout to the same file system as the Conda package cache.

### Environment variables

* `DATA_GEN_PYTHON` - required - path to a Python executable that can be
used to generate benchmark test objects/files; see
[Data generation](#data-generation). The benchmark runner sets this 
automatically, but will defer to any value already set in the shell.
* `BENCHMARK_DATA` - optional - path to a directory for benchmark synthetic
test data, which the benchmark scripts will create if it doesn't already
exist. Defaults to `<root>/benchmarks/.data/` if not set. Note that some of
the generated files, especially in the 'SPerf' suite, are many GB in size so
plan accordingly.
* `ON_DEMAND_BENCHMARKS` - optional - when set (to any value): benchmarks
decorated with `@on_demand_benchmark` are included in the ASV run. Usually
coupled with the ASV `--bench` argument to only run the benchmark(s) of
interest. Is set during the benchmark runner `sperf` sub-commands.
* `ASV_COMMIT_ENVS` - optional - instruct the 
[delegated environment management](#benchmark-environments) to create a
dedicated environment for each commit being benchmarked when set (to any 
value). This means that benchmarking commits with different environment 
requirements will not be delayed by repeated environment setup - especially 
relevant given the [benchmark runner](bm_runner.py)'s use of
[--interleave-rounds](https://asv.readthedocs.io/en/stable/commands.html?highlight=interleave-rounds#asv-run),
or any time you know you will repeatedly benchmark the same commit. **NOTE:**
SciTools environments tend to large so this option can consume a lot of disk 
space.

## Writing benchmarks

[See the ASV docs](https://asv.readthedocs.io/) for full detail.

### What benchmarks to write

It is not possible to maintain a full suite of 'unit style' benchmarks:

* Benchmarks take longer to run than tests.
* Small benchmarks are more vulnerable to noise - they report a lot of false
positive regressions.

We therefore recommend writing benchmarks representing scripts or single
operations that are likely to be run at the user level.

The drawback of this approach: a reported regression is less likely to reveal
the root cause (e.g. if a commit caused a regression in coordinate-creation 
time, but the only benchmark covering this was for file-loading). Be prepared
for manual investigations; and consider committing any useful benchmarks as 
[on-demand benchmarks](#on-demand-benchmarks) for future developers to use.

### Data generation

**Important:** be sure not to use the benchmarking environment to generate any
test objects/files, as this environment changes with each commit being
benchmarked, creating inconsistent benchmark 'conditions'. The
[generate_data](./benchmarks/generate_data/__init__.py) module offers a
solution; read more detail there.

### ASV re-run behaviour

Note that ASV re-runs a benchmark multiple times between its `setup()` routine.
This is a problem for benchmarking certain SciTools operations such as data
realisation, since the data will no longer be lazy after the first run.
Consider writing extra steps to restore objects' original state _within_ the
benchmark itself.

If adding steps to the benchmark will skew the result too much then re-running
can be disabled by setting an attribute on the benchmark: `number = 1`. To
maintain result accuracy this should be accompanied by increasing the number of
repeats _between_ `setup()` calls using the `repeat` attribute.
`warmup_time = 0` is also advisable since ASV performs independent re-runs to
estimate run-time, and these will still be subject to the original problem.
The `@disable_repeat_between_setup` decorator in 
[`benchmarks/__init__.py`](benchmarks/__init__.py) offers a convenience for 
all this.

### Custom benchmarks

SciTools benchmarking implements custom benchmark types, such as a `tracemalloc`
benchmark to measure memory growth. See [custom_bms/](./custom_bms) for more
detail.

### Scaling / non-Scaling Performance Differences

**(We no longer advocate the below for benchmarks run during CI, given the
limited available runtime and risk of false-positives. It remains useful for
manual investigations).**

When comparing performance between commits/file-type/whatever it can be helpful
to know if the differences exist in scaling or non-scaling parts of the 
operation under test. This can be done using a size parameter, setting
one value to be as small as possible (e.g. a scalar value), and the other to
be significantly larger (e.g. a 1000x1000 array). Performance differences
might only be seen for the larger value, or the smaller, or both, getting you
closer to the root cause.

### On-demand benchmarks

Some benchmarks provide useful insight but are inappropriate to be included in
a benchmark run by default, e.g. those with long run-times or requiring a local
file. These benchmarks should be decorated with `@on_demand_benchmark`
(see [benchmarks init](./benchmarks/__init__.py)), which
sets the benchmark to only be included in a run when the `ON_DEMAND_BENCHMARKS`
environment variable is set. Examples include the SPerf benchmark
suite for the UK Met Office NG-VAT project.

## Benchmark environments

We have disabled ASV's standard environment management, instead using an
environment built using the same scripts that set up the package test 
environments. 
This is done using ASV's plugin architecture - see
[`asv_delegated.py`](asv_delegated.py) and associated 
references in [`asv.conf.json`](asv.conf.json) (`environment_type` and 
`plugins`).

(ASV is written to control the environment(s) that benchmarks are run in -
minimising external factors and also allowing it to compare between a matrix
of dependencies (each in a separate environment). We have chosen to sacrifice
these features in favour of testing each commit with its intended dependencies,
controlled by the test environment setup script(s)).
