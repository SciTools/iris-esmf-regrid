# iris-esmf-regrid Performance Benchmarking

iris-esmf-regrid uses an
[Airspeed Velocity](https://github.com/airspeed-velocity/asv)
(ASV) setup to benchmark performance. This is primarily designed to check for
performance shifts between commits using statistical analysis, but can also
be easily repurposed for manual comparative and scalability analyses.

The benchmarks are run as part of the CI (the `benchmark_task` in
[`.cirrus.yml`](../.cirrus.yml)), with any notable shifts in performance
raising a ❌ failure.

## Running benchmarks

`asv ...` commands must be run from this directory. You will need to have ASV
installed, as well as Nox (see
[Benchmark environments](#benchmark-environments)).

[iris-esmf-regrid's noxfile](../noxfile.py) includes a `benchmarks` session
that provides conveniences for setting up before benchmarking, and can also
replicate the CI run locally. See the session docstring for detail.

### Environment variables

* `DATA_GEN_PYTHON` - required - path to a Python executable that can be
used to generate benchmark test objects/files; see
[Data generation](#data-generation). The Nox session sets this automatically,
but will defer to any value already set in the shell.
* `BENCHMARK_DATA` - optional - path to a directory for benchmark synthetic
test data, which the benchmark scripts will create if it doesn't already
exist. Defaults to `<root>/benchmarks/.data/` if not set. Note that some of
the generated files, especially in the 'SPerf' suite, are many GB in size so
plan accordingly.
* `ON_DEMAND_BENCHMARKS` - optional - when set (to any value): benchmarks
decorated with `@on_demand_benchmark` are included in the ASV run. Usually
coupled with the ASV `--bench` argument to only run the benchmark(s) of
interest. Is set during the Nox `sperf` session.

### Reducing run time

Before benchmarks are run on a commit, the benchmark environment is
automatically aligned with the lock-file for that commit. You can significantly
speed up any environment updates by co-locating the benchmark environment and your
[Conda package cache](https://conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-package-directories-pkgs-dirs)
on the same [file system](https://en.wikipedia.org/wiki/File_system). This can
be done in several ways:

* Move your iris-esmf-regrid checkout, this being the default location for the
  benchmark environment.
* Move your package cache by editing
  [`pkgs_dirs` in Conda config](https://conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-package-directories-pkgs-dirs).
* Move the benchmark environment by **locally** editing the environment path of
  `delegated_env_commands` and `delegated_env_parent` in
  [asv.conf.json](asv.conf.json).

## Writing benchmarks

[See the ASV docs](https://asv.readthedocs.io/) for full detail.

### Data generation
**Important:** be sure not to use the benchmarking environment to generate any
test objects/files, as this environment changes with each commit being
benchmarked, creating inconsistent benchmark 'conditions'. The
[generate_data](./benchmarks/generate_data.py) module offers a
solution; read more detail there.

### ASV re-run behaviour

Note that ASV re-runs a benchmark multiple times between its `setup()` routine.
This is a problem for benchmarking certain Iris operations such as data
realisation, since the data will no longer be lazy after the first run.
Consider writing extra steps to restore objects' original state _within_ the
benchmark itself.

If adding steps to the benchmark will skew the result too much then re-running
can be disabled by setting an attribute on the benchmark: `number = 1`. To
maintain result accuracy this should be accompanied by increasing the number of
repeats _between_ `setup()` calls using the `repeat` attribute.
`warmup_time = 0` is also advisable since ASV performs independent re-runs to
estimate run-time, and these will still be subject to the original problem. A
decorator is available for this - `@disable_repeat_between_setup` in
[benchmarks init](./benchmarks/__init__.py).

### Scaling / non-Scaling Performance Differences

When comparing performance between commits/file-type/whatever it can be helpful
to know if the differences exist in scaling or non-scaling parts of the Iris
functionality in question. This can be done using a size parameter, setting
one value to be as small as possible (e.g. a scalar `Cube`), and the other to
be significantly larger (e.g. a 1000x1000 `Cube`). Performance differences
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
environment built using the same Nox scripts as Iris' test environments. This
is done using ASV's plugin architecture - see
[asv_delegated_conda.py](asv_delegated_conda.py) and the extra config items in
[asv.conf.json](asv.conf.json).

(ASV is written to control the environment(s) that benchmarks are run in -
minimising external factors and also allowing it to compare between a matrix
of dependencies (each in a separate environment). We have chosen to sacrifice
these features in favour of testing each commit with its intended dependencies,
controlled by Nox + lock-files).
