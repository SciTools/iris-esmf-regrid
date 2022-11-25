"""Benchmark tests for iris-esmf-regrid"""


from os import environ


def disable_repeat_between_setup(benchmark_object):
    """
    Decorator for benchmarks where object persistence would be inappropriate.

    E.g:
        * Data is realised during testing.

    Can be applied to benchmark classes/methods/functions.

    https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks

    """
    # Prevent repeat runs between setup() runs - object(s) will persist after 1st.
    benchmark_object.number = 1
    # Compensate for reduced certainty by increasing number of repeats.
    #  (setup() is run between each repeat).
    #  Minimum 5 repeats, run up to 30 repeats / 20 secs whichever comes first.
    benchmark_object.repeat = (5, 30, 20.0)
    # ASV uses warmup to estimate benchmark time before planning the real run.
    #  Prevent this, since object(s) will persist after first warmup run,
    #  which would give ASV misleading info (warmups ignore ``number``).
    benchmark_object.warmup_time = 0.0

    return benchmark_object


def skip_benchmark(benchmark_object):
    """
    Decorator for benchmarks skipping benchmarks.

    Simply doesn't return the object.

    Warnings
    --------
    ASV's architecture means decorated classes cannot be sub-classed. Code for
    inheritance should be in a mixin class that doesn't include any methods
    which ASV will recognise as benchmarks
    (e.g. ``def time_something(self):`` ).

    """
    pass


def on_demand_benchmark(benchmark_object):
    """
    Decorator. Disables these benchmark(s) unless ON_DEMAND_BENCHARKS env var is set.

    For benchmarks that, for whatever reason, should not be run by default.
    E.g:
        * Require a local file
        * Used for scalability analysis instead of commit monitoring.

    Can be applied to benchmark classes/methods/functions.

    Warnings
    --------
    ASV's architecture means decorated classes cannot be sub-classed. Code for
    inheritance should be in a mixin class that doesn't include any methods
    which ASV will recognise as benchmarks
    (e.g. ``def time_something(self):`` ).

    """
    if "ON_DEMAND_BENCHMARKS" in environ:
        return benchmark_object
