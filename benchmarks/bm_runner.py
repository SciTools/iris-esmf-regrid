"""Argparse conveniences for executing common types of benchmark runs."""

from abc import ABC, abstractmethod
import argparse
from argparse import ArgumentParser
from datetime import datetime
from importlib import import_module
from os import environ
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import NamedTemporaryFile

# The threshold beyond which shifts are 'notable'. See `asv compare`` docs
#  for more.
COMPARE_FACTOR = 1.2

BENCHMARKS_DIR = Path(__file__).parent
ROOT_DIR = BENCHMARKS_DIR.parent
# Storage location for reports used in GitHub actions.
GH_REPORT_DIR = ROOT_DIR.joinpath(".github", "workflows", "benchmark_reports")

# Common ASV arguments for all run_types except `custom`.
ASV_HARNESS = "run {posargs} --attribute rounds=4 --interleave-rounds --show-stderr"


def _echo(echo_string: str):
    # Use subprocess for printing to reduce chance of printing out of sequence
    #  with the subsequent calls.
    subprocess.run(["echo", f"BM_RUNNER DEBUG: {echo_string}"], check=False)


def _subprocess_runner(args, asv=False, **kwargs):
    # Avoid permanent modifications if the same arguments are used more than once.
    args = args.copy()
    kwargs = kwargs.copy()
    if asv:
        args.insert(0, "asv")
        kwargs["cwd"] = BENCHMARKS_DIR
    _echo(" ".join(args))
    kwargs.setdefault("check", True)
    return subprocess.run(args, **kwargs)


def _subprocess_runner_capture(args, **kwargs) -> str:
    result = _subprocess_runner(args, capture_output=True, **kwargs)
    return result.stdout.decode().rstrip()


def _check_requirements(package: str) -> None:
    try:
        import_module(package)
    except ImportError as exc:
        message = (
            f"No {package} install detected. Benchmarks can only "
            f"be run in an environment including {package}."
        )
        raise Exception(message) from exc


def _prep_data_gen_env() -> None:
    """Create or access a separate, unchanging environment for generating test data."""
    python_version = "3.12"
    data_gen_var = "DATA_GEN_PYTHON"
    if data_gen_var in environ:
        _echo("Using existing data generation environment.")
    else:
        _echo("Setting up the data generation environment ...")
        # Get Nox to build an environment for the `tests` session, but don't
        #  run the session. Will reuse a cached environment if appropriate.
        _subprocess_runner(
            [
                "nox",
                f"--noxfile={ROOT_DIR / 'noxfile.py'}",
                "--session=tests",
                "--install-only",
                f"--python={python_version}",
            ]
        )
        # Find the environment built above, set it to be the data generation
        #  environment.
        data_gen_python = next(
            (ROOT_DIR / ".nox").rglob(f"tests*/bin/python{python_version}")
        ).resolve()
        environ[data_gen_var] = str(data_gen_python)

        _echo("Data generation environment ready.")


def _setup_common() -> None:
    _check_requirements("asv")
    _check_requirements("nox")

    _prep_data_gen_env()

    _echo("Setting up ASV ...")
    _subprocess_runner(["machine", "--yes"], asv=True)

    _echo("Setup complete.")


def _asv_compare(*commits: str, fail: bool = False) -> None:
    """Run through a list of commits comparing each one to the next."""
    _commits = [commit[:8] for commit in commits]
    for i in range(len(_commits) - 1):
        before = _commits[i]
        after = _commits[i + 1]
        asv_command = shlex.split(
            f"compare {before} {after} --factor={COMPARE_FACTOR} --split"
        )

        comparison = _subprocess_runner_capture(asv_command, asv=True)
        _echo(comparison)
        shifts = _subprocess_runner_capture([*asv_command, "--only-changed"], asv=True)
        _echo(shifts)
        if fail and shifts:
            message = (
                f"Performance shifts detected between commits {before} and {after}.\n"
            )
            raise RuntimeError(message)


class _SubParserGenerator(ABC):
    """Convenience for holding all the necessary argparse info in 1 place."""

    name: str = NotImplemented
    description: str = NotImplemented
    epilog: str = NotImplemented

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        self.subparser: ArgumentParser = subparsers.add_parser(
            self.name,
            description=self.description,
            epilog=self.epilog,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.add_arguments()
        self.add_asv_arguments()
        self.subparser.set_defaults(func=self.func)

    @abstractmethod
    def add_arguments(self) -> None:
        """All custom self.subparser.add_argument() calls."""
        _ = NotImplemented

    def add_asv_arguments(self) -> None:
        self.subparser.add_argument(
            "asv_args",
            nargs=argparse.REMAINDER,
            help="Any number of arguments to pass down to the ASV benchmark command.",
        )

    @staticmethod
    @abstractmethod
    def func(args: argparse.Namespace):
        """Return when the subparser is parsed.

        `func` is then called, performing the user's selected sub-command.

        """
        _ = args
        return NotImplemented


class Branch(_SubParserGenerator):
    """Class for parsing and running the 'branch' argument."""

    name = "branch"
    description = (
        "Benchmarks the two commits,``HEAD``, and ``HEAD``'s merge-base with the "
        "input **base_branch**. If running on GitHub Actions: HEAD will be "
        "GitHub's merge commit and merge-base will be the merge target. Performance "
        "comparisons will be posted in the CI run which will fail if regressions "
        "exceed the tolerance.\n"
        "Uses `asv run`."
    )
    epilog = (
        "e.g. python bm_runner.py branch upstream/main\n"
        "e.g. python bm_runner.py branch upstream/main --bench=regridding"
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "base_branch",
            type=str,
            help="A branch that has the merge-base with ``HEAD`` - ``HEAD`` will be benchmarked against that merge-base.",
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()

        git_command = shlex.split("git rev-parse HEAD")
        head_sha = _subprocess_runner_capture(git_command)[:8]

        git_command = shlex.split(f"git merge-base {head_sha} {args.base_branch}")
        merge_base = _subprocess_runner_capture(git_command)[:8]

        with NamedTemporaryFile("w") as hashfile:
            hashfile.writelines([merge_base, "\n", head_sha])
            hashfile.flush()
            commit_range = f"HASHFILE:{hashfile.name}"
            asv_command = shlex.split(ASV_HARNESS.format(posargs=commit_range))
            _subprocess_runner([*asv_command, *args.asv_args], asv=True)

        _asv_compare(merge_base, head_sha, fail=True)


class SPerf(_SubParserGenerator):
    """Class for parsing and running the 'sperf' argument."""

    name = "sperf"
    description = (
        "Run the on-demand Sperf suite of benchmarks (measuring "
        "scalability) for the ``HEAD`` of ``upstream/main`` only, "
        "and publish the results to the input **publish_dir**, within a "
        "unique subdirectory for this run.\n"
        "Uses `asv run`."
    )
    epilog = (
        "e.g. python bm_runner.py sperf my_publish_dir\n"
        "e.g. python bm_runner.py sperf my_publish_dir --bench=regridding"
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "publish_dir",
            type=str,
            help="HTML results will be published to a sub-dir in this dir.",
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()

        publish_dir = Path(args.publish_dir)
        if not publish_dir.is_dir():
            message = f"Input 'publish directory' is not a directory: {publish_dir}"
            raise NotADirectoryError(message)
        publish_subdir = (
            publish_dir / f"sperf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        publish_subdir.mkdir()

        # Activate on demand benchmarks (C/SPerf are deactivated for
        #  'standard' runs).
        environ["ON_DEMAND_BENCHMARKS"] = "True"
        commit_range = "upstream/main^!"

        asv_command = (
            ASV_HARNESS.format(posargs=commit_range) + " --bench=.*Scalability.*"
        )

        # Only do a single round.
        _asv_command = shlex.split(re.sub(r"rounds=\d", "rounds=1", asv_command))
        try:
            _subprocess_runner([*_asv_command, *args.asv_args], asv=True)
        except subprocess.CalledProcessError as err:
            # C/SPerf benchmarks are much bigger than the CI ones:
            # Don't fail the whole run if memory blows on 1 benchmark.
            # ASV produces return code of 2 if the run includes crashes.
            if err.returncode != 2:
                raise

        _asv_command = shlex.split(
            f"publish {commit_range} --html-dir={publish_subdir}"
        )
        _subprocess_runner(_asv_command, asv=True)

        # Print completion message.
        location = BENCHMARKS_DIR / ".asv"
        _echo(
            f'New ASV results for "sperf".\n'
            f'See "{publish_subdir}",'
            f'\n  or JSON files under "{location / "results"}".'
        )


class Custom(_SubParserGenerator):
    """Class for parsing and running the 'custom' argument."""

    name = "custom"
    description = (
        "Run ASV with the input **ASV sub-command**, without any preset "
        "arguments - must all be supplied by the user. So just like running "
        "ASV manually, with the convenience of reusing the runner's "
        "scripted setup steps."
    )
    epilog = "e.g. python bm_runner.py custom continuous a1b23d4 HEAD --quick"

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "asv_sub_command",
            type=str,
            help="The ASV command to run.",
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()
        _subprocess_runner([args.asv_sub_command, *args.asv_args], asv=True)


def main():
    parser = ArgumentParser(
        description="Run the Iris performance benchmarks (using Airspeed Velocity).",
        epilog="More help is available within each sub-command.",
    )
    subparsers = parser.add_subparsers(required=True)

    for gen in (Branch, SPerf, Custom):
        _ = gen(subparsers).subparser

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
