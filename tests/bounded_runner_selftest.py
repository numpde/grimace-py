from __future__ import annotations

import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_bounded_tests.py"
SPEC = importlib.util.spec_from_file_location("bounded_test_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def invoke(
    command: list[str],
    *,
    wall: float = 5,
    rss: int = 512,
    cpu: int = 5,
    address_space: int = 1024,
    lock_wait: float = 300,
    cleanup: float = 5,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--wall-seconds",
            str(wall),
            "--rss-mib",
            str(rss),
            "--cpu-seconds",
            str(cpu),
            "--address-space-mib",
            str(address_space),
            "--lock-wait-seconds",
            str(lock_wait),
            "--cleanup-seconds",
            str(cleanup),
            "--",
            *command,
        ],
        text=True,
        capture_output=True,
        timeout=12,
        check=False,
    )


def report(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    lines = [line for line in result.stderr.splitlines() if line.startswith(runner.REPORT_PREFIX)]
    if not lines:
        raise AssertionError(f"runner emitted no terminal report: {result.stderr}")
    return json.loads(lines[-1][len(runner.REPORT_PREFIX) :])


def pid_is_live(pid: int) -> bool:
    try:
        stat_fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return stat_fields[0] not in ("Z", "X")


class BoundedRunnerTests(unittest.TestCase):
    def test_success_and_failure_keep_command_status(self) -> None:
        success = invoke([sys.executable, "-c", "print('ok')"])
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(report(success)["cleanup"], "verified_no_live_members")

        failure = invoke([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(failure.returncode, 7, failure.stderr)
        self.assertEqual(report(failure)["reason"], "command_exit")

    def test_wall_and_sampled_rss_limits_clean_the_group(self) -> None:
        timed = invoke([sys.executable, "-c", "import time; time.sleep(2)"], wall=0.3)
        self.assertEqual(timed.returncode, 124, timed.stderr)
        self.assertTrue(report(timed)["reason"].startswith("wall_time_limit"))
        self.assertEqual(report(timed)["cleanup"], "verified_no_live_members")

        allocation = (
            "import time; data = bytearray(128 * 1024 * 1024); "
            "[data.__setitem__(i, 1) for i in range(0, len(data), 4096)]; time.sleep(2)"
        )
        limited = invoke(
            [sys.executable, "-c", allocation], wall=5, rss=40, address_space=512
        )
        self.assertEqual(limited.returncode, 125, limited.stderr)
        self.assertEqual(report(limited)["reason"], "aggregate_rss_limit")
        self.assertEqual(report(limited)["cleanup"], "verified_no_live_members")
        self.assertIn("not a hard kernel cap", report(limited)["rss_limit_kind"])

    def test_cpu_and_address_space_limits_reach_child(self) -> None:
        cpu_limited = invoke(
            [sys.executable, "-c", "import signal; signal.alarm(5); exec('while True: pass')"],
            wall=8,
            cpu=1,
            address_space=512,
        )
        self.assertEqual(cpu_limited.returncode, 128 + signal.SIGXCPU, cpu_limited.stderr)
        self.assertEqual(report(cpu_limited)["reason"], "cpu_limit")
        self.assertEqual(report(cpu_limited)["command_exit_status"], -signal.SIGXCPU)

        address_limited = invoke(
            [
                sys.executable,
                "-c",
                "data = bytearray(512 * 1024 * 1024); print(len(data))",
            ],
            wall=5,
            rss=512,
            address_space=256,
        )
        self.assertNotEqual(address_limited.returncode, 0)
        self.assertEqual(report(address_limited)["reason"], "command_exit")

    def test_normal_parent_exit_cleans_a_surviving_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "child.pid"
            ready_path = Path(directory) / "child.ready"
            adopted_parent_path = Path(directory) / "adopted.parent"
            expected_parent_path = Path(directory) / "adopted.parent.expected"
            child_command = (
                "import os,signal,sys,time; parent_pid=int(sys.argv[1]); "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); os.setsid(); "
                f"open({str(ready_path)!r},'w').write('ready')\n"
                "deadline=time.monotonic()+2\n"
                "while os.getppid()==parent_pid and time.monotonic()<deadline:\n"
                "    time.sleep(.005)\n"
                f"open({str(adopted_parent_path)!r},'w').write(str(os.getppid()))\n"
                "signal.alarm(8); time.sleep(60)"
            )
            command = (
                "import os,signal,subprocess,sys,time; signal.alarm(8); "
                "runner_pid=os.getppid(); parent_pid=os.getpid(); "
                f"child_command={child_command!r}; "
                "p=subprocess.Popen([sys.executable,'-c',child_command,str(parent_pid)]); "
                "deadline=time.monotonic()+2\n"
                f"while not os.path.exists({str(ready_path)!r}) and time.monotonic()<deadline:\n"
                "    time.sleep(.005)\n"
                f"open({str(pid_path)!r},'w').write(str(p.pid))\n"
                f"open({str(expected_parent_path)!r},'w').write(str(runner_pid))"
            )
            result = invoke([sys.executable, "-c", command])
            self.assertEqual(result.returncode, 125, result.stderr)
            self.assertEqual(
                report(result)["reason"], "command_exit_with_surviving_children"
            )
            self.assertFalse(pid_is_live(int(pid_path.read_text())))
            self.assertEqual(
                int(adopted_parent_path.read_text()),
                int(expected_parent_path.read_text()),
            )

    def test_runner_signals_clean_descendants(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory() as directory:
                pid_path = Path(directory) / "child.pid"
                command = (
                    "import signal,subprocess,sys,time; signal.alarm(8); "
                    "p=subprocess.Popen([sys.executable,'-c',"
                    "'import signal,time;signal.alarm(8);time.sleep(60)']); "
                    f"open({str(pid_path)!r},'w').write(str(p.pid)); time.sleep(60)"
                )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(RUNNER_PATH),
                        "--wall-seconds",
                        "30",
                        "--rss-mib",
                        "512",
                        "--cpu-seconds",
                        "30",
                        "--address-space-mib",
                        "1024",
                        "--",
                        sys.executable,
                        "-c",
                        command,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                deadline = time.monotonic() + 5
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                if not pid_path.exists():
                    process.send_signal(signal.SIGTERM)
                    process.communicate(timeout=5)
                    self.fail("child process did not start before its bounded wait")
                child_pid = int(pid_path.read_text())
                process.send_signal(sig)
                try:
                    _stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.send_signal(signal.SIGTERM)
                    _stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 128 + sig, stderr)
                self.assertFalse(pid_is_live(child_pid))
                payload = report(subprocess.CompletedProcess([], process.returncode, "", stderr))
                self.assertEqual(payload["cleanup"], "verified_no_live_members")

    def test_lock_wait_is_bounded(self) -> None:
        lock_fd = os.open(runner.LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        self.fail("shared runner lock stayed busy during bounded contention test")
                    time.sleep(0.05)
            result = invoke(
                [sys.executable, "-c", "raise SystemExit(0)"], lock_wait=0.15
            )
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertIn("lock_wait_limit", report(result)["reason"])
        self.assertEqual(report(result)["cleanup"], "no_command_started")

    def test_overall_wall_expiry_before_spawn_is_not_lock_contention(self) -> None:
        from unittest.mock import patch

        args = runner.parse_args(
            [
                "--wall-seconds", "0.000000001", "--rss-mib", "128",
                "--cpu-seconds", "1", "--address-space-mib", "128",
                "--", sys.executable, "-c", "pass",
            ]
        )
        with patch("sys.stderr", new=io.StringIO()) as stderr:
            with patch.object(runner.subprocess, "Popen") as popen:
                self.assertEqual(runner.run_command(args), 125)
                popen.assert_not_called()
        lines = [
            line[len(runner.REPORT_PREFIX) :]
            for line in stderr.getvalue().splitlines()
            if line.startswith(runner.REPORT_PREFIX)
        ]
        payload = json.loads(lines[-1])
        self.assertIn("overall_wall_budget_expired", payload["reason"])
        self.assertNotIn("lock_wait_limit", payload["reason"])
        self.assertEqual(payload["cleanup"], "no_command_started")

    def test_accounting_preflight_fails_closed_before_spawn(self) -> None:
        from unittest.mock import patch

        args = runner.parse_args(
            [
                "--wall-seconds",
                "1",
                "--rss-mib",
                "128",
                "--cpu-seconds",
                "1",
                "--address-space-mib",
                "128",
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )
        with patch.object(runner, "_process_snapshot", side_effect=runner.AccountingError("test")):
            with patch.object(runner.subprocess, "Popen") as popen:
                self.assertEqual(runner.run_command(args), 125)
                popen.assert_not_called()

    def test_unrelated_namespace_is_ignored_but_owned_namespace_is_rejected(self) -> None:
        main_namespace = (1, 1)
        foreign_namespace = (2, 2)
        runner_pid = 4_000_000_000
        root_pid = runner_pid + 1
        unrelated_pid = runner_pid + 2

        def process(
            pid: int,
            parent: int,
            process_group: int,
            session: int,
            namespace: tuple[int, int],
        ) -> runner.Process:
            return runner.Process(
                pid,
                parent,
                process_group,
                session,
                1,
                "S",
                0,
                namespace,
            )

        snapshot = runner.ProcessTable()
        snapshot[runner_pid] = process(runner_pid, 1, runner_pid, runner_pid, main_namespace)
        snapshot[root_pid] = process(root_pid, runner_pid, root_pid, root_pid, main_namespace)
        snapshot[unrelated_pid] = process(unrelated_pid, 1, root_pid, root_pid, foreign_namespace)
        runner._validate_runner_entry(snapshot, runner_pid, 1, main_namespace)
        tracked = runner._descendants(snapshot, root_pid, runner_pid)
        runner._validate_tracked_namespaces(tracked, main_namespace)
        self.assertNotIn(unrelated_pid, tracked)

        owned_pid = runner_pid + 3
        snapshot[owned_pid] = process(owned_pid, root_pid, root_pid, root_pid, foreign_namespace)
        tracked = runner._descendants(snapshot, root_pid, runner_pid)
        with self.assertRaisesRegex(runner.AccountingError, "outside the runner's PID namespace"):
            runner._validate_tracked_namespaces(tracked, main_namespace)

        with self.assertRaisesRegex(runner.AccountingError, "does not identify the runner"):
            runner._validate_runner_entry(runner.ProcessTable(), runner_pid, 1, main_namespace)
        wrong_runner = runner.ProcessTable()
        wrong_runner[runner_pid] = process(
            runner_pid, 1, runner_pid, runner_pid, foreign_namespace
        )
        with self.assertRaisesRegex(runner.AccountingError, "does not identify the runner"):
            runner._validate_runner_entry(wrong_runner, runner_pid, 1, main_namespace)

    def test_new_child_missing_from_proc_enumeration_is_reconciled(self) -> None:
        from unittest.mock import patch

        namespace = (1, 1)
        runner_pid = 4_000_000_000
        root_pid = runner_pid + 1
        child_pid = runner_pid + 2

        def process(pid: int, parent: int, group: int, session: int) -> runner.Process:
            return runner.Process(pid, parent, group, session, pid, "S", 17, namespace)

        snapshot = runner.ProcessTable()
        snapshot[runner_pid] = process(runner_pid, 1, runner_pid, runner_pid)
        snapshot[root_pid] = process(root_pid, runner_pid, root_pid, root_pid)
        snapshot[child_pid] = process(child_pid, 1, 1, 1)
        tracked = runner._descendants(snapshot, root_pid, runner_pid)
        observed = {runner_pid: 0, root_pid: 0}

        def children(parent_pid: int, _deadline: float | None = None) -> set[int]:
            observed[parent_pid] = observed.get(parent_pid, 0) + 1
            return {root_pid} if parent_pid == runner_pid else {child_pid}

        original_read_stat = runner._read_stat

        def read_stat(path: Path) -> tuple[int, int, int, int, str, int]:
            if path == Path("/proc") / str(root_pid) / "stat":
                return runner_pid, root_pid, root_pid, root_pid, "S", 0
            return original_read_stat(path)

        fresh_child = process(child_pid, root_pid, root_pid, root_pid)

        def identity(pid: int) -> runner.Process:
            return fresh_child if pid == child_pid else snapshot[pid]

        with patch.object(runner, "_task_children", side_effect=children):
            with patch.object(runner, "_read_stat", side_effect=read_stat):
                with patch.object(runner, "_process_identity", side_effect=identity):
                    reconciled = runner._validate_process_tree(
                        snapshot,
                        tracked,
                        runner_pid,
                        namespace,
                        root_pid,
                        time.monotonic() + 2,
                    )
        self.assertIn(child_pid, reconciled)
        self.assertEqual(reconciled[child_pid].start_time, child_pid)
        self.assertEqual(reconciled[child_pid].rss_bytes, 17)

    def test_reused_child_pid_is_not_adopted_from_stale_membership(self) -> None:
        from unittest.mock import patch

        namespace = (1, 1)
        runner_pid = 4_000_000_000
        root_pid = runner_pid + 1
        child_pid = runner_pid + 2
        snapshot = runner.ProcessTable()
        snapshot[runner_pid] = runner.Process(
            runner_pid, 1, runner_pid, runner_pid, 1, "S", 0, namespace
        )
        snapshot[root_pid] = runner.Process(
            root_pid, runner_pid, root_pid, root_pid, 2, "S", 0, namespace
        )
        snapshot[child_pid] = runner.Process(child_pid, 1, 1, 1, 3, "S", 0, namespace)
        tracked = runner._descendants(snapshot, root_pid, runner_pid)
        unrelated = runner.Process(child_pid, 1, 1, 1, 3, "S", 0, namespace)

        def children(parent_pid: int, _deadline: float | None = None) -> set[int]:
            return {root_pid} if parent_pid == runner_pid else {child_pid}

        original_read_stat = runner._read_stat

        def read_stat(path: Path) -> tuple[int, int, int, int, str, int]:
            if path == Path("/proc") / str(root_pid) / "stat":
                return runner_pid, root_pid, root_pid, 2, "S", 0
            return original_read_stat(path)

        def identity(pid: int) -> runner.Process:
            return unrelated if pid == child_pid else snapshot[pid]

        with patch.object(runner, "_task_children", side_effect=children):
            with patch.object(runner, "_read_stat", side_effect=read_stat):
                with patch.object(runner, "_process_identity", side_effect=identity):
                    with self.assertRaisesRegex(runner.AccountingError, "cannot prove ownership"):
                        runner._validate_process_tree(
                            snapshot,
                            tracked,
                            runner_pid,
                            namespace,
                            root_pid,
                            time.monotonic() + 2,
                        )

    def test_child_forked_after_proc_enumeration_is_reconciled(self) -> None:
        from unittest.mock import patch

        namespace = runner._namespace_identity(Path("/proc/self/ns/pid"))
        original_iterdir = Path.iterdir
        child: subprocess.Popen[bytes] | None = None

        def enumerate_then_fork(path: Path):
            nonlocal child
            entries = tuple(original_iterdir(path))
            if path == Path("/proc") and child is None:
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(5)"]
                )
            return iter(entries)

        try:
            with patch.object(Path, "iterdir", enumerate_then_fork):
                snapshot = runner._process_snapshot(namespace)
            assert child is not None
            self.assertNotIn(child.pid, snapshot)
            tracked = runner._descendants(snapshot, child.pid, os.getpid())
            reconciled = runner._validate_process_tree(
                snapshot,
                tracked,
                os.getpid(),
                namespace,
                child.pid,
                time.monotonic() + 2,
            )
            self.assertIn(child.pid, reconciled)
            self.assertEqual(reconciled[child.pid].start_time, runner._process_identity(child.pid).start_time)
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
            if child is not None:
                child.wait(timeout=2)

    def test_child_exiting_during_reconciliation_is_confirmed_absent(self) -> None:
        from unittest.mock import patch

        namespace = (1, 1)
        runner_pid = 4_000_000_000
        root_pid = runner_pid + 1
        child_pid = runner_pid + 2
        snapshot = runner.ProcessTable()
        snapshot[runner_pid] = runner.Process(
            runner_pid, 1, runner_pid, runner_pid, 1, "S", 0, namespace
        )
        snapshot[root_pid] = runner.Process(
            root_pid, runner_pid, root_pid, root_pid, 2, "S", 0, namespace
        )
        tracked = runner._descendants(snapshot, root_pid, runner_pid)
        child_reads = 0

        def children(parent_pid: int, _deadline: float | None = None) -> set[int]:
            if parent_pid == runner_pid:
                return {root_pid}
            nonlocal child_reads
            child_reads += 1
            return {child_pid} if child_reads == 1 else set()

        original_read_stat = runner._read_stat

        def read_stat(path: Path) -> tuple[int, int, int, int, str, int]:
            if path == Path("/proc") / str(root_pid) / "stat":
                return runner_pid, root_pid, root_pid, 2, "S", 0
            return original_read_stat(path)

        def identity(pid: int) -> runner.Process:
            if pid == child_pid:
                raise FileNotFoundError
            return snapshot[pid]

        with patch.object(runner, "_task_children", side_effect=children):
            with patch.object(runner, "_read_stat", side_effect=read_stat):
                with patch.object(runner, "_process_identity", side_effect=identity):
                    reconciled = runner._validate_process_tree(
                        snapshot,
                        tracked,
                        runner_pid,
                        namespace,
                        root_pid,
                        time.monotonic() + 2,
                    )
        self.assertNotIn(child_pid, reconciled)
        self.assertGreaterEqual(child_reads, 2)

    def test_nonfinite_budget_is_rejected_before_command_start(self) -> None:
        with self.assertRaises(SystemExit) as error:
            runner.parse_args(
                [
                    "--wall-seconds",
                    "nan",
                    "--rss-mib",
                    "128",
                    "--cpu-seconds",
                    "1",
                    "--address-space-mib",
                    "128",
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                ]
            )
        self.assertEqual(error.exception.code, 2)

    def test_accounting_loss_after_spawn_terminates_before_return(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "child.pid"
            command = (
                "import os,signal,time; signal.alarm(8); "
                f"open({str(pid_path)!r},'w').write(str(os.getpid())); time.sleep(60)"
            )
            args = runner.parse_args(
                [
                    "--wall-seconds",
                    "10",
                    "--rss-mib",
                    "512",
                    "--cpu-seconds",
                    "10",
                    "--address-space-mib",
                    "1024",
                    "--",
                    sys.executable,
                    "-c",
                    command,
                ]
            )
            read_processes = runner._process_snapshot
            calls = 0

            def lose_one_snapshot(
                namespace: tuple[int, int], deadline: float | None = None
            ) -> dict[int, runner.Process]:
                nonlocal calls
                calls += 1
                if calls == 2:
                    deadline = time.monotonic() + 2
                    while not pid_path.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(pid_path.exists(), "child did not reach its marker")
                    raise runner.AccountingError("simulated transient accounting failure")
                return read_processes(namespace, deadline)

            with patch.object(runner, "_process_snapshot", side_effect=lose_one_snapshot):
                self.assertEqual(runner.run_command(args), 125)
            self.assertGreaterEqual(calls, 3)
            self.assertFalse(pid_is_live(int(pid_path.read_text())))

    def test_persistent_accounting_loss_reports_bounded_unverified_cleanup(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "child.pid"
            command = (
                "import os,signal,time; signal.alarm(8); "
                f"open({str(pid_path)!r},'w').write(str(os.getpid())); time.sleep(60)"
            )
            args = runner.parse_args(
                [
                    "--wall-seconds", "10", "--rss-mib", "512",
                    "--cpu-seconds", "10", "--address-space-mib", "1024",
                    "--cleanup-seconds", "0.25", "--", sys.executable, "-c", command,
                ]
            )
            original_snapshot = runner._process_snapshot
            calls = 0

            def lose_accounting(
                namespace: tuple[int, int], deadline: float | None = None
            ) -> dict[int, runner.Process]:
                nonlocal calls
                calls += 1
                if calls >= 2:
                    deadline = time.monotonic() + 2
                    while not pid_path.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(pid_path.exists(), "child did not reach its marker")
                    raise runner.AccountingError("persistent simulated accounting loss")
                return original_snapshot(namespace, deadline)

            started = time.monotonic()
            with patch.object(runner, "_process_snapshot", side_effect=lose_accounting):
                with patch("sys.stderr", new=io.StringIO()) as stderr:
                    status = runner.run_command(args)
            elapsed = time.monotonic() - started
            self.assertEqual(status, 125)
            self.assertGreaterEqual(elapsed, 0.2)
            self.assertLess(elapsed, 2)
            self.assertFalse(pid_is_live(int(pid_path.read_text())))
            payload = json.loads(
                next(
                    line[len(runner.REPORT_PREFIX) :]
                    for line in reversed(stderr.getvalue().splitlines())
                    if line.startswith(runner.REPORT_PREFIX)
                )
            )
            self.assertEqual(payload["cleanup"].split(";")[0], "cleanup deadline expired")
            self.assertIn("cleanup_unverified", payload["cleanup"])
            self.assertIn("accounting_failure=", payload["cleanup"])
            self.assertIn("known_group_signal_attempted=true", payload["cleanup"])

    def test_surviving_killed_state_expires_cleanup_deadline(self) -> None:
        from unittest.mock import patch

        namespace = runner._namespace_identity(Path("/proc/self/ns/pid"))
        runner_pid = os.getpid()
        root_pid = 4_000_000_000
        snapshot = runner._process_snapshot(namespace)
        snapshot[root_pid] = runner.Process(
            root_pid, runner_pid, root_pid, root_pid, 1, "D", 0, namespace
        )
        started = time.monotonic()
        with patch.object(runner, "_process_snapshot", return_value=snapshot):
            with patch.object(runner, "_signal_tracked"):
                with patch.object(runner, "_signal_known_group", return_value=False):
                    result = runner._terminate_and_verify(
                        root_pid,
                        namespace,
                        {root_pid: snapshot[root_pid]},
                        0.15,
                    )
        elapsed = time.monotonic() - started
        self.assertFalse(result.verified)
        self.assertIn("cleanup_unverified", result.detail)
        self.assertLess(elapsed, 1)


if __name__ == "__main__":
    unittest.main()
