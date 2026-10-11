#!/usr/bin/env python3
"""Run one command under serialized, observable Linux process-tree budgets.

RSS is sampled as the sum of visible `/proc` member RSS values; shared pages
may be counted more than once. This is a detection limit, not a kernel memory
ceiling. A runner killed with SIGKILL cannot perform process-tree cleanup.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass

POLL_SECONDS = 0.05
LOCK_POLL_SECONDS = 0.05
LOCK_FILE = Path("/tmp") / f"grimace-bounded-tests-{os.getuid()}.lock"
REPORT_PREFIX = "bounded-test-runner: "


class AccountingError(RuntimeError):
    pass


class WallBudgetExpired(TimeoutError):
    pass


class LockWaitExpired(TimeoutError):
    pass


class TreeDeadlineExpired(AccountingError):
    pass


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    process_group: int
    session: int
    start_time: int
    state: str
    rss_bytes: int
    namespace: tuple[int, int] | None

    @property
    def live(self) -> bool:
        return self.state not in ("Z", "X")


class ProcessTable(dict[int, Process]):
    def __init__(self) -> None:
        super().__init__()
        self.unreadable_pids: set[int] = set()


@dataclass(frozen=True)
class CleanupResult:
    verified: bool
    detail: str


def _validate_runner_entry(
    snapshot: ProcessTable,
    runner_pid: int,
    runner_start_time: int,
    namespace: tuple[int, int],
) -> None:
    runner = snapshot.get(runner_pid)
    if (
        runner is None
        or runner.start_time != runner_start_time
        or runner.namespace != namespace
    ):
        raise AccountingError(
            "the /proc PID view does not identify the runner process consistently"
        )


def _namespace_identity(path: Path) -> tuple[int, int]:
    try:
        info = path.stat()
    except OSError as exc:
        raise AccountingError(f"cannot verify PID namespace for {path}: {exc}") from exc
    return info.st_dev, info.st_ino


def _process_identity(pid: int) -> Process:
    path = Path("/proc") / str(pid)
    parent, process_group, session, start_time, state, pages = _read_stat(path / "stat")
    try:
        namespace = _namespace_identity(path / "ns" / "pid")
    except AccountingError:
        namespace = None
    return Process(
        pid,
        parent,
        process_group,
        session,
        start_time,
        state,
        pages * os.sysconf("SC_PAGE_SIZE"),
        namespace,
    )


def _read_stat(path: Path) -> tuple[int, int, int, int, str, int]:
    try:
        contents = path.read_text()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise AccountingError(f"cannot read process accounting data at {path}: {exc}") from exc
    closing = contents.rfind(")")
    if closing < 0:
        raise AccountingError(f"malformed process accounting data at {path}")
    fields = contents[closing + 2 :].split()
    try:
        state = fields[0]
        parent = int(fields[1])
        process_group = int(fields[2])
        session = int(fields[3])
        start_time = int(fields[19])
        resident_pages = int(fields[21])
    except (IndexError, ValueError) as exc:
        raise AccountingError(f"malformed process accounting data at {path}") from exc
    if resident_pages < 0:
        raise AccountingError(f"negative resident-page count in process accounting data at {path}")
    return parent, process_group, session, start_time, state, resident_pages


def _process_snapshot(
    namespace: tuple[int, int], deadline: float | None = None
) -> ProcessTable:
    if _namespace_identity(Path("/proc/self/ns/pid")) != namespace:
        raise AccountingError("runner PID namespace changed during process accounting")
    proc = Path("/proc")
    try:
        entries = tuple(proc.iterdir())
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError) as exc:
        raise AccountingError(f"cannot enumerate /proc process accounting: {exc}") from exc
    result = ProcessTable()
    for entry in entries:
        if deadline is not None:
            _check_tree_deadline(deadline)
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            parent, process_group, session, start_time, state, pages = _read_stat(entry / "stat")
        except FileNotFoundError:
            # Processes can exit between /proc enumeration and the reads.
            continue
        except AccountingError:
            # Ownership is not known yet. Fail closed only if an owned process
            # lists this PID as a child when its tree is checked.
            result.unreadable_pids.add(pid)
            continue
        try:
            identity = _namespace_identity(entry / "ns" / "pid")
        except FileNotFoundError:
            try:
                current = _read_stat(entry / "stat")
            except FileNotFoundError:
                continue
            except AccountingError:
                result.unreadable_pids.add(pid)
                continue
            if current[3] != start_time:
                continue
            identity = None
        except AccountingError:
            identity = None
        result[pid] = Process(
            pid,
            parent,
            process_group,
            session,
            start_time,
            state,
            pages * page_size,
            identity,
        )
    try:
        runner_stat = _read_stat(Path("/proc/self/stat"))
    except (FileNotFoundError, AccountingError) as exc:
        raise AccountingError(f"cannot verify runner process identity in /proc: {exc}") from exc
    _validate_runner_entry(result, os.getpid(), runner_stat[3], namespace)
    return result


def _enable_subreaper() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("bounded process-tree cleanup requires Linux /proc and subreaper support")
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
        prctl.restype = ctypes.c_int
        if prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    except (AttributeError, OSError) as exc:
        raise RuntimeError(f"cannot enable Linux child subreaper accounting: {exc}") from exc


def _descendants(snapshot: ProcessTable, root_pid: int, runner_pid: int) -> dict[int, Process]:
    tracked: dict[int, Process] = {}
    runner = snapshot.get(runner_pid)
    runner_namespace = runner.namespace if runner is not None else None
    parents = {runner_pid}
    owned_groups = {(root_pid, root_pid)}
    changed = True
    while changed:
        changed = False
        for process in snapshot.values():
            if process.pid in tracked:
                continue
            if (
                (
                    runner_namespace is not None
                    and process.namespace == runner_namespace
                    and (process.process_group, process.session) in owned_groups
                )
                or process.parent in parents
            ):
                tracked[process.pid] = process
                parents.add(process.pid)
                owned_groups.add((process.process_group, process.session))
                changed = True
    return tracked


def _validate_tracked_namespaces(
    processes: dict[int, Process], namespace: tuple[int, int]
) -> None:
    for process in processes.values():
        if process.namespace != namespace:
            raise AccountingError(
                f"PID {process.pid} is outside the runner's PID namespace; process accounting is incomplete"
            )


def _unreadable_owned_children(
    snapshot: ProcessTable,
    tracked: dict[int, Process],
    runner_pid: int,
    namespace: tuple[int, int],
    deadline: float,
) -> int:
    parents = set(tracked) | {runner_pid}
    owned_groups = {
        (process.process_group, process.session) for process in tracked.values()
    }
    added = 0
    for parent_pid in parents:
        _check_tree_deadline(deadline)
        if parent_pid != runner_pid:
            expected = tracked[parent_pid]
            try:
                current = _read_stat(Path("/proc") / str(parent_pid) / "stat")
            except FileNotFoundError:
                continue
            if current[3] != expected.start_time:
                continue
        for child_pid in _task_children(parent_pid, deadline):
            _check_tree_deadline(deadline)
            previous = snapshot.get(child_pid)
            try:
                child = _process_identity(child_pid)
            except FileNotFoundError:
                # Confirm retirement against the current parent membership.
                if child_pid not in _task_children(parent_pid, deadline):
                    snapshot.unreadable_pids.discard(child_pid)
                    continue
                try:
                    child = _process_identity(child_pid)
                except FileNotFoundError:
                    if child_pid not in _task_children(parent_pid, deadline):
                        snapshot.unreadable_pids.discard(child_pid)
                        continue
                    raise AccountingError(
                        f"cannot verify listed child PID {child_pid} of tracked PID {parent_pid}"
                    )
            if child.namespace != namespace:
                raise AccountingError(
                    f"PID {child_pid} is outside the runner's PID namespace; process accounting is incomplete"
                )
            if child.parent not in parents and (child.process_group, child.session) not in owned_groups:
                # The process may have exited and its PID may have been reused
                # between the child-list and stat reads. Refresh both facts
                # once; persistent ambiguity remains an accounting failure.
                current_children = _task_children(parent_pid, deadline)
                try:
                    refreshed = _process_identity(child_pid)
                except FileNotFoundError:
                    if child_pid not in current_children:
                        snapshot.unreadable_pids.discard(child_pid)
                        continue
                    raise AccountingError(
                        f"cannot verify listed child PID {child_pid} of tracked PID {parent_pid}"
                    )
                if refreshed.start_time != child.start_time:
                    if child_pid not in current_children:
                        snapshot.pop(child_pid, None)
                        snapshot.unreadable_pids.discard(child_pid)
                        continue
                    child = refreshed
                else:
                    child = refreshed
                if child.namespace != namespace:
                    raise AccountingError(
                        f"PID {child_pid} is outside the runner's PID namespace; process accounting is incomplete"
                    )
                if child.live and child.parent not in parents and (
                    child.process_group,
                    child.session,
                ) not in owned_groups:
                    if previous is not None and previous.start_time != child.start_time and (
                        child_pid not in current_children
                    ):
                        snapshot.pop(child_pid, None)
                        snapshot.unreadable_pids.discard(child_pid)
                        continue
                    raise AccountingError(
                        f"cannot prove ownership of live child PID {child_pid}"
                    )
            if previous is None or previous.start_time != child.start_time:
                added += 1
            snapshot[child_pid] = child
            snapshot.unreadable_pids.discard(child_pid)
    return added


def _validate_process_tree(
    snapshot: ProcessTable,
    tracked: dict[int, Process],
    runner_pid: int,
    namespace: tuple[int, int],
    root_pid: int,
    deadline: float,
) -> dict[int, Process]:
    while True:
        _check_tree_deadline(deadline)
        added = _unreadable_owned_children(
            snapshot, tracked, runner_pid, namespace, deadline
        )
        tracked = _descendants(snapshot, root_pid, runner_pid)
        _validate_tracked_namespaces(tracked, namespace)
        if not added:
            return tracked


def _check_tree_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TreeDeadlineExpired("process-tree reconciliation exceeded its budget")


def _task_children(parent_pid: int, deadline: float | None = None) -> set[int]:
    if deadline is not None:
        _check_tree_deadline(deadline)
    task_root = Path("/proc") / str(parent_pid) / "task"
    try:
        tasks = tuple(task_root.iterdir())
    except FileNotFoundError:
        return set()
    except OSError as exc:
        raise AccountingError(f"cannot inspect tracked PID {parent_pid} tasks: {exc}") from exc
    children: set[int] = set()
    for task in tasks:
        if deadline is not None:
            _check_tree_deadline(deadline)
        if not task.name.isdecimal():
            continue
        try:
            child_ids = (task / "children").read_text().split()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise AccountingError(
                f"cannot inspect tracked PID {parent_pid} child accounting: {exc}"
            ) from exc
        try:
            children.update(map(int, child_ids))
        except ValueError as exc:
            raise AccountingError(
                f"malformed child accounting for tracked PID {parent_pid}"
            ) from exc
    if deadline is not None:
        _check_tree_deadline(deadline)
    return children


def _signal_tracked(processes: dict[int, Process], sig: int) -> None:
    for process in processes.values():
        if not process.live:
            continue
        try:
            current = _read_stat(Path("/proc") / str(process.pid) / "stat")
        except FileNotFoundError:
            continue
        if current[3] != process.start_time:
            # The original process is gone; never signal a different task that reused its PID.
            continue
        try:
            os.kill(process.pid, sig)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            raise AccountingError(f"cannot signal tracked PID {process.pid}: {exc}") from exc


def _signal_known_group(
    root_pid: int,
    known: dict[int, Process],
    namespace: tuple[int, int],
    sig: int,
) -> bool:
    """Signal the command group only while a previously seen member proves its identity."""
    for process in known.values():
        if (
            not process.live
            or process.namespace != namespace
            or process.process_group != root_pid
            or process.session != root_pid
        ):
            continue
        try:
            current = _process_identity(process.pid)
        except FileNotFoundError:
            continue
        if (
            current.start_time != process.start_time
            or current.namespace != namespace
            or current.process_group != root_pid
            or current.session != root_pid
        ):
            continue
        try:
            os.killpg(root_pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise AccountingError(f"cannot signal tracked process group {root_pid}: {exc}") from exc
        return True
    return False


def _reap_adopted(root_pid: int, tracked: dict[int, Process]) -> None:
    for pid in tracked:
        if pid == root_pid:
            continue
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            continue


def _limits(cpu_seconds: int, address_space_bytes: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))


def _acquire_lock(wait_seconds: float, wall_deadline: float) -> int:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(LOCK_FILE, flags, 0o600)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            os.close(fd)
            raise RuntimeError(f"shared test lock has unexpected ownership or type: {LOCK_FILE}")
    except OSError as exc:
        raise RuntimeError(f"cannot open shared test lock {LOCK_FILE}: {exc}") from exc
    lock_deadline = time.monotonic() + wait_seconds
    deadline = min(lock_deadline, wall_deadline)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                if wall_deadline <= lock_deadline:
                    raise WallBudgetExpired(
                        "overall wall budget expired while waiting for the shared test lock"
                    )
                raise LockWaitExpired(
                    "shared test lock exceeded its bounded wait budget"
                )
            time.sleep(min(LOCK_POLL_SECONDS, max(0.0, deadline - time.monotonic())))


def _release_lock(fd: int | None) -> None:
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _terminate_and_verify(
    root_pid: int,
    namespace: tuple[int, int],
    known: dict[int, Process],
    cleanup_seconds: float,
) -> CleanupResult:
    started = time.monotonic()
    deadline = started + cleanup_seconds
    term_deadline = started + min(0.5, cleanup_seconds * 0.4)
    phase = signal.SIGTERM
    accounting_errors: list[str] = []
    group_signal_attempted = False
    while time.monotonic() < deadline:
        try:
            snapshot = _process_snapshot(namespace, deadline)
            live = _descendants(snapshot, root_pid, os.getpid())
            live = _validate_process_tree(
                snapshot, live, os.getpid(), namespace, root_pid, deadline
            )
            known.update(live)
            _reap_adopted(root_pid, live)
            if not any(process.live for process in live.values()):
                return CleanupResult(True, "all owned members exited and accounting completed")
        except AccountingError as exc:
            accounting_errors.append(str(exc))
            live = {}
        if time.monotonic() >= term_deadline:
            phase = signal.SIGKILL
        try:
            if live:
                _signal_tracked(live, phase)
        except AccountingError as exc:
            accounting_errors.append(str(exc))
        try:
            group_signal_attempted = (
                _signal_known_group(root_pid, known, namespace, phase)
                or group_signal_attempted
            )
        except AccountingError as exc:
            accounting_errors.append(str(exc))
        time.sleep(POLL_SECONDS)
    detail = (
        "cleanup deadline expired; cleanup_unverified"
        f"; safe_owned_pid_tokens={len(known)}"
        f"; known_group_signal_attempted={str(group_signal_attempted).lower()}"
    )
    if accounting_errors:
        detail += "; accounting_failure=" + " | ".join(dict.fromkeys(accounting_errors))
    return CleanupResult(False, detail)


def _report(
    *,
    command: list[str],
    exit_code: int,
    command_exit_code: int | None,
    reason: str,
    elapsed: float,
    peak_rss_mib: float,
    cleanup: str,
    wall_budget_seconds: float,
    lock_budget_seconds: float,
    cleanup_budget_seconds: float,
) -> None:
    payload = {
        "command": command,
        "runner_exit_status": exit_code,
        "command_exit_status": command_exit_code,
        "reason": reason,
        "elapsed_seconds": round(elapsed, 3),
        "peak_process_tree_rss_mib": round(peak_rss_mib, 2),
        "cleanup": cleanup,
        "wall_budget_seconds": wall_budget_seconds,
        "lock_wait_budget_seconds": lock_budget_seconds,
        "cleanup_budget_seconds": cleanup_budget_seconds,
        "rss_limit_kind": "summed sampled /proc RSS; shared pages may repeat; not a hard kernel cap",
    }
    print(REPORT_PREFIX + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def run_command(args: argparse.Namespace) -> int:
    command = args.command
    lock_fd: int | None = None
    child: subprocess.Popen[bytes] | None = None
    known_members: dict[int, Process] = {}
    runner_namespace: tuple[int, int] | None = None
    started = time.monotonic()
    wall_deadline = started + args.wall_seconds
    peak_rss = 0
    termination_signal: int | None = None
    old_handlers: dict[int, object] = {}

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal termination_signal
        termination_signal = signum

    try:
        lock_fd = _acquire_lock(args.lock_wait_seconds, wall_deadline)
        if time.monotonic() >= wall_deadline:
            raise WallBudgetExpired("overall wall budget expired before command startup")
        _enable_subreaper()
        runner_namespace = _namespace_identity(Path("/proc/self/ns/pid"))
        _process_snapshot(runner_namespace)  # Verify the runner namespace and /proc availability.
        if time.monotonic() >= wall_deadline:
            raise WallBudgetExpired("overall wall budget expired during runner startup")
        old_handlers = {
            sig: signal.signal(sig, request_stop)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        child = subprocess.Popen(
            command,
            start_new_session=True,
            preexec_fn=lambda: _limits(
                args.cpu_seconds,
                args.address_space_mib * 1024 * 1024,
            ),
        )
        known_members[child.pid] = _process_identity(child.pid)
        command_started = time.monotonic()
        deadline = wall_deadline
        reason = "command_exit"
        child_status: int | None = None
        while True:
            snapshot = _process_snapshot(runner_namespace, deadline)
            active = _descendants(snapshot, child.pid, os.getpid())
            active = _validate_process_tree(
                snapshot, active, os.getpid(), runner_namespace, child.pid, deadline
            )
            known_members.update(active)
            _reap_adopted(child.pid, active)
            peak_rss = max(
                peak_rss,
                sum(proc.rss_bytes for proc in active.values() if proc.live) / (1024 * 1024),
            )
            child_status = child.poll()
            if termination_signal is not None:
                reason = f"runner_signal_{signal.Signals(termination_signal).name}"
                break
            if peak_rss > args.rss_mib:
                reason = "aggregate_rss_limit"
                break
            if time.monotonic() >= deadline:
                reason = "wall_time_limit"
                break
            if child_status is not None:
                final_snapshot = _process_snapshot(runner_namespace, deadline)
                active = _descendants(final_snapshot, child.pid, os.getpid())
                active = _validate_process_tree(
                    final_snapshot,
                    active,
                    os.getpid(),
                    runner_namespace,
                    child.pid,
                    deadline,
                )
                _reap_adopted(child.pid, active)
                if any(process.live for process in active.values()):
                    reason = "command_exit_with_surviving_children"
                else:
                    if child_status == -signal.SIGXCPU:
                        reason = "cpu_limit"
                    elif child_status == -signal.SIGKILL:
                        reason = "command_terminated_by_sigkill"
                    exit_code = 128 + -child_status if child_status < 0 else child_status
                    _report(
                        command=command,
                        exit_code=exit_code,
                        command_exit_code=child_status,
                        reason=reason,
                        elapsed=time.monotonic() - started,
                        peak_rss_mib=peak_rss,
                        cleanup="verified_no_live_members",
                        wall_budget_seconds=args.wall_seconds,
                        lock_budget_seconds=args.lock_wait_seconds,
                        cleanup_budget_seconds=args.cleanup_seconds,
                    )
                    return exit_code
                break
            time.sleep(POLL_SECONDS)

        cleanup_namespace = _namespace_identity(Path("/proc/self/ns/pid"))
        if cleanup_namespace != runner_namespace:
            raise AccountingError("runner PID namespace changed during the command")
        cleanup = _terminate_and_verify(
            child.pid, runner_namespace, known_members, args.cleanup_seconds
        )
        child_status = child.poll()
        elapsed = time.monotonic() - started
        if termination_signal is not None:
            exit_code = 128 + termination_signal
        elif reason == "wall_time_limit":
            exit_code = 124
        elif reason == "aggregate_rss_limit":
            exit_code = 125
        elif reason == "command_exit_with_surviving_children":
            exit_code = 125
        elif child_status is not None and child_status < 0:
            exit_code = 128 + -child_status
        else:
            exit_code = child_status
        if not cleanup.verified:
            exit_code = 125
            reason = f"{reason}; {cleanup.detail}"
        _report(
            command=command,
            exit_code=exit_code,
            command_exit_code=child_status,
            reason=reason,
            elapsed=elapsed,
            peak_rss_mib=peak_rss,
            cleanup="verified_no_live_members" if cleanup.verified else cleanup.detail,
            wall_budget_seconds=args.wall_seconds,
            lock_budget_seconds=args.lock_wait_seconds,
            cleanup_budget_seconds=args.cleanup_seconds,
        )
        return exit_code
    except TreeDeadlineExpired as exc:
        cleanup = (
            _terminate_and_verify(
                child.pid,
                runner_namespace,
                known_members,
                args.cleanup_seconds,
            )
            if child is not None and runner_namespace is not None
            else CleanupResult(False, "cleanup_unverified; runner namespace unavailable")
        )
        exit_code = 124 if cleanup.verified else 125
        reason = f"wall_time_limit: {exc}"
        if not cleanup.verified:
            reason += f"; {cleanup.detail}"
        _report(
            command=command,
            exit_code=exit_code,
            command_exit_code=child.poll() if child is not None else None,
            reason=reason,
            elapsed=time.monotonic() - started,
            peak_rss_mib=peak_rss,
            cleanup="verified_no_live_members" if cleanup.verified else cleanup.detail,
            wall_budget_seconds=args.wall_seconds,
            lock_budget_seconds=args.lock_wait_seconds,
            cleanup_budget_seconds=args.cleanup_seconds,
        )
        return exit_code
    except (WallBudgetExpired, LockWaitExpired) as exc:
        reason = (
            "overall_wall_budget_expired"
            if isinstance(exc, WallBudgetExpired)
            else "lock_wait_limit"
        )
        _report(
            command=command,
            exit_code=125,
            command_exit_code=None,
            reason=f"{reason}: {exc}",
            elapsed=time.monotonic() - started,
            peak_rss_mib=peak_rss,
            cleanup="no_command_started",
            wall_budget_seconds=args.wall_seconds,
            lock_budget_seconds=args.lock_wait_seconds,
            cleanup_budget_seconds=args.cleanup_seconds,
        )
        return 125
    except BaseException as exc:
        cleanup_result: CleanupResult | None = None
        if child is not None:
            if runner_namespace is not None:
                cleanup_result = _terminate_and_verify(
                    child.pid, runner_namespace, known_members, args.cleanup_seconds
                )
            else:
                cleanup_result = CleanupResult(
                    False,
                    "cleanup_unverified; safe_owned_pid_tokens=0; "
                    "known_group_signal_attempted=false; "
                    "accounting_failure=runner namespace was not verified",
                )
        if isinstance(exc, KeyboardInterrupt):
            exit_code = 130
        elif isinstance(exc, SystemExit):
            raise
        else:
            exit_code = 125
        _report(
            command=command,
            exit_code=exit_code,
            command_exit_code=child.poll() if child is not None else None,
            reason=(
                f"runner_error: {type(exc).__name__}: {exc}"
                if cleanup_result is None or cleanup_result.verified
                else f"runner_error: {type(exc).__name__}: {exc}; {cleanup_result.detail}"
            ),
            elapsed=time.monotonic() - started,
            peak_rss_mib=peak_rss,
            cleanup=(
                "no_command_started"
                if child is None
                else "verified_no_live_members"
                if cleanup_result is not None and cleanup_result.verified
                else cleanup_result.detail
                if cleanup_result is not None
                else "cleanup_unverified; cleanup did not complete"
            ),
            wall_budget_seconds=args.wall_seconds,
            lock_budget_seconds=args.lock_wait_seconds,
            cleanup_budget_seconds=args.cleanup_seconds,
        )
        return 125 if child is not None and cleanup_result and not cleanup_result.verified else exit_code
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        _release_lock(lock_fd)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Linux command with serialized wall, sampled RSS, CPU, and address-space budgets."
    )
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--rss-mib", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--address-space-mib", type=int, required=True)
    parser.add_argument("--lock-wait-seconds", type=float, default=300)
    parser.add_argument("--cleanup-seconds", type=float, default=5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if (
        not math.isfinite(args.wall_seconds)
        or args.wall_seconds <= 0
        or args.rss_mib <= 0
        or args.cpu_seconds <= 0
        or args.address_space_mib <= 0
        or not math.isfinite(args.cleanup_seconds)
        or args.cleanup_seconds <= 0
    ):
        parser.error("wall, RSS, CPU, address-space, and cleanup budgets must be positive")
    if not math.isfinite(args.lock_wait_seconds) or args.lock_wait_seconds < 0:
        parser.error("lock wait must be a finite nonnegative number")
    return args


def main(argv: list[str] | None = None) -> int:
    return run_command(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
