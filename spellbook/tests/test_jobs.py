import fcntl
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

import jobs  # noqa: E402
from utils import resources as res  # noqa: E402


def _submit(root, argv=None, **kwargs):
    if argv is None:
        argv = [sys.executable, "-c", "print('hi')"]
    kwargs.setdefault("cwd", root)
    kwargs.setdefault("scannet_root", root)
    kwargs.setdefault("launch", False)
    return jobs.submit(argv, **kwargs)


class SubmitTests(unittest.TestCase):
    def test_rejects_empty_argv(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                _submit(root, argv=[])

    def test_rejects_missing_executable(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                _submit(root, argv=["/nonexistent/bin/xyz"])

    def test_rejects_missing_cwd(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                jobs.submit([sys.executable, "-c", "pass"],
                            cwd=os.path.join(root, "nope"),
                            scannet_root=root, launch=False)

    def test_rejects_interactive(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                _submit(root, argv=[sys.executable, "spellbook/main.py",
                                    "--visualize"])

    def test_submit_writes_registry(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            d = os.path.join(root, "derived", "jobs", job_id)
            job = jobs._read_json(os.path.join(d, "job.json"))
            status = jobs._read_json(os.path.join(d, "status.json"))
            self.assertEqual(job["job_id"], job_id)
            self.assertEqual(job["kind"], "command")
            self.assertTrue(os.path.isabs(job["argv"][0]))
            self.assertEqual(job["cwd"], root)
            self.assertEqual(job["run_key"], "command:")
            self.assertTrue(os.path.isfile(job["runner"]))
            self.assertEqual(status["state"], "queued")
            self.assertFalse(os.path.isfile(os.path.join(d, "job.json.tmp")))

    def test_concurrent_status_updates_never_collide(self):
        import multiprocessing

        def hammer(path, tag, n):
            for i in range(n):
                doc = jobs._read_json(path)
                doc.update({tag: i})
                jobs._atomic_write_json(path, doc)

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "status.json")
            jobs._atomic_write_json(path, {"job_id": "x"})
            procs = [multiprocessing.Process(target=hammer, args=(path, t, 30))
                     for t in ("a", "b")]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=60)
                self.assertEqual(p.exitcode, 0)
            final = jobs._read_json(path)
            # Last-writer-wins on content is fine; the point is no writer
            # ever crashes on a stolen temp file and the record stays valid.
            self.assertEqual(final["job_id"], "x")
            self.assertTrue("a" in final or "b" in final)

    def test_duplicate_active_returns_existing(self):
        with tempfile.TemporaryDirectory() as root:
            first = _submit(root)
            jobs._update_status(first, root, state="running")
            with mock.patch("jobs._unit_active", return_value=True):
                with self.assertRaises(ValueError) as ctx:
                    _submit(root)
            self.assertIn(first, str(ctx.exception))
            self.assertEqual(len(os.listdir(os.path.join(root, "derived", "jobs"))), 1)

    def test_submit_refuses_duplicate_run_id(self):
        with tempfile.TemporaryDirectory() as root:
            first = _submit(root, kind="predict", run_id="dup-run")
            with self.assertRaises(ValueError) as ctx:
                _submit(root, kind="predict", run_id="dup-run")
            self.assertIn(first, str(ctx.exception))
            self.assertEqual(len(os.listdir(os.path.join(root, "derived", "jobs"))), 1)

    def test_submit_allows_different_run_ids(self):
        with tempfile.TemporaryDirectory() as root:
            _submit(root, kind="predict", run_id="run-a")
            _submit(root, kind="predict", run_id="run-b")
            self.assertEqual(len(os.listdir(os.path.join(root, "derived", "jobs"))), 2)

    def test_run_key_ignores_cwd_and_commit(self):
        argv = [sys.executable, "spellbook/main.py", "--engine", "metashape",
                "--scene", "9004"]
        with tempfile.TemporaryDirectory() as root, \
                tempfile.TemporaryDirectory() as other:
            _submit(root, argv=argv, cwd=root, kind="reconstruct")
            with self.assertRaises(ValueError):
                _submit(root, argv=argv, cwd=other, kind="reconstruct")

    def test_reconstruct_identity_from_flags(self):
        argv = [sys.executable, "main.py", "--engine", "metashape",
                "--scene", "9004"]
        self.assertEqual(
            jobs._run_key(argv, "reconstruct", None),
            "reconstruct:--engine=metashape|--scene=9004")


class RunWrapperTests(unittest.TestCase):
    def test_success_records_status_and_log(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            rc = jobs._run(job_id, scannet_root=root)
            self.assertEqual(rc, 0)
            status = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "status.json"))
            self.assertEqual(status["state"], "succeeded")
            self.assertEqual(status["exit_code"], 0)
            self.assertIn("started", status)
            self.assertIn("ended", status)
            with open(os.path.join(root, "derived", "jobs", job_id, "job.log")) as f:
                log = f.read()
            self.assertIn("hi", log)

    def test_nonzero_exit_marks_failed(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root, argv=[sys.executable, "-c",
                                         "import sys; sys.exit(7)"])
            rc = jobs._run(job_id, scannet_root=root)
            self.assertEqual(rc, 7)
            status = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "status.json"))
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["exit_code"], 7)

    def test_missing_job_json_returns_2(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(jobs._run("job-none", scannet_root=root), 2)

    def test_cancel_stops_mocked_unit(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            jobs._update_status(job_id, root, state="running")
            with mock.patch("jobs._unit_active", side_effect=[True, False]), \
                    mock.patch("jobs.subprocess.run") as run:
                run.return_value = mock.Mock(returncode=0)
                state = jobs.cancel_job(job_id, scannet_root=root)
            self.assertEqual(state, "cancelled")
            status = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "status.json"))
            self.assertEqual(status["state"], "cancelled")

    def test_retry_preserves_spec(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root, argv=[sys.executable, "-c", "pass"],
                             kind="predict", run_id="run-x")
            jobs._update_status(job_id, root, state="failed", exit_code=3)
            with mock.patch("jobs._launch_unit", return_value="u"):
                new_id = jobs.retry_job(job_id, scannet_root=root)
            self.assertNotEqual(new_id, job_id)
            new_job = jobs._read_json(os.path.join(
                root, "derived", "jobs", new_id, "job.json"))
            old_job = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "job.json"))
            self.assertEqual(new_job["argv"], old_job["argv"])
            self.assertEqual(new_job["run_id"], "run-x")
            self.assertEqual(new_job["run_key"], old_job["run_key"])
            self.assertEqual(new_job["retry_of"], job_id)

    def test_retry_refuses_active(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            jobs._update_status(job_id, root, state="running")
            with mock.patch("jobs._unit_active", return_value=True):
                with self.assertRaises(ValueError):
                    jobs.retry_job(job_id, scannet_root=root)

    def test_retry_refuses_missing_cwd(self):
        with tempfile.TemporaryDirectory() as root:
            nested = os.path.join(root, "work")
            os.makedirs(nested)
            job_id = _submit(root, cwd=nested)
            import shutil
            shutil.rmtree(nested)
            with self.assertRaises(ValueError) as ctx:
                jobs.retry_job(job_id, scannet_root=root)
            self.assertIn("cwd is gone", str(ctx.exception))


class GpuWaitTests(unittest.TestCase):
    def test_run_initializes_gpu_wait(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            jobs._run(job_id, scannet_root=root)
            status = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "status.json"))
            self.assertEqual(status["gpu_wait"], 0.0)

    def test_record_gpu_wait_accumulates(self):
        from utils.gpu import _record_gpu_wait
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            jobs._update_status(job_id, root, gpu_wait=2.5)
            with mock.patch.dict(os.environ,
                                 {"SPELLBOOK_JOB_ID": job_id}, clear=False):
                _record_gpu_wait(root, 1.25)
            status = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "status.json"))
            self.assertAlmostEqual(status["gpu_wait"], 3.75)

    def test_record_gpu_wait_never_raises_without_job(self):
        from utils.gpu import _record_gpu_wait
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPELLBOOK_JOB_ID", None)
            _record_gpu_wait(root, 1.0)

    def test_list_jobs_exposes_gpu_wait(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            jobs._update_status(job_id, root, gpu_wait=12.5)
            rows = jobs.list_jobs(root)
            self.assertEqual(rows[0]["gpu_wait"], 12.5)


class WorkerLimitTests(unittest.TestCase):
    def test_foreground_default(self):
        from utils.gpu import detached_worker_limit
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPELLBOOK_JOB_MAX_GPU_WORKERS", None)
            os.environ.pop("SPELLBOOK_JOB_ID", None)
            self.assertEqual(detached_worker_limit(4), 4)

    def test_detached_cap(self):
        from utils.gpu import detached_worker_limit
        env = {"SPELLBOOK_JOB_ID": "job-1", "SPELLBOOK_JOB_MAX_GPU_WORKERS": "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(detached_worker_limit(4), 1)

    def test_limit_without_job_rejected(self):
        from utils.gpu import detached_worker_limit
        env = {"SPELLBOOK_JOB_MAX_GPU_WORKERS": "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("SPELLBOOK_JOB_ID", None)
            with self.assertRaises(RuntimeError):
                detached_worker_limit(4)

    def test_out_of_range_rejected(self):
        from utils.gpu import detached_worker_limit
        for bad in ("0", "5", "x"):
            env = {"SPELLBOOK_JOB_ID": "job-1",
                   "SPELLBOOK_JOB_MAX_GPU_WORKERS": bad}
            with mock.patch.dict(os.environ, env, clear=False):
                with self.assertRaises(RuntimeError):
                    detached_worker_limit(4)


class JobContextTests(unittest.TestCase):
    def _clean_env(self):
        return mock.patch.dict(os.environ, {}, clear=False)

    def test_job_context_creates_and_completes_record(self):
        with tempfile.TemporaryDirectory() as root, self._clean_env():
            os.environ.pop("SPELLBOOK_JOB_ID", None)
            with jobs.job_context([sys.executable, "-c", "print('hi')"],
                                  cwd=root, kind="command",
                                  scannet_root=root) as job_id:
                self.assertEqual(os.environ.get("SPELLBOOK_JOB_ID"), job_id)
                status = jobs._read_json(os.path.join(
                    root, "derived", "jobs", job_id, "status.json"))
                self.assertEqual(status["state"], "running")
            self.assertNotIn("SPELLBOOK_JOB_ID", os.environ)
            job = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "job.json"))
            status = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "status.json"))
            self.assertIsNone(job["unit"])
            self.assertEqual(status["state"], "succeeded")

    def test_job_context_adopts_existing_job(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"SPELLBOOK_JOB_ID": "job-x"},
                                 clear=False), \
                    mock.patch("jobs.submit") as submit, \
                    mock.patch("jobs._update_status") as update:
                with jobs.job_context([sys.executable, "-c", "pass"],
                                      cwd=root,
                                      scannet_root=root) as job_id:
                    self.assertEqual(job_id, "job-x")
            submit.assert_not_called()
            update.assert_not_called()

    def test_job_context_marks_failed_on_exception(self):
        with tempfile.TemporaryDirectory() as root, self._clean_env():
            os.environ.pop("SPELLBOOK_JOB_ID", None)
            with self.assertRaises(RuntimeError):
                with jobs.job_context([sys.executable, "-c", "pass"],
                                      cwd=root, scannet_root=root):
                    raise RuntimeError("boom")
            self.assertNotIn("SPELLBOOK_JOB_ID", os.environ)
            rows = jobs.list_jobs(root)
            self.assertEqual(rows[0]["state"], "failed")

    def test_lease_requires_job_record(self):
        from utils.gpu import gpu_lease
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("utils.gpu._present_gpus", return_value={1}), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPELLBOOK_JOB_ID", None)
            with self.assertRaises(RuntimeError) as ctx:
                with gpu_lease([1], root):
                    pass
            self.assertIn("job record", str(ctx.exception))

    def test_derive_state_persists_lost(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            jobs._update_status(job_id, root, state="running",
                                wrapper_pid=99999999)
            rows = jobs.list_jobs(root)
            self.assertEqual(rows[0]["state"], "lost")
            status = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "status.json"))
            self.assertEqual(status["state"], "lost")

    def test_foreground_record_lost_when_pid_dead(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            jobs._update_status(job_id, root, state="running",
                                wrapper_pid=99999999)
            job = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "job.json"))
            self.assertIsNone(job.get("unit"))
            status = jobs._read_json(os.path.join(
                root, "derived", "jobs", job_id, "status.json"))
            self.assertEqual(
                jobs.derive_state(job_id, job, status, root), "lost")

    def test_canonical_runner_resolves_main_checkout(self):
        runner = jobs._canonical_spellbook(jobs._SPELLBOOK)
        self.assertTrue(runner.endswith("spellbook"))
        self.assertTrue(os.path.isfile(os.path.join(runner, "jobs.py")))
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(jobs._canonical_spellbook(root),
                             jobs._SPELLBOOK)


class ResourceTests(unittest.TestCase):
    def test_lease_missing_is_free_and_not_created(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(res.gpu_leases(root, [1, 2]), {})
            self.assertEqual(os.listdir(root), [])

    def test_lease_held_and_released(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "gpu-1.lock")
            open(path, "w").close()
            fd = os.open(path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(res.gpu_leases(root, [1]), {1: os.getpid()})
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            self.assertEqual(res.gpu_leases(root, [1]), {})

    def test_cpu_helpers_smoke(self):
        prev = res.cpu_times()
        self.assertIsNotNone(prev)
        cur = res.cpu_times()
        util = res.cpu_util(prev, cur)
        self.assertTrue(util is None or 0.0 <= util <= 1.0)
        la = res.load_average()
        self.assertEqual(len(la), 3)

    def test_gpu_stats_empty_without_driver(self):
        with mock.patch("utils.resources.subprocess.run",
                        side_effect=OSError("no nvidia-smi")):
            self.assertEqual(res.gpu_stats(), {})

    def test_process_info_describes_current_process(self):
        info = res.process_info(os.getpid())
        self.assertEqual(info["pid"], os.getpid())
        self.assertTrue(info["argv"])
        self.assertIsNotNone(info["cwd"])
        self.assertIsInstance(info["started"], float)

    def test_render_snapshot_lists(self):
        with tempfile.TemporaryDirectory() as root:
            _submit(root, run_id="run-a")
            fake = mock.Mock()
            fake.cpu_times.return_value = {"total": 200, "idle": 100}
            fake.cpu_util.return_value = 0.5
            fake.load_average.return_value = (1.0, 2.0, 3.0)
            fake.gpu_stats.return_value = {
                0: {"util": 0.0, "mem_used": 5.0, "mem_total": 100.0, "temp": 33.0},
                1: {"util": 80.0, "mem_used": 50.0, "mem_total": 100.0, "temp": 60.0},
            }
            fake.gpu_leases.return_value = {}
            with mock.patch("jobs._unit_active", return_value=True):
                lines = jobs.render_snapshot(
                    fake, [1], {"total": 100, "idle": 50}, scannet_root=root)
            text = "\n".join(lines)
            self.assertIn("cpu ", text)
            self.assertIn("CURRENT JOBS", text)
            self.assertIn("FUTURE JOBS", text)
            self.assertIn("PAST JOBS", text)
            self.assertIn("reserved", text)
            self.assertIn("unleased", text)
            self.assertIn("run-a", text)
            self.assertIn("QUEUE", text)
            self.assertNotIn("WORK", text)

    def test_cgroup_helpers(self):
        self.assertEqual(res.read_cpu_usec("/nonexistent-cgroup-xyz"), None)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "cg")
            os.makedirs(path)
            with open(os.path.join(path, "cpu.stat"), "w") as f:
                f.write("usage_usec 123456\nuser_usec 1\nsystem_usec 2\n")
            self.assertEqual(res.read_cpu_usec(path), 123456)
            self.assertEqual(res.cgroup_pids(path), [])
        self.assertIsNone(res.job_cgroup_path(0) if False else
                          res.job_cgroup_path("not-a-pid"))
        with mock.patch("builtins.open", side_effect=OSError("gone")):
            self.assertIsNone(res.job_cgroup_path(os.getpid()))
            self.assertEqual(res.cgroup_pids("/whatever"), [])
            self.assertEqual(res.proc_cmdline(os.getpid()), [])


class StatusUtilTests(unittest.TestCase):
    def _fake_res(self):
        fake = mock.Mock()
        fake.cpu_times.return_value = {"total": 200, "idle": 100}
        fake.cpu_util.return_value = 0.5
        fake.load_average.return_value = (1.0, 2.0, 3.0)
        fake.gpu_stats.return_value = {
            0: {"util": 0.0, "mem_used": 5.0, "mem_total": 100.0, "temp": 33.0},
            1: {"util": 80.0, "mem_used": 50.0, "mem_total": 100.0, "temp": 60.0},
        }
        fake.gpu_leases.return_value = {}
        fake.process_info.return_value = {}
        return fake

    def _rows(self):
        now = 1700000000.0
        return [{
            "job_id": "job-20260910-120000-0001", "kind": "predict",
            "run_id": "run-a", "state": "running", "submitted": now - 60,
            "started": now - 30, "ended": None, "gpu_wait": 0.0,
            "exit_code": None, "branch": "master",
            "log": "/tmp/job-1/job.log", "argv": [],
            "wrapper_pid": None, "child_pid": None,
        }]

    def _render(self, fake, rows, root, pool=None, **kwargs):
        from utils.status import render_snapshot
        return "\n".join(render_snapshot(
            fake, pool if pool is not None else [1],
            {"total": 100, "idle": 50}, rows, root, **kwargs))

    def test_sections_merged_no_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            text = self._render(self._fake_res(), [], root)
        self.assertIn("CURRENT JOBS", text)
        self.assertIn("FUTURE JOBS", text)
        self.assertIn("PAST JOBS", text)
        self.assertNotIn("WORK", text)
        self.assertNotIn("GPU POOL", text)
        self.assertNotIn("RECENT JOBS", text)
        self.assertNotIn("ACTIVE JOBS", text)
        self.assertIn("PROGRESS", text)
        current = text.split("CURRENT JOBS\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("PROGRESS", current)

    def test_low_cpu_unleased_job_is_future_without_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            text = self._render(self._fake_res(), self._rows(), root)
        self.assertIn("reserved", text)
        self.assertIn("unleased", text)
        future = text.split("FUTURE JOBS\n", 1)[1]
        self.assertIn("WAIT", future)
        self.assertIn("run-a", future)
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertNotIn("run-a", current)

    def test_current_gpu_runtime_subtracts_completed_wait(self):
        from utils import status as st
        fake = self._fake_res()
        fake.gpu_leases.return_value = {1: 42}
        fake.process_info.return_value = {"pid": 42, "argv": [], "cwd": None,
                                          "started": 1699999970.0,
                                          "job_id": "job-20260910-120000-0001"}
        rows = self._rows()
        rows[0]["gpu_wait"] = 10.0
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("utils.status.time.time", return_value=1700000000.0):
            text = self._render(fake, rows, root)
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertRegex(current, r"RUN\s+120000-0001\s+1\s+80%.*20s")
        self.assertEqual(st._effective_runtime(rows[0], 1700000000.0), "20s")

    def test_cpu_job_renders_run_with_gpu_dash(self):
        with tempfile.TemporaryDirectory() as root:
            text = self._render(self._fake_res(), self._rows(), root,
                                cpu_cores={"job-20260910-120000-0001": 8.1})
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertRegex(current, r"RUN\s+120000-0001\s+-\s+-")
        self.assertIn("8.1c", current)
        self.assertNotIn("STATE CPU", text)

    def test_gpu_util_attributed_only_to_owner(self):
        fake = self._fake_res()
        fake.gpu_leases.return_value = {1: 42}
        fake.process_info.return_value = {"pid": 42, "argv": [], "cwd": None,
                                          "started": 1699999970.0,
                                          "job_id": "job-20260910-120000-0001"}
        other = dict(self._rows()[0])
        other.update({"job_id": "job-20260910-120000-0002", "run_id": "run-b"})
        rows = self._rows() + [other]
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("utils.status.time.time", return_value=1700000000.0):
            text = self._render(fake, rows, root)
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertEqual(current.count("80%"), 1)
        self.assertIn("run-a", current)
        future = text.split("FUTURE JOBS\n", 1)[1]
        self.assertIn("run-b", future)

    def test_multi_gpu_job_is_single_row(self):
        fake = self._fake_res()
        fake.gpu_stats.return_value = {
            0: {"util": 0.0, "mem_used": 5.0, "mem_total": 100.0, "temp": 33.0},
            1: {"util": 80.0, "mem_used": 50.0, "mem_total": 100.0, "temp": 60.0},
            2: {"util": 70.0, "mem_used": 40.0, "mem_total": 100.0, "temp": 59.0},
        }
        fake.gpu_leases.return_value = {1: 42, 2: 42}
        fake.process_info.return_value = {"pid": 42, "argv": [], "cwd": None,
                                          "started": 1699999970.0,
                                          "job_id": "job-20260910-120000-0001"}
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("utils.status.time.time", return_value=1700000000.0):
            text = self._render(fake, self._rows(), root)
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertEqual(current.count("120000-0001"), 1)
        self.assertIn("1,2", current)
        self.assertIn("80%/70%", current)

    def test_gpu_conditions_all_visible(self):
        base_stats = {
            0: {"util": 0.0, "mem_used": 5.0, "mem_total": 100.0, "temp": 33.0},
            1: {"util": 80.0, "mem_used": 50.0, "mem_total": 100.0, "temp": 60.0},
            2: {"util": 0.0, "mem_used": 10.0, "mem_total": 100.0, "temp": 35.0},
        }
        fake = self._fake_res()
        fake.gpu_stats.return_value = dict(base_stats)
        fake.gpu_leases.return_value = {}
        with tempfile.TemporaryDirectory() as root:
            text = self._render(fake, [], root, pool=[1, 2])
            self.assertIn("RESERVED", text)
            self.assertIn("UNLEASED", text)
            self.assertIn("FREE", text)
            text = self._render(fake, [], root, pool=[1])
            self.assertIn("UNMANAGED", text)
            fake.gpu_stats.return_value = {}
            text = self._render(fake, [], root, pool=[1])
            self.assertIn("UNAVAILABLE", text)

    def test_foreground_lease_keeps_identity_without_job_fields(self):
        fake = self._fake_res()
        fake.gpu_leases.return_value = {1: 42}
        fake.process_info.return_value = {
            "pid": 42,
            "argv": ["spellbook/main.py", "--predict", "--models", "search3d",
                     "--run-id", "smoke-search3d"],
            "cwd": "/tmp/debug/prediction-search3d",
            "started": 1699999940.0, "job_id": None}
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("utils.status.time.time", return_value=1700000000.0):
            text = self._render(fake, [], root)
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertIn("smoke-search3d", current)
        self.assertIn("1m", current)
        self.assertNotIn("python", current)

    def test_cancelling_placement(self):
        rows = self._rows()
        rows[0]["state"] = "cancelling"
        with tempfile.TemporaryDirectory() as root:
            text = self._render(self._fake_res(), rows, root)
        future = text.split("FUTURE JOBS\n", 1)[1].split("PAST JOBS", 1)[0]
        self.assertIn("STOP", future)
        with tempfile.TemporaryDirectory() as root:
            text = self._render(self._fake_res(), rows, root,
                                cpu_cores={"job-20260910-120000-0001": 2.0})
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertIn("STOP", current)

    def test_future_queue_oldest_first_each_once(self):
        now = 1700000000.0
        rows = [{
            "job_id": f"job-20260910-120000-000{i}", "kind": "command",
            "run_id": None, "state": "running", "submitted": now - i * 10,
            "started": now - 5, "ended": None, "gpu_wait": 0.0,
            "exit_code": None, "branch": "master", "log": "/tmp/x.log",
            "argv": [], "wrapper_pid": None, "child_pid": None,
        } for i in (3, 1, 2)]
        with tempfile.TemporaryDirectory() as root:
            text = self._render(self._fake_res(), rows, root)
        future = text.split("FUTURE JOBS\n", 1)[1].split("PAST JOBS", 1)[0]
        self.assertLess(future.index("0003"), future.index("0002"))
        self.assertLess(future.index("0002"), future.index("0001"))
        for suffix in ("0001", "0002", "0003"):
            self.assertEqual(text.count(suffix), 1)

    def test_past_total_and_cap(self):
        from utils.status import render_snapshot
        rows = [{
            "job_id": "job-old", "kind": "predict", "run_id": "old-run",
            "state": "succeeded", "submitted": 1600000000.0,
            "started": 1000.0, "ended": 1100.0, "gpu_wait": 40.0,
            "exit_code": 0, "branch": "master", "log": "/tmp/old.log",
        }, {
            "job_id": "job-lost", "kind": "predict", "run_id": "lost-run",
            "state": "lost", "submitted": 1699999030.0,
            "started": None, "ended": None, "gpu_wait": None,
            "exit_code": None, "branch": "master", "log": "/tmp/lost.log",
        }]
        with tempfile.TemporaryDirectory() as root:
            text = "\n".join(render_snapshot(
                self._fake_res(), [1], {"total": 100, "idle": 50}, rows, root))
        past = text.split("PAST JOBS\n", 1)[1]
        self.assertRegex(past, r"DONE\s+old\s+1m\s+master\s+old-run")
        self.assertRegex(past, r"LOST\s+lost\s+-")
        rows = []
        for i in range(21):
            ended = 1700000000.0 + i
            rows.append({
                "job_id": f"job-20260910-120000-{i:04d}", "kind": "predict",
                "run_id": f"past-{i:02d}", "state": "succeeded",
                "submitted": 1600000000.0, "started": ended - 60,
                "ended": ended, "gpu_wait": 0.0, "exit_code": 0,
                "branch": "master", "log": f"/tmp/job-{i}.log",
            })
        with tempfile.TemporaryDirectory() as root:
            text = "\n".join(render_snapshot(
                self._fake_res(), [1], {"total": 100, "idle": 50}, rows, root))
        past = text.split("PAST JOBS\n", 1)[1]
        self.assertEqual(past.count("past-"), 20)
        self.assertNotIn("past-00", past)
        self.assertLess(past.index("past-20"), past.index("past-19"))

    def test_predict_progress_and_eta(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            eval_root = os.path.join(root, "derived", "evaluations",
                                     "ScanNet20", "run-p")
            model_dir = os.path.join(eval_root, "mosaic3d")
            os.makedirs(model_dir)
            with open(os.path.join(eval_root, "run.json"), "w") as f:
                json.dump({"schema": 2, "benchmark": "ScanNet20",
                           "run_id": "run-p",
                           "scenes": ["scene0046_00", "scene0084_01",
                                      "scene0086_01", "scene0100_02"],
                           "methods": {"mosaic3d": {"parameters": {}}}}, f)
            with open(os.path.join(eval_root, "mosaic3d.tasks"), "w") as f:
                f.write("scene0046_00\nscene0084_01\nscene0046_00\n")
            for scene, elapsed in (("scene0046_00", 60.0),
                                   ("scene0084_01", 120.0)):
                with open(os.path.join(model_dir, scene + ".timing.json"),
                          "w") as f:
                    json.dump({"schema": 1, "kind": "prediction",
                               "scene_id": scene, "run_id": "run-p",
                               "model": "mosaic3d", "elapsed_s": elapsed}, f)
            row = {
                "job_id": "job-20260910-120000-0001", "kind": "predict",
                "run_id": "run-p", "state": "running",
                "submitted": 1699999900.0, "started": 1699999940.0,
                "ended": None, "gpu_wait": 0.0, "exit_code": None,
                "branch": "master", "log": "/tmp/x.log",
                "argv": ["python", "main.py", "--predict",
                         "--models", "mosaic3d", "--run-id", "run-p",
                         "--benchmark", "ScanNet20"],
                "wrapper_pid": None, "child_pid": None}
            fake = self._fake_res()
            fake.gpu_leases.return_value = {1: 42}
            fake.process_info.return_value = {
                "pid": 42, "argv": [], "cwd": None,
                "started": 1699999940.0,
                "job_id": "job-20260910-120000-0001"}
            with mock.patch("utils.status.time.time",
                            return_value=1700000000.0):
                text = self._render(fake, [row], root)
            current = text.split("CURRENT JOBS\n", 1)[1].split(
                "FUTURE JOBS", 1)[0]
            self.assertIn("2/4", current)
            self.assertIn("1m", current)
            self.assertIn("3m", current)

    def test_predict_multi_model_is_dash(self):
        with tempfile.TemporaryDirectory() as root:
            row = dict(self._rows()[0])
            row.update({
                "kind": "predict", "run_id": "run-m",
                "argv": ["python", "main.py", "--predict",
                         "--models", "mosaic3d,openins3d",
                         "--run-id", "run-m"]})
            text = self._render(self._fake_res(), [row], root,
                                cpu_cores={row["job_id"]: 1.0})
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertNotIn("mosaic3d,openins3d", current)

    def test_reconstruct_progress_bounded(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            scans = os.path.join(root, "scans")
            os.makedirs(os.path.join(scans, "scene9004_40", "recon"))
            with open(os.path.join(scans, "scene9004_40", "recon",
                                   "timing.json"), "w") as f:
                json.dump({"schema": 1, "kind": "reconstruction",
                           "scene_id": "scene9004_40", "engine": "open3d",
                           "elapsed_s": 30.0}, f)
            with mock.patch("reconstruct.SCANS_DIR", scans):
                row = dict(self._rows()[0])
                row.update({
                    "kind": "reconstruct", "run_id": None,
                    "argv": ["python", "main.py", "--engine", "open3d",
                             "--scene", "9004"]})
                text = self._render(self._fake_res(), [row], root,
                                    cpu_cores={row["job_id"]: 1.0})
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertIn("1/1", current)

    def test_extract_progress_and_plan(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            man_dir = os.path.join(root, "derived", "reconstruction",
                                   "frames", "scene9009")
            os.makedirs(man_dir)
            with open(os.path.join(man_dir, "frames_manifest.json"), "w") as f:
                json.dump({"color_n": 5}, f)
            row = dict(self._rows()[0])
            row.update({
                "kind": "extract", "run_id": None,
                "argv": ["python", "main.py", "--extract-frames",
                         "--scene", "9009"]})
            text = self._render(self._fake_res(), [row], root,
                                cpu_cores={row["job_id"]: 1.0})
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertIn("1/1", current)

    def test_corrupt_artifacts_and_command_render_dashes(self):
        with tempfile.TemporaryDirectory() as root:
            eval_root = os.path.join(root, "derived", "evaluations",
                                     "ScanNet20", "run-c")
            os.makedirs(eval_root)
            with open(os.path.join(eval_root, "run.json"), "w") as f:
                f.write("{corrupt")
            row = dict(self._rows()[0])
            row.update({
                "kind": "predict", "run_id": "run-c",
                "argv": ["python", "main.py", "--predict",
                         "--models", "mosaic3d", "--run-id", "run-c",
                         "--benchmark", "ScanNet20"]})
            cmd = dict(self._rows()[0])
            cmd.update({"job_id": "job-20260910-120000-0009",
                        "kind": "command", "run_id": None,
                        "argv": ["/bin/true"]})
            text = self._render(
                self._fake_res(), [row, cmd], root,
                cpu_cores={row["job_id"]: 1.0, cmd["job_id"]: 1.0})
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertIn("run-c", current)
        self.assertNotIn("Traceback", current)

    def test_retries_are_separate_rows(self):
        rows = self._rows()
        retry = dict(rows[0])
        retry.update({"job_id": "job-20260910-120000-0002",
                      "run_id": "run-a"})
        rows = rows + [retry]
        with tempfile.TemporaryDirectory() as root:
            text = self._render(
                self._fake_res(), rows, root,
                cpu_cores={r["job_id"]: 1.0 for r in rows})
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertIn("120000-0001", current)
        self.assertIn("120000-0002", current)

    def test_log_tail_cr_normalized_and_bounded(self):
        from utils import status as st
        with tempfile.TemporaryDirectory() as root:
            log = os.path.join(root, "job.log")
            with open(log, "wb") as f:
                f.write(b"[INFO] running mosaic3d on scene0046_00 ...\r"
                        b"progress\r[INFO] running mosaic3d on scene0084_01 ...\n")
            row = {"log": log}
            tail = st._log_tail(row, limit=128)
            self.assertIn("scene0084_01", tail)
            self.assertNotIn("\r", tail)

    def test_render_is_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            before = set(os.listdir(root))
            self._render(self._fake_res(), self._rows(), root,
                         cpu_cores={"job-20260910-120000-0001": 1.0})
            self.assertEqual(set(os.listdir(root)), before)

    def test_future_plan_without_runtime_or_progress(self):
        with tempfile.TemporaryDirectory() as root:
            text = self._render(self._fake_res(), self._rows(), root)
        future = text.split("FUTURE JOBS\n", 1)[1].split("PAST JOBS", 1)[0]
        self.assertIn("16 scenes x ?", future)
        self.assertNotIn("PROGRESS", future)
        self.assertNotRegex(future, r"\d+m\s+\d+\.\d+c")

    def test_eta_dash_without_two_timings(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            eval_root = os.path.join(root, "derived", "evaluations",
                                     "ScanNet20", "run-e")
            model_dir = os.path.join(eval_root, "mosaic3d")
            os.makedirs(model_dir)
            with open(os.path.join(eval_root, "run.json"), "w") as f:
                json.dump({"schema": 2, "benchmark": "ScanNet20",
                           "run_id": "run-e",
                           "scenes": ["scene0046_00", "scene0084_01"],
                           "methods": {"mosaic3d": {"parameters": {}}}}, f)
            with open(os.path.join(eval_root, "mosaic3d.tasks"), "w") as f:
                f.write("scene0046_00\n")
            with open(os.path.join(model_dir, "scene0046_00.timing.json"),
                      "w") as f:
                json.dump({"schema": 1, "kind": "prediction",
                           "scene_id": "scene0046_00", "run_id": "run-e",
                           "model": "mosaic3d", "elapsed_s": 60.0}, f)
            row = dict(self._rows()[0])
            row.update({
                "kind": "predict", "run_id": "run-e",
                "argv": ["python", "main.py", "--predict",
                         "--models", "mosaic3d", "--run-id", "run-e",
                         "--benchmark", "ScanNet20"]})
            text = self._render(self._fake_res(), [row], root,
                                cpu_cores={row["job_id"]: 1.0})
        current = text.split("CURRENT JOBS\n", 1)[1].split("FUTURE JOBS", 1)[0]
        self.assertIn("1/2", current)

    def test_main_status_delegates(self):
        import main as main_mod
        with mock.patch.object(sys, "argv", ["main.py", "--status"]), \
                mock.patch("utils.status.show_once") as show:
            main_mod.main()
        show.assert_called_once_with()

    def test_main_status_rejects_predict(self):
        import main as main_mod
        with mock.patch.object(sys, "argv",
                               ["main.py", "--status", "--predict",
                                "--models", "mosaic3d"]):
            with self.assertRaises(SystemExit):
                main_mod.main()

    def test_predict_detaches_by_default(self):
        import main as main_mod
        argv = ["main.py", "--predict", "--models", "mosaic3d",
                "--run-id", "run-x"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("jobs.submit", return_value="job-1") as submit:
            main_mod.main()
        self.assertEqual(submit.call_count, 1)
        _, kwargs = submit.call_args
        self.assertEqual(kwargs["kind"], "predict")
        self.assertEqual(kwargs["run_id"], "run-x")
        self.assertIn("--foreground", submit.call_args[0][0])

    def test_foreground_runs_inline(self):
        import main as main_mod
        argv = ["main.py", "--foreground", "--predict", "--models", "mosaic3d",
                "--run-id", "run-x"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("predict.runner.predict", return_value=[]) as predict, \
                mock.patch("jobs.submit", return_value="job-fg") as submit, \
                mock.patch("jobs._update_status"), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPELLBOOK_JOB_ID", None)
            main_mod.main()
        predict.assert_called_once()
        submit.assert_called_once()
        self.assertNotIn("SPELLBOOK_JOB_ID", os.environ)


if __name__ == "__main__":
    unittest.main()
