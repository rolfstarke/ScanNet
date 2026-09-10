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
                second = _submit(root)
            self.assertEqual(first, second)
            self.assertEqual(len(os.listdir(os.path.join(root, "derived", "jobs"))), 1)


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
            self.assertEqual(new_job["retry_of"], job_id)

    def test_retry_refuses_active(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = _submit(root)
            jobs._update_status(job_id, root, state="running")
            with mock.patch("jobs._unit_active", return_value=True):
                with self.assertRaises(ValueError):
                    jobs.retry_job(job_id, scannet_root=root)


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
            self.assertIn("GPU UTIL", text)
            self.assertIn("reserved", text)
            self.assertIn("unleased", text)
            self.assertIn("run-a", text)
            self.assertRegex(text, r"QUEUE\s+-\s+-\s+run-a")


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
            "job_id": "job-1", "kind": "predict", "run_id": "run-a",
            "state": "running", "submitted": now - 60, "started": now - 30,
            "ended": None, "exit_code": None, "branch": "master",
            "log": "/tmp/job-1/job.log",
        }]

    def test_render_snapshot_direct(self):
        from utils.status import render_snapshot
        with tempfile.TemporaryDirectory() as root:
            text = "\n".join(render_snapshot(
                self._fake_res(), [1], {"total": 100, "idle": 50},
                self._rows(), root))
        self.assertIn("cpu ", text)
        self.assertIn("reserved", text)
        self.assertIn("unleased", text)
        self.assertNotIn("job-1", text)
        self.assertIn("WAIT", text)

    def test_gpu_row_names_foreground_run(self):
        from utils.status import render_snapshot
        fake = self._fake_res()
        fake.gpu_leases.return_value = {1: 42}
        fake.process_info.return_value = {
            "pid": 42,
            "argv": ["spellbook/main.py", "--predict", "--models", "search3d",
                     "--run-id", "smoke-search3d"],
            "cwd": "/tmp/debug/prediction-search3d",
            "started": 1699999940.0,
            "job_id": None,
        }
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("utils.status.time.time", return_value=1700000000.0):
            text = "\n".join(render_snapshot(
                fake, [1], {"total": 100, "idle": 50}, [], root))
        self.assertIn("1m search3d     smoke-search3d", text)
        self.assertNotIn("python", text)

    def test_owned_job_is_not_repeated_as_waiting(self):
        from utils.status import render_snapshot
        fake = self._fake_res()
        fake.gpu_leases.return_value = {1: 42}
        fake.process_info.return_value = {
            "pid": 42,
            "argv": [],
            "cwd": None,
            "started": 1699999970.0,
            "job_id": "job-1",
        }
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("utils.status.time.time", return_value=1700000000.0):
            text = "\n".join(render_snapshot(
                fake, [1], {"total": 100, "idle": 50}, self._rows(), root))
        self.assertIn("run-a", text)
        self.assertNotIn("STATE    TIME", text)

    def test_recent_history_uses_process_runtime(self):
        from utils.status import render_snapshot
        rows = self._rows() + [{
            "job_id": "job-old", "kind": "predict", "run_id": "old-run",
            "state": "cancelled", "submitted": 1600000000.0,
            "started": 1699999010.0, "ended": 1699999020.0,
            "gpu_wait": 0.0, "exit_code": None, "branch": "master",
            "log": "/tmp/old.log",
        }, {
            "job_id": "job-lost", "kind": "predict", "run_id": "lost-run",
            "state": "lost", "submitted": 1699999030.0,
            "started": None, "ended": None, "gpu_wait": None,
            "exit_code": None, "branch": "master", "log": "/tmp/lost.log",
        }]
        with tempfile.TemporaryDirectory() as root:
            text = "\n".join(render_snapshot(
                self._fake_res(), [1], {"total": 100, "idle": 50}, rows, root))
        history = text.split("RECENT JOBS\n", 1)[1]
        self.assertRegex(history, r"CANCEL\s+10s\s+master\s+old-run")
        self.assertRegex(history, r"LOST\s+-\s+master\s+lost-run")
        self.assertNotIn("run-a", history)

    def test_recent_total_subtracts_gpu_wait(self):
        from utils.status import render_snapshot
        rows = [{
            "job_id": "job-old", "kind": "predict", "run_id": "old-run",
            "state": "succeeded", "submitted": 1600000000.0,
            "started": 1000.0, "ended": 1100.0, "gpu_wait": 40.0,
            "exit_code": 0, "branch": "master", "log": "/tmp/old.log",
        }]
        with tempfile.TemporaryDirectory() as root:
            text = "\n".join(render_snapshot(
                self._fake_res(), [1], {"total": 100, "idle": 50}, rows, root))
        self.assertRegex(text, r"DONE\s+1m\s+master\s+old-run")

    def test_recent_total_missing_wait_shows_dash(self):
        from utils.status import render_snapshot
        rows = [{
            "job_id": "job-old", "kind": "predict", "run_id": "old-run",
            "state": "succeeded", "submitted": 1600000000.0,
            "started": 1000.0, "ended": 1100.0,
            "exit_code": 0, "branch": "master", "log": "/tmp/old.log",
        }]
        with tempfile.TemporaryDirectory() as root:
            text = "\n".join(render_snapshot(
                self._fake_res(), [1], {"total": 100, "idle": 50}, rows, root))
        self.assertRegex(text, r"DONE\s+-\s+master\s+old-run")

    def test_recent_history_is_newest_20(self):
        from utils.status import render_snapshot
        rows = []
        for i in range(21):
            ended = 1700000000.0 + i
            rows.append({
                "job_id": f"job-{i}", "kind": "predict",
                "run_id": f"past-{i:02d}", "state": "succeeded",
                "submitted": 1600000000.0, "started": ended - 60,
                "ended": ended, "gpu_wait": 0.0, "exit_code": 0,
                "branch": "master",
                "log": f"/tmp/job-{i}.log",
            })
        with tempfile.TemporaryDirectory() as root:
            text = "\n".join(render_snapshot(
                self._fake_res(), [1], {"total": 100, "idle": 50}, rows, root))
        history = text.split("RECENT JOBS\n", 1)[1]
        self.assertEqual(history.count("past-"), 20)
        self.assertNotIn("past-00", history)
        self.assertLess(history.index("past-20"), history.index("past-19"))

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
                mock.patch("jobs.submit") as submit:
            main_mod.main()
        predict.assert_called_once()
        submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
