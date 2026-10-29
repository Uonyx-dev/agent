from __future__ import annotations

import inspect
import json
import os
import re
import unittest
from unittest.mock import MagicMock, patch

from agent.job_purge import (
    PURGE_SCRIPT,
    RESULT_MARKER,
    BATCH_SIZE,
    WAIT_TIMEOUT_AFTER_PURGE,
    WAIT_TIMEOUT_WITHOUT_PURGE,
    build_purge_command,
    parse_purge_result,
    purge_was_clean,
)

REDIS_URL = os.environ.get("AGENT_TEST_REDIS_URL", "redis://127.0.0.1:11000")
TEST_QUEUE_KEY = "agent:test:job_purge:queue"


def extract_lua_sweep() -> str:
    """The Lua script the container runs, pulled out of PURGE_SCRIPT.

    The sweep mutates a live queue shared with every other site on the bench,
    so it is tested directly rather than through a container.
    """
    match = re.search(r'LUA_DROP = """(.*?)"""', PURGE_SCRIPT, re.S)
    assert match, "LUA_DROP script not found in PURGE_SCRIPT"
    return match.group(1)


def get_redis_connection():
    """A real Redis if one is reachable, otherwise fakeredis (needs lupa), else skip."""
    try:
        import redis as redis_module
    except ImportError:
        raise unittest.SkipTest("redis client not installed") from None

    connection = redis_module.from_url(REDIS_URL, decode_responses=True)
    try:
        connection.ping()
        return connection
    except Exception:
        pass

    try:
        import fakeredis

        connection = fakeredis.FakeStrictRedis(decode_responses=True)
        connection.register_script("return 1")()  # needs lupa
        return connection
    except Exception as error:
        raise unittest.SkipTest(f"no redis at {REDIS_URL} and no fakeredis+lupa: {error}") from None


def run_sweep(queue: list[str], site_prefix: str, budget: int = BATCH_SIZE):
    """Execute the real Lua sweep, exactly as the container does."""
    connection = get_redis_connection()

    job_keys = [f"rq:job:{job_id}" for job_id in queue]
    job_keys += [f"rq:job:{job_id}:dependents" for job_id in queue]
    connection.delete(TEST_QUEUE_KEY, *job_keys) if job_keys else connection.delete(TEST_QUEUE_KEY)
    try:
        if queue:
            connection.rpush(TEST_QUEUE_KEY, *queue)
            for job_id in queue:
                connection.set(f"rq:job:{job_id}", "payload")
                connection.sadd(f"rq:job:{job_id}:dependents", "child")

        sweep = connection.register_script(extract_lua_sweep())
        total_purged = 0
        remaining = len(queue)
        while remaining > 0:
            purged, rotated = sweep(
                keys=[TEST_QUEUE_KEY],
                args=["rq:job:", site_prefix, min(budget, remaining)],
            )
            if not rotated:
                break
            total_purged += purged
            remaining -= rotated

        survivors = connection.lrange(TEST_QUEUE_KEY, 0, -1)
        deleted = [key for key in job_keys if not connection.exists(key)]
        return total_purged, survivors, deleted
    finally:
        connection.delete(TEST_QUEUE_KEY, *job_keys) if job_keys else connection.delete(TEST_QUEUE_KEY)


class TestPurgeCommand(unittest.TestCase):
    def test_site_name_and_options_are_shell_quoted_into_the_command(self):
        command = build_purge_command("site-1.example.com", 60, force_stop=False)
        self.assertIn("site-1.example.com", command)
        self.assertTrue(command.startswith("/home/frappe/frappe-bench/env/bin/python -"))

    def test_force_stop_is_passed_as_an_explicit_flag_and_defaults_off(self):
        self.assertIn(" 0 ", build_purge_command("site.example.com", 60, force_stop=False))
        self.assertIn(" 1 ", build_purge_command("site.example.com", 60, force_stop=True))

    def test_grace_seconds_is_coerced_to_an_integer_so_it_cannot_inject_arguments(self):
        command = build_purge_command("site.example.com", "60", force_stop=False)
        self.assertIn(" 60 ", command)

    def test_result_is_parsed_from_the_marker_line_ignoring_other_container_output(self):
        output = "WARN: deprecation\n" + RESULT_MARKER + json.dumps({"total_purged": 52000}) + "\nbye\n"
        self.assertEqual(parse_purge_result(output)["total_purged"], 52000)

    def test_missing_marker_raises_with_the_output_tail_for_debugging(self):
        with self.assertRaises(ValueError) as error:
            parse_purge_result("Traceback: redis connection refused")
        self.assertIn("redis connection refused", str(error.exception))


class TestLuaSweep(unittest.TestCase):
    """The sweep must purge only the target site and preserve everyone else.

    Run for both job-id separators: Frappe v15 builds "site::id", v16 "site||id".
    """

    SEPARATORS = ("::", "||")

    def test_only_the_target_sites_jobs_are_purged(self):
        for sep in self.SEPARATORS:
            with self.subTest(sep=sep):
                queue = [f"a.example.com{sep}1", f"b.example.com{sep}1", f"a.example.com{sep}2"]
                purged, survivors, _ = run_sweep(queue, f"a.example.com{sep}")
                self.assertEqual(purged, 2)
                self.assertEqual(survivors, [f"b.example.com{sep}1"])

    def test_a_site_whose_name_prefixes_another_is_not_purged_with_it(self):
        # frappe's own any_job_pending uses a bare startswith(site), so purging
        # workspace-138 must not take workspace-1380 with it.
        for sep in self.SEPARATORS:
            with self.subTest(sep=sep):
                queue = [f"workspace-138.example.com{sep}1", f"workspace-1380.example.com{sep}1"]
                purged, survivors, _ = run_sweep(queue, f"workspace-138.example.com{sep}")
                self.assertEqual(purged, 1)
                self.assertEqual(survivors, [f"workspace-1380.example.com{sep}1"])

    def test_the_wrong_separator_purges_nothing(self):
        # This is the failure mode the derived prefix exists to prevent: a v15
        # prefix against v16 job ids silently matches nothing.
        queue = ["a.example.com||1", "a.example.com||2"]
        purged, survivors, _ = run_sweep(queue, "a.example.com::")
        self.assertEqual(purged, 0)
        self.assertEqual(survivors, queue)

    def test_surviving_jobs_keep_their_original_order(self):
        queue = [f"other.example.com||{i}" for i in range(10)]
        queue.insert(5, "mine.example.com||x")
        purged, survivors, _ = run_sweep(queue, "mine.example.com||")
        self.assertEqual(purged, 1)
        self.assertEqual(survivors, [f"other.example.com||{i}" for i in range(10)])

    def test_order_is_preserved_across_a_chunked_sweep(self):
        queue = []
        for i in range(50):
            queue.append(f"other.example.com||{i}")
            queue.append(f"mine.example.com||{i}")
        purged, survivors, _ = run_sweep(queue, "mine.example.com||", budget=7)
        self.assertEqual(purged, 50)
        self.assertEqual(survivors, [f"other.example.com||{i}" for i in range(50)])

    def test_purged_jobs_have_their_hash_and_dependents_keys_deleted(self):
        _, _, deleted = run_sweep(["mine.example.com||x"], "mine.example.com||")
        self.assertIn("rq:job:mine.example.com||x", deleted)
        self.assertIn("rq:job:mine.example.com||x:dependents", deleted)

    def test_an_empty_queue_is_a_no_op(self):
        self.assertEqual(run_sweep([], "mine.example.com||"), (0, [], []))

    def test_a_queue_with_no_jobs_for_this_site_is_left_untouched(self):
        queue = [f"other.example.com||{i}" for i in range(20)]
        purged, survivors, deleted = run_sweep(queue, "mine.example.com||", budget=3)
        self.assertEqual((purged, survivors, deleted), (0, queue, []))


class TestPrefixIsDerivedFromFrappe(unittest.TestCase):
    """The job-id separator must come from the bench's own Frappe, never a literal."""

    def _derive(self, create_job_id, site="a.example.com"):
        match = re.search(r"def job_namespace\(\):.*?return namespace\n", PURGE_SCRIPT, re.S)
        self.assertTrue(match, "job_namespace not found in PURGE_SCRIPT")
        namespace = {"SITE": site, "create_job_id": create_job_id}
        exec(match.group(0), namespace)
        return namespace["job_namespace"]()

    def test_v16_separator(self):
        self.assertEqual(self._derive(lambda job_id: f"a.example.com||{job_id}"), "a.example.com||")

    def test_v15_separator(self):
        self.assertEqual(self._derive(lambda job_id: f"a.example.com::{job_id}"), "a.example.com::")

    def test_a_namespace_that_is_not_site_plus_separator_is_refused(self):
        for bad in ("a.example.comx", "other.example.com||x", "a.example.com"):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                self._derive(lambda job_id, bad=bad: bad)

    def test_the_script_has_no_hard_coded_separator(self):
        self.assertIn('create_job_id("x")', PURGE_SCRIPT)
        self.assertNotIn('SITE + "::"', PURGE_SCRIPT)
        self.assertNotIn('SITE + "||"', PURGE_SCRIPT)


class TestWaitWindow(unittest.TestCase):
    def test_a_clean_purge_earns_the_short_window(self):
        self.assertTrue(purge_was_clean({"total_purged": 52000, "running_at_end": []}))
        self.assertTrue(purge_was_clean({"total_purged": 0}))

    def test_a_purge_that_did_not_run_keeps_the_full_window(self):
        self.assertFalse(purge_was_clean(None))
        self.assertFalse(purge_was_clean({}))

    def test_jobs_still_running_after_the_purge_keep_the_full_window(self):
        self.assertFalse(purge_was_clean({"total_purged": 10, "running_at_end": ["site||job"]}))

    def test_the_fallback_window_is_never_shorter_than_the_original(self):
        self.assertEqual(WAIT_TIMEOUT_WITHOUT_PURGE, 300, "must match the pre-existing timeout")
        self.assertLess(WAIT_TIMEOUT_AFTER_PURGE, WAIT_TIMEOUT_WITHOUT_PURGE)

    def test_callers_that_do_not_purge_keep_the_original_window(self):
        from agent.site import Site

        default = inspect.signature(Site.wait_till_ready).parameters["timeout"].default
        self.assertEqual(default, WAIT_TIMEOUT_WITHOUT_PURGE)


class TestPurgeIsNonFatal(unittest.TestCase):
    """A failed purge must degrade to the old behaviour, never abort a job."""

    def _site(self):
        from agent.site import Site

        site = Site.__new__(Site)
        site.name = "site.example.com"
        return site

    def test_a_successful_purge_returns_its_result(self):
        from agent.site import Site

        result = {"total_purged": 52000, "running_at_end": []}
        with patch.object(Site, "purge_pending_jobs", return_value=result):
            self.assertEqual(self._site().purge_pending_jobs_if_possible(), result)

    def test_a_purge_that_raises_is_swallowed_and_returns_none(self):
        from agent.site import Site

        with patch.object(Site, "purge_pending_jobs", side_effect=RuntimeError("container gone")):
            self.assertIsNone(self._site().purge_pending_jobs_if_possible())


class TestStandbySites(unittest.TestCase):
    def test_pooled_sites_are_recognised_by_press_naming(self):
        from agent.bench import is_standby_site

        self.assertTrue(is_standby_site("standby-00000081.uonyx.com"))

    def test_claimed_and_ordinary_sites_are_not_standby(self):
        from agent.bench import is_standby_site

        for name in ("qatestcycle79.uonyx.com", "acme.uonyx.com", "my-standby-1.uonyx.com"):
            with self.subTest(name=name):
                self.assertFalse(is_standby_site(name))

    def _run_new_site(self, name):
        """Run Bench.new_site's body with the heavy collaborators mocked."""
        from agent import bench as bench_module

        bench = bench_module.Bench.__new__(bench_module.Bench)
        bench.bench_new_site = MagicMock()
        bench.setup_nginx = MagicMock()
        bench.server = MagicMock()
        site = MagicMock()
        with patch.object(bench_module, "Site", return_value=site):
            bench_module.Bench.new_site.__wrapped__(bench, name, {}, [], "root-pw", "admin-pw")
        return site

    def test_a_new_standby_site_keeps_its_scheduler_off(self):
        site = self._run_new_site("standby-00000099.uonyx.com")
        site.enable_scheduler.assert_not_called()

    def test_a_new_ordinary_site_still_gets_its_scheduler(self):
        site = self._run_new_site("customer.uonyx.com")
        site.enable_scheduler.assert_called_once()

    def test_renaming_a_claimed_site_turns_the_scheduler_on(self):
        # The other half of the contract: a standby is only ever left off
        # because rename_site_job switches it on at the end of the claim.
        from agent import bench as bench_module

        source = inspect.getsource(bench_module.Bench.rename_site_job.__wrapped__)
        self.assertIn("site.enable_scheduler()", source)
        self.assertLess(source.index("site.disable_maintenance_mode()"), source.index("site.enable_scheduler()"))


if __name__ == "__main__":
    unittest.main()
