"""Drop one site's queued background jobs before a job that needs the site idle.

`bench ready-for-migration` refuses to pass while any RQ job whose id starts
with the site name is queued or started. On a bench that hosts many sites the
queue is deep enough that a site's own scheduled jobs wait longer than the
300s gate, so a Rename or Migrate fails although nothing is wrong with the
site.

Run right after maintenance mode is switched on -- which stops the scheduler
adding more -- the purge turns that gate into a quick confirmation instead of
a long wait.
"""

from __future__ import annotations

import json
from shlex import quote

# Lets the result line be found among whatever else the container prints.
RESULT_MARKER = "__AGENT_PURGE_RESULT__"

# Seconds to let jobs that are already executing finish by themselves.
DEFAULT_GRACE_SECONDS = 60

# List entries handled per Redis script call. A call blocks Redis while it
# runs, so this keeps each one to a few tens of milliseconds.
BATCH_SIZE = 10000

BENCH_PYTHON = "/home/frappe/frappe-bench/env/bin/python"

# After a clean purge the readiness gate passes first time, so a long window
# buys nothing. Without one, keep the original 300s: a failed purge must never
# leave a job with less time than it had before.
WAIT_TIMEOUT_AFTER_PURGE = 60
WAIT_TIMEOUT_WITHOUT_PURGE = 300

# Executed inside the bench container (stdin), where Frappe and RQ live.
PURGE_SCRIPT = r'''
import json
import sys
import time

import frappe
from frappe.utils.background_jobs import create_job_id, get_queue, get_queue_list, get_redis_conn
from rq.command import send_stop_job_command
from rq.exceptions import InvalidJobOperation, NoSuchJobError
from rq.job import Job, JobStatus
from rq.worker import Worker

SITE = sys.argv[1]
GRACE_SECONDS = int(sys.argv[2])
STOP_RUNNING = sys.argv[3] == "1"
BATCH = int(sys.argv[4])
MARKER = sys.argv[5]

HASH_PREFIX = Job.redis_job_namespace_prefix

# Pops the first N entries of a queue list. Entries owned by the site are
# deleted together with their job hash; every other entry goes back on the
# tail, so one full pass leaves the other sites' jobs in the same relative
# order. Doing this in Redis is linear in the queue length, whereas removing
# jobs one by one (LREM per job) is quadratic.
LUA_DROP = """
local key, hash_prefix, site_prefix = KEYS[1], ARGV[1], ARGV[2]
local count = tonumber(ARGV[3])
local size = redis.call("LLEN", key)
if count > size then count = size end
local dropped = 0
for _ = 1, count do
    local id = redis.call("LPOP", key)
    if not id then break end
    if string.sub(id, 1, string.len(site_prefix)) == site_prefix then
        redis.call("DEL", hash_prefix .. id, hash_prefix .. id .. ":dependents")
        dropped = dropped + 1
    else
        redis.call("RPUSH", key, id)
    end
end
return {dropped, count}
"""


def job_namespace():
    """The "<site><separator>" every job id of SITE starts with on this bench.

    The separator is not the same across Frappe versions ("site::id" on v15,
    "site||id" on v16), so ask Frappe instead of assuming one: a wrong guess
    matches nothing, purges nothing, and still reports success. The separator
    is part of the prefix on purpose -- a bare startswith(site) would also
    catch "shop-12" when purging "shop-1".
    """
    namespace = create_job_id("x")[:-1]
    well_formed = (
        namespace.startswith(SITE) and len(namespace) > len(SITE) and not namespace[-1].isalnum()
    )
    if not well_formed:
        raise RuntimeError(f"Unexpected job id namespace {namespace!r} for site {SITE!r}")
    return namespace


def owned_ids(registry, namespace):
    return [job_id for job_id in registry.get_job_ids() if job_id and job_id.startswith(namespace)]


def forget(connection, registry, job_id):
    registry.remove(job_id)
    connection.delete(HASH_PREFIX + job_id, HASH_PREFIX + job_id + ":dependents")


def drain_list(drop, queue, namespace):
    """Pass over every entry the queue holds right now, exactly once.

    Survivors are re-pushed on the tail, so the number of entries rotated must
    equal the length at the start: fewer strands this site's jobs behind
    survivors already moved, more re-reads survivors.
    """
    left = queue.count
    dropped = 0
    while left > 0:
        batch_dropped, batch_seen = drop(
            keys=[queue.key], args=[HASH_PREFIX, namespace, min(BATCH, left)]
        )
        if not batch_seen:
            break
        dropped += batch_dropped
        left -= batch_seen
    return dropped


def drop_from_registry(connection, registry, namespace):
    """Deferred and scheduled jobs never sit in the list but become queued later."""
    ids = owned_ids(registry, namespace)
    for job_id in ids:
        forget(connection, registry, job_id)
    return len(ids)


def reap_dead_started(connection, registry, namespace, live_workers):
    """Remove this site's started entries that nothing is actually running.

    RQ expires those on its own only after timeout + 60s (26 minutes for the
    long queue); until then the readiness check sees a job that is not there.
    """
    registry.cleanup()
    reaped = []
    for job_id in owned_ids(registry, namespace):
        try:
            job = Job.fetch(job_id, connection=connection)
            status = job.get_status()
        except (NoSuchJobError, AttributeError):
            forget(connection, registry, job_id)
            reaped.append(job_id)
            continue
        worker = getattr(job, "worker_name", None)
        if status != JobStatus.STARTED or (worker and worker not in live_workers):
            forget(connection, registry, job_id)
            reaped.append(job_id)
    return reaped


def running_now(queues, namespace):
    ids = set()
    for queue in queues:
        ids.update(owned_ids(queue.started_job_registry, namespace))
    return sorted(ids)


def main():
    frappe.init(SITE)
    namespace = job_namespace()
    connection = get_redis_conn()
    drop = connection.register_script(LUA_DROP)
    queues = [get_queue(name) for name in get_queue_list()]
    workers = {worker.name for worker in Worker.all(connection=connection)}

    report = {
        "site": SITE,
        "namespace": namespace,
        "grace_seconds": GRACE_SECONDS,
        "stop_running_requested": STOP_RUNNING,
        "purged": {},
        "deferred_purged": {},
        "scheduled_purged": {},
        "dead_started_reaped": [],
        "force_stopped": [],
    }

    for queue in queues:
        report["purged"][queue.name] = drain_list(drop, queue, namespace)
        report["deferred_purged"][queue.name] = drop_from_registry(
            connection, queue.deferred_job_registry, namespace
        )
        report["scheduled_purged"][queue.name] = drop_from_registry(
            connection, queue.scheduled_job_registry, namespace
        )
        report["dead_started_reaped"].extend(
            reap_dead_started(connection, queue.started_job_registry, namespace, workers)
        )

    report["running_at_start"] = running_now(queues, namespace)

    # A job that is mid-execution can still enqueue children, so give the few
    # that are running time to finish, then sweep once more for what they left.
    deadline = time.time() + GRACE_SECONDS
    while time.time() < deadline and running_now(queues, namespace):
        time.sleep(2)

    if STOP_RUNNING:
        for job_id in running_now(queues, namespace):
            try:
                send_stop_job_command(connection=connection, job_id=job_id)
                report["force_stopped"].append(job_id)
            except (InvalidJobOperation, NoSuchJobError):
                pass
        time.sleep(2)

    for queue in queues:
        report["purged"][queue.name] += drain_list(drop, queue, namespace)

    report["running_at_end"] = running_now(queues, namespace)
    report["total_purged"] = sum(report["purged"].values())
    print(MARKER + json.dumps(report, sort_keys=True))


try:
    main()
finally:
    frappe.destroy()
'''


def build_purge_command(site: str, grace_seconds: int, force_stop: bool) -> str:
    """The script is fed through stdin, so only plain values go on the command line."""
    return " ".join(
        [
            BENCH_PYTHON,
            "-",
            quote(site),
            quote(str(int(grace_seconds))),
            "1" if force_stop else "0",
            quote(str(BATCH_SIZE)),
            quote(RESULT_MARKER),
        ]
    )


def purge_was_clean(result: dict | None) -> bool:
    """True when the shortened readiness window is safe to use.

    A purge that ran but left jobs executing has not cleared the queue, so the
    caller keeps the full original window instead of failing faster than it
    did before the purge existed.
    """
    return bool(result) and not result.get("running_at_end")


def parse_purge_result(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER) :])
    raise ValueError(f"Job purge did not report a result. Output tail:\n{output[-2000:]}")
