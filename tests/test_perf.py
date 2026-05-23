import time

from pi_fpv_companion.perf import PerfMonitor, PiBudget


def test_empty_stats_safe():
    m = PerfMonitor()
    s = m.stats()
    assert s.n_ticks == 0
    assert s.tick_p50_ms == 0.0


def test_records_tick_times():
    m = PerfMonitor()
    for _ in range(5):
        t0 = m.tick_start()
        time.sleep(0.005)
        m.tick_end(t0)
    s = m.stats()
    assert s.n_ticks == 5
    assert s.tick_p50_ms >= 4.0          # ~5ms sleeps, allowing some scheduler slop


def test_report_contains_verdict_line():
    m = PerfMonitor(PiBudget(max_tick_ms=1000.0, max_rss_mb=10000.0, pi_scale_factor=6.0))
    for _ in range(3):
        t0 = m.tick_start()
        time.sleep(0.001)
        m.tick_end(t0)
    rep = m.report()
    assert "VERDICT" in rep
    assert "fits Pi budget" in rep


def test_over_budget_flagged_in_report():
    # Tight budget; sleep 20ms ticks; pi scaling 6 -> ~120ms estimate >> budget
    m = PerfMonitor(PiBudget(max_tick_ms=10.0, max_rss_mb=10000.0, pi_scale_factor=6.0))
    for _ in range(20):
        t0 = m.tick_start()
        time.sleep(0.02)
        m.tick_end(t0)
    rep = m.report()
    assert "OVER tick budget on Pi" in rep
