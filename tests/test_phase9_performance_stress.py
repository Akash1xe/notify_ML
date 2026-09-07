from pathlib import Path
import tempfile

from app.performance.models import ArtifactMetrics, HardwareSummary, PerformanceMode, PerformanceRun, StagePerformanceMetrics
from app.performance.profiler import StageTimer, calculate_workspace_size, compare_performance, streaming_sha256
from app.stress.runner import ObservedResources, ResourceGrowthDetector, StressScenarioRegistry, StressTestRunner


def make_run(runtime: float, memory: int, disk: int) -> PerformanceRun:
    return PerformanceRun(run_id='r',sample_id='s',mode=PerformanceMode.COLD_RUN,pipeline_fingerprint='p',performance_config_fingerprint='c',hardware=HardwareSummary(cpu_count_logical=1,ram_total_bytes=1,platform='x',python_version='3.11'),stage_metrics=[StagePerformanceMetrics(stage='x',duration_seconds=runtime,rss_memory_peak_bytes=memory)],artifact_metrics=ArtifactMetrics(workspace_bytes=disk),video_duration_seconds=60)


def test_stage_timer_and_performance_comparison():
    timer=StageTimer(); timer.start('x'); assert timer.stop('x')>=0
    baseline=make_run(10,1000,1000); candidate=make_run(9,1100,1100)
    comp=compare_performance(baseline,candidate)
    assert comp.runtime_improvement_ratio==0.1
    assert comp.scorecard.accepted


def test_workspace_size_and_streaming_hash_are_safe():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); (root/'a').write_bytes(b'abc')
        assert calculate_workspace_size(root)==3
        assert len(streaming_sha256(root/'a'))==64


def test_stress_scenarios_and_limits():
    registry=StressScenarioRegistry(); scenario=registry.get('candidate_heavy')
    report=StressTestRunner().evaluate(scenario,ObservedResources(total_runtime_seconds=1,peak_rss_bytes=1,candidates=999,semantic_candidates=999,final_screenshots=200,pdf_pages=200,pdf_bytes=1),pipeline_fingerprint='p')
    assert report.status.value in {'PASS','PASS_WITH_WARNINGS'}
    bad=StressTestRunner().evaluate(scenario,ObservedResources(candidates=1001),pipeline_fingerprint='p')
    assert bad.status.value=='LIMIT_EXCEEDED'


def test_growth_detector_and_possible_leak():
    analysis=ResourceGrowthDetector.classify(2,2.1,'rss')
    assert analysis.classification.value=='APPROX_LINEAR'
    assert ResourceGrowthDetector.possible_leak([100,125,150,180,220])
    assert not ResourceGrowthDetector.possible_leak([100,102,101,103,102])
