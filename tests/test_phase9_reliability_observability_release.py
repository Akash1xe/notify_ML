from pathlib import Path
import json, tempfile

from app.observability.environment import SensitiveValueRedactor
from app.observability.models import EventName, ObservabilityEvent
from app.observability.diagnostics import JsonLineEventSink
from app.release.runner import ReleaseValidationRunner
from app.reliability.runner import PipelineRecoveryCoordinator, ReliabilityScenarioRegistry, ReliabilityTestRunner


def test_reliability_registry_and_ci_simulation():
    registry=ReliabilityScenarioRegistry()
    assert 'pdf_corrupt' in registry.list_ids()
    outcome,score=ReliabilityTestRunner(registry).simulate('pdf_corrupt')
    assert outcome.failure_detected and score.passed
    assert outcome.resume_stage=='PDF_GENERATION'


def test_recovery_plan_earliest_invalid_and_temp_cleanup():
    plan=PipelineRecoveryCoordinator().plan({'PHASE_2':True,'PHASE_3':True,'PHASE_4':False,'PHASE_5':True})
    assert plan.last_valid_phase=='PHASE_3'
    assert plan.resume_phase=='PHASE_4'
    assert 'PHASE_5' in plan.checkpoints_to_clear
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); (root/'x.tmp').write_text('x'); (root/'keep.json').write_text('{}')
        removed=PipelineRecoveryCoordinator.cleanup_temp(root)
        assert removed==['x.tmp'] and (root/'keep.json').exists()


def test_redaction_and_jsonl_partial_line_tolerance():
    payload=SensitiveValueRedactor.redact({'authorization':'Bearer x','nested':{'api_key':'secret'},'ok':'value'})
    assert payload['authorization']=='***REDACTED***' and payload['ok']=='value'
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'events.jsonl'; sink=JsonLineEventSink(path)
        sink.emit(ObservabilityEvent(event_name=EventName.JOB_STARTED,job_id='j'))
        with path.open('a') as h: h.write('{partial')
        assert len(sink.read())==1


def test_release_runner_required_gate_and_fingerprint():
    runner=ReleaseValidationRunner(release_version='1.0.0',pipeline_fingerprint='p',production_config_fingerprint='c')
    runner.add_gate('ok','OK',lambda:(True,{}))
    report=runner.finalize()
    assert report.release_ready.value=='READY'
    fp=report.fingerprint
    runner2=ReleaseValidationRunner(release_version='1.0.0',pipeline_fingerprint='p',production_config_fingerprint='c')
    runner2.add_gate('ok','OK',lambda:(True,{}))
    assert runner2.finalize().fingerprint==fp
    runner3=ReleaseValidationRunner(release_version='1.0.0',pipeline_fingerprint='p',production_config_fingerprint='c')
    runner3.add_gate('bad','Bad',lambda:(False,{}))
    assert runner3.finalize().release_ready.value=='NOT_READY'
