from pathlib import Path
import json

from app.evaluation.benchmark import BenchmarkRunner, DatasetRepository, EndToEndQualityEvaluator, TemporalMatcher
from app.evaluation.calibration import CalibrationParameterRegistry, CalibrationRunner, RecommendationStatus
from app.evaluation.models import BenchmarkConfig, GroundTruthAnnotation, PredictionLevel, PredictionRecord, QualityBaselineReport

ROOT = Path(__file__).resolve().parents[1]


def load_bundle():
    repo = DatasetRepository(ROOT / 'evaluation')
    manifest = repo.load_manifest('datasets/ci_manifest.json')
    annotations = {sample.sample_id: repo.load_annotation(sample.annotation_path) for sample in manifest.samples}
    raw = json.loads((ROOT / 'evaluation/fixtures/reference/ci_predictions.json').read_text())
    predictions = {
        sid: {PredictionLevel(level): [PredictionRecord.model_validate(v) for v in values] for level, values in stages.items()}
        for sid, stages in raw.items()
    }
    return repo, manifest, annotations, predictions


def test_dataset_and_benchmark_smoke():
    repo, manifest, annotations, predictions = load_bundle()
    assert repo.validate(manifest) == []
    report = BenchmarkRunner().run(dataset=manifest, annotations=annotations, predictions=predictions, config=BenchmarkConfig(prediction_level=PredictionLevel.FINAL_SCREENSHOT))
    assert report.aggregate.precision == 1.0
    assert report.aggregate.recall == 1.0


def test_end_to_end_quality_funnel_and_loss_attribution():
    _, manifest, annotations, predictions = load_bundle()
    report = EndToEndQualityEvaluator().evaluate(dataset=manifest, annotations=annotations, predictions=predictions, pipeline_fingerprint='test', lock=True)
    assert report.status == 'LOCKED'
    assert report.required_visual_state_recall == 1.0
    assert report.stage_loss_counts == {}

    altered = {sid: {level: list(values) for level, values in stages.items()} for sid, stages in predictions.items()}
    altered['writing_001'][PredictionLevel.SEMANTIC] = []
    altered['writing_001'][PredictionLevel.FINAL_SCREENSHOT] = []
    altered['writing_001'][PredictionLevel.DOCUMENT] = []
    failed = EndToEndQualityEvaluator().evaluate(dataset=manifest, annotations=annotations, predictions=altered, pipeline_fingerprint='test')
    assert failed.stage_loss_counts['SEMANTIC_FALSE_REJECTION'] == 1


def test_temporal_match_is_one_to_one_and_boundaries_inclusive():
    annotation = GroundTruthAnnotation.model_validate({
        'sample_id': 'x', 'review_status': 'LOCKED',
        'states': [
            {'state_id':'a','target_timestamp_seconds':1,'acceptable_start_seconds':1,'acceptable_end_seconds':2},
            {'state_id':'b','target_timestamp_seconds':2,'acceptable_start_seconds':1,'acceptable_end_seconds':2},
        ]
    })
    preds=[PredictionRecord(prediction_id='p',timestamp_seconds=1)]
    matches, _, fns = TemporalMatcher().match(annotation.states,preds)
    assert len(matches)==1
    assert len(fns)==1


def test_calibration_registry_and_guardrails():
    reg=CalibrationParameterRegistry()
    values=reg.local_sweep('min_stable_duration_seconds')
    assert 2.0 in values and values==sorted(values)
    runner=CalibrationRunner(reg)
    profiles=runner.generate_local_profiles(parameter_name='min_stable_duration_seconds',base_pipeline_fingerprint='base',max_runs=3)
    assert len(profiles)==3
    base=QualityBaselineReport(dataset_fingerprint='d',pipeline_fingerprint='p',overall={},candidate={},semantic={},final_screenshot={},document={},required_visual_state_recall=.9,final_visual_note_precision=.9,final_visual_note_f1=.9)
    better=base.model_copy(update={'required_visual_state_recall':.92,'final_visual_note_precision':.9,'final_visual_note_f1':.91})
    assert runner.compare(base,better).recommendation is RecommendationStatus.ACCEPT
    regressed=better.model_copy(update={'final_visual_note_precision':.7})
    assert runner.compare(base,regressed).recommendation is not RecommendationStatus.ACCEPT
