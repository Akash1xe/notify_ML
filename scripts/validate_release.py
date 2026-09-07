from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from app.core.config import get_settings
from app.evaluation.models import stable_fingerprint
from app.observability.environment import EnvironmentValidator
from app.release.runner import ReleaseValidationRunner

APP_VERSION='1.0.0'

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--official', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--json', action='store_true'); p.add_argument('--output'); a=p.parse_args()
    settings=get_settings(); snapshot=EnvironmentValidator(settings).config_snapshot(application_version=APP_VERSION)
    runner=ReleaseValidationRunner(release_version=APP_VERSION,pipeline_fingerprint='notify-v1-pipeline',production_config_fingerprint=snapshot.config_fingerprint)
    env=EnvironmentValidator(settings).validate(deep=False)
    runner.add_gate('environment','Environment readiness',lambda:(env.ready,env.model_dump(mode='json')))
    runner.add_gate('configuration','Production configuration',lambda:(settings.processor_mode in {'document','full','screenshots','semantic','transcription','candidates','analysis','ingestion','fake'},{'processor_mode':settings.processor_mode}))
    root=Path(__file__).resolve().parents[1]
    if not a.smoke:
        runner.run_command_gate('backend_tests','Backend test suite',['pytest'],cwd=root,timeout=900)
        runner.run_command_gate('compileall','Python compile gate',['python','-m','compileall','-q','app','scripts','tests'],cwd=root,timeout=300)
        frontend=root/'frontend'
        if frontend.exists():
            runner.run_command_gate('frontend_typecheck','Frontend typecheck',['npm','run','typecheck'],cwd=frontend,timeout=300)
            runner.run_command_gate('frontend_lint','Frontend lint',['npm','run','lint'],cwd=frontend,timeout=300)
            runner.run_command_gate('frontend_tests','Frontend tests',['npm','run','test'],cwd=frontend,timeout=300)
            runner.run_command_gate('frontend_build','Frontend build',['npm','run','build'],cwd=frontend,timeout=300)
    else:
        runner.add_gate('smoke','Release smoke',lambda:(True,{'mode':'smoke'}))
    report=runner.finalize(known_limitations=['Real-model/real-lecture release measurements are hardware and media dependent and remain local validation gates.'])
    payload=report.model_dump(mode='json')|{'release_fingerprint':report.fingerprint}
    if a.output:
        out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(payload,indent=2) if a.json else f'ReleaseReadiness: {report.release_ready.value}')
    return 0 if report.release_ready.value in {'READY','READY_WITH_WARNINGS'} else 2
if __name__=='__main__': raise SystemExit(main())
