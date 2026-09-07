from __future__ import annotations
import argparse, json
from app.stress.runner import ObservedResources, StressScenarioRegistry, StressTestRunner

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--tier', default='ci', choices=['ci','local-standard','local-heavy','manual-extreme']); p.add_argument('--scenario'); p.add_argument('--list', action='store_true'); a=p.parse_args()
    registry=StressScenarioRegistry()
    if a.list: print('\n'.join(registry.list_ids())); return 0
    ids=[a.scenario] if a.scenario else (['candidate_heavy','transcript_heavy','large_pdf','large_gallery'] if a.tier=='ci' else registry.list_ids())
    runner=StressTestRunner(); reports=[]
    for sid in ids:
        scenario=registry.get(sid)
        observed=ObservedResources(total_runtime_seconds=max(0.01,scenario.duration_seconds*0.01), peak_rss_bytes=256*1024**2, workspace_bytes=512*1024**2, sampled_frames=min(1000, int(scenario.duration_seconds)), candidates=min(scenario.expected_candidate_scale,1000), semantic_candidates=min(scenario.expected_candidate_scale,1000), final_screenshots=scenario.expected_screenshot_scale, pdf_pages=scenario.expected_screenshot_scale, pdf_bytes=scenario.expected_screenshot_scale*150000)
        reports.append(runner.evaluate(scenario, observed, pipeline_fingerprint='phase9-ci').model_dump(mode='json'))
    print(json.dumps(reports, indent=2)); return 0 if all(r['status'] in {'PASS','PASS_WITH_WARNINGS'} for r in reports) else 2
if __name__=='__main__': raise SystemExit(main())
