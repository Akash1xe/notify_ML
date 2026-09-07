from __future__ import annotations

from app.core.config import AppSettings
from app.core.exceptions import SemanticAnalysisError
from app.semantic_analysis.models import SEMANTIC_PROMPT_VERSION, SemanticInputRecord, TemporalVisualContext
from app.video_analysis.fingerprints import stable_hash


class SemanticPromptBuilder:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @property
    def version(self) -> str:
        return SEMANTIC_PROMPT_VERSION

    def build(self, semantic_input: SemanticInputRecord, context: TemporalVisualContext, *, repair: bool = False) -> str:
        image_roles: list[str] = []
        image_number = 1
        if context.previous:
            image_roles.append(f"Image {image_number}: PREVIOUS nearby lecture visual state.")
            image_number += 1
        image_roles.append(f"Image {image_number}: CURRENT candidate to evaluate.")
        image_number += 1
        if context.next:
            image_roles.append(f"Image {image_number}: NEXT nearby lecture visual state.")

        transcript = (
            f"BEFORE speech: {semantic_input.before_text or '[none]'}\n"
            f"CURRENT speech: {semantic_input.current_text or '[none]'}\n"
            f"AFTER speech: {semantic_input.after_text or '[none]'}"
        )
        no_transcript = not semantic_input.has_transcript_context
        repair_text = "\nThis is a repair attempt. Return only one syntactically valid JSON object matching the schema exactly." if repair else ""
        prompt = f"""You analyze lecture visuals. Evaluate only the CURRENT frame as a candidate teaching state. Do not choose between PRIMARY/ALTERNATE candidates and do not decide final PDF inclusion.

Candidate ID: {semantic_input.candidate_id}

IMAGE ROLES
{chr(10).join(image_roles)}

TEMPORAL METADATA
candidate_timestamp_seconds={semantic_input.candidate_timestamp_seconds:.3f}
stable_window_start_seconds={semantic_input.stable_window_start_seconds:.3f}
stable_window_end_seconds={semantic_input.stable_window_end_seconds:.3f}
relative_position_in_stable_window={semantic_input.relative_position_in_stable_window:.4f}
visual_boundary_timestamp_seconds={semantic_input.visual_boundary_timestamp_seconds if semantic_input.visual_boundary_timestamp_seconds is not None else 'none'}
seconds_after_boundary={semantic_input.seconds_after_boundary if semantic_input.seconds_after_boundary is not None else 'none'}

TRANSCRIPT CONTEXT
{transcript}
Transcript availability: {'none; use visual evidence only' if no_transcript else 'available as supporting evidence'}.

JUDGMENT GUIDANCE
- If NEXT adds substantial writing/content that CURRENT lacks, CURRENT is likely incomplete or still changing.
- If CURRENT meaningfully advances from PREVIOUS and NEXT remains visually similar, CURRENT is more likely settled/complete.
- Blank/fade/half-transition/motion-blurred states should be TRANSITION or low usefulness.
- Teacher presence alone is not a rejection; penalize only if essential content is materially occluded.
- For boards, code, equations and diagrams, judge visible structural completion conservatively without requiring perfect OCR.
- Transcript is supporting evidence and must not override clear visual evidence blindly.
- Scores are heuristic judgments from 0 to 1, not calibrated probabilities.
- Do not provide hidden chain-of-thought. Use only short reason codes and an optional brief rationale.

Return JSON only. No Markdown and no prose outside the JSON object.
Use exactly these fields and enum values:
{{
  "candidate_id": {semantic_input.candidate_id},
  "content_type": "SLIDE|WHITEBOARD|BLACKBOARD|CODE|DIAGRAM|EQUATION|DOCUMENT|UI_DEMO|MIXED|OTHER|EMPTY_OR_LOW_INFORMATION|UNKNOWN",
  "completion_state": "COMPLETE|MOSTLY_COMPLETE|INCOMPLETE|TRANSITION|UNCERTAIN",
  "completion_score": 0.0,
  "visual_change_state": "STILL_CHANGING|MOSTLY_SETTLED|SETTLED|UNCERTAIN",
  "educational_usefulness": "HIGH|MEDIUM|LOW|NONE",
  "usefulness_score": 0.0,
  "transition_probability": 0.0,
  "has_meaningful_visual_content": true,
  "teacher_still_writing_likely": false,
  "transcript_visual_consistency": "CONSISTENT|PARTIALLY_CONSISTENT|UNRELATED|NO_TRANSCRIPT|UNCERTAIN",
  "confidence": 0.0,
  "reason_codes": ["SHORT_REASON_CODE"],
  "short_rationale": "optional <= 240 chars"
}}{repair_text}
"""
        if len(prompt) > self._settings.semantic_prompt_max_characters:
            raise SemanticAnalysisError("Semantic prompt exceeds configured safety limit.")
        return prompt

    def fingerprint(self, semantic_input: SemanticInputRecord, context: TemporalVisualContext) -> str:
        # Role and Phase-4 ranking are intentionally excluded to prevent bias and improve cache reuse.
        return stable_hash({"version": self.version, "prompt": self.build(semantic_input, context)})
