"""Reporter Agent - Generates natural language reports from analysis results.

This agent creates human-readable reports summarizing the analysis,
including detected elements, confidence levels, and caveats.
"""

import time
import json
from typing import Dict, Any
import mlflow

from langchain_google_genai import ChatGoogleGenerativeAI

from ..base import BaseAgent
from ..state import PipelineState


class ReporterAgent(BaseAgent):
    """Generates natural language reports from validated results.
    
    Responsibilities:
    - Format analysis results into natural language
    - Include confidence levels and caveats
    - Highlight multi-modal detections
    - Flag low-confidence predictions
    - Generate user-friendly summaries
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI = None):
        """Initialize reporter agent.
        
        Args:
            llm: Language model for report generation (defaults to Gemini)
        """
        super().__init__("ReporterAgent")
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )
    
    def process(self, state: PipelineState) -> PipelineState:
        """Generate natural language report.

        Args:
            state: Pipeline state with validated_results

        Returns:
            Updated state with report in metadata
        """
        # Get validated results
        validated_results = state.get("validated_results", [])
        validation_flags = state.get("validation_flags", [])
        inference_results = state.get("inference_results", {})
        kb = state.get("knowledge_base", {})

        # NEW: Get spectral summary if available
        metadata = state.get("metadata", {})
        spectral_summary = metadata.get("spectral_summary")
        spectral_features_desc = metadata.get("spectral_features_description", "")
        
        # Log inputs
        try:
            mlflow.log_param("reporter_num_predictions", len(validated_results))
            mlflow.log_param("reporter_num_flags", len(validation_flags))
            mlflow.log_param("reporter_llm_model", "gemini-2.0-flash-exp")
        except Exception:
            pass
        
        if not validated_results:
            state.setdefault("metadata", {})["report"] = "No predictions to report."
            return state
        
        # Prepare report context
        context = self._prepare_context(
            validated_results, validation_flags, inference_results, kb,
            spectral_summary, spectral_features_desc
        )
        
        # Generate report using LLM
        try:
            with mlflow.start_span(name="llm_report_generation") as span:
                start_time = time.time()
                
                report = self._generate_report(context)
                
                generation_time = time.time() - start_time
                span.set_attribute("generation_time_ms", int(generation_time * 1000))
                span.set_attribute("report_length", len(report))
                
                mlflow.log_metric("reporter_generation_time_ms", generation_time * 1000)
                mlflow.log_metric("reporter_report_length", len(report))
                mlflow.log_metric("reporter_llm_success", 1)
                
        except Exception as exc:
            mlflow.log_metric("reporter_llm_error", 1)
            # Fallback to template-based report
            report = self._generate_fallback_report(context)
        
        # Format and store report
        try:
            with mlflow.start_span(name="format_report") as span:
                formatted_report = self._format_report(report, context)
                
                span.set_attribute("num_elements", context["num_elements"])
                span.set_attribute("num_caveats", context["num_caveats"])
                
                mlflow.log_param("reporter_report_length", len(formatted_report))
                mlflow.log_param("reporter_num_elements", context["num_elements"])
                mlflow.log_param("reporter_num_caveats", context["num_caveats"])
                
        except Exception as exc:
            mlflow.log_metric("reporter_format_error", 1)
            formatted_report = report  # Use unformatted
        
        # Store report in state
        state.setdefault("metadata", {})["report"] = formatted_report
        state.setdefault("metadata", {})["report_summary"] = context["summary"]
        
        # Log final metrics
        try:
            mlflow.log_metric("reporter_agent_success", 1)
        except Exception:
            pass
        
        return state
    
    def _prepare_context(
        self,
        validated_results: list,
        validation_flags: list,
        inference_results: dict,
        kb: dict,
        spectral_summary: dict = None,
        spectral_features_desc: str = ""
    ) -> Dict[str, Any]:
        """Prepare context for report generation.

        Args:
            validated_results: Validated predictions
            validation_flags: Validation flags
            inference_results: Inference results
            kb: Knowledge Base
            spectral_summary: Parsed spectral features (NEW)
            spectral_features_desc: Natural language spectral description (NEW)

        Returns:
            Report context dictionary
        """
        # Get high confidence predictions
        high_conf = [p for p in validated_results if p["probability"] >= 0.6]
        low_conf = [p for p in validated_results if p["probability"] < 0.4]
        
        # Get multi-modal detections
        multi_modal = [p for p in validated_results if p.get("multi_modal", False)]
        
        summary = inference_results.get("summary", {})

        return {
            "num_elements": len(validated_results),
            "unique_elements": summary.get("unique_elements", len(validated_results)),
            "high_confidence": high_conf,
            "low_confidence": low_conf,
            "multi_modal": multi_modal,
            "validation_flags": validation_flags,
            "num_caveats": len(validation_flags),
            "avg_confidence": summary.get("avg_confidence", 0),
            "summary": summary,
            "kb": kb,
            # NEW: Include spectral parsing results
            "spectral_summary": spectral_summary,
            "spectral_features_desc": spectral_features_desc
        }
    
    def _generate_report(self, context: Dict[str, Any]) -> str:
        """Generate report using LLM.

        Args:
            context: Report context

        Returns:
            Generated report text
        """
        # NEW: Include spectral features in the prompt
        spectral_context = ""
        if context.get('spectral_features_desc'):
            spectral_context = f"""

SPECTRAL DATA ANALYSIS (Raw Features Detected):
{context['spectral_features_desc']}

Note: The LLM can now "see" the actual spectral features (peaks, valleys, absorption lines)
instead of just making educated guesses from the filename.
"""

        prompt = f"""Generate a concise scientific report for astronomical element detection analysis.

Context:
- Total elements detected: {context['num_elements']}
- Unique elements: {context['unique_elements']}
- High confidence detections: {len(context['high_confidence'])}
- Low confidence detections: {len(context['low_confidence'])}
- Multi-modal detections: {len(context['multi_modal'])}
- Average confidence: {context['avg_confidence']:.1%}
- Validation flags: {context['num_caveats']}
{spectral_context}
High confidence elements:
{json.dumps([{'element': p['element'], 'confidence': p['probability'], 'source': p.get('source', 'unknown')} for p in context['high_confidence'][:5]], indent=2)}

Report requirements:
1. Start with a brief summary referencing the SPECTRAL DATA ANALYSIS if available (2-3 sentences)
2. Mention key spectral features detected (absorption valleys, emission peaks, molecular signatures)
3. List detected elements with confidence levels and how they correlate with observed spectral features
4. Highlight multi-modal detections (detected by both spectral and image analysis)
5. Include caveats for low-confidence predictions
6. Keep it concise and scientific
7. Use proper astronomical terminology

Generate the report:"""

        response = self.llm.invoke(prompt)

        # Log token usage for cost tracking
        from ..base import log_llm_usage
        log_llm_usage(response, "reporter", "report_generation")

        return response.content
    
    def _generate_fallback_report(self, context: Dict[str, Any]) -> str:
        """Generate template-based report as fallback.
        
        Args:
            context: Report context
            
        Returns:
            Template-based report
        """
        lines = []
        
        lines.append("## Analysis Report")
        lines.append("")
        lines.append(f"Detected {context['num_elements']} element predictions ({context['unique_elements']} unique elements).")
        lines.append(f"Average confidence: {context['avg_confidence']:.1%}")
        lines.append("")
        
        if context['high_confidence']:
            lines.append("### High Confidence Detections")
            for pred in context['high_confidence'][:5]:
                lines.append(f"- **{pred['element']}**: {pred['probability']:.1%} ({pred.get('source', 'unknown')} analysis)")
            lines.append("")
        
        if context['multi_modal']:
            lines.append("### Multi-Modal Detections")
            lines.append("The following elements were detected by both spectral and image analysis:")
            for pred in context['multi_modal']:
                lines.append(f"- **{pred['element']}**: {pred['probability']:.1%}")
            lines.append("")
        
        if context['low_confidence']:
            lines.append("### Caveats")
            lines.append(f"{len(context['low_confidence'])} predictions have low confidence (<40%).")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_report(self, report: str, context: Dict[str, Any]) -> str:
        """Format report with metadata.
        
        Args:
            report: Raw report text
            context: Report context
            
        Returns:
            Formatted report
        """
        # Add metadata footer
        footer = f"\n\n---\n*Analysis completed with {context['num_elements']} predictions, {context['num_caveats']} validation flags*"
        
        return report + footer
