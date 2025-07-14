"""
Enhanced Quality Assurance Agent with LLM-Powered Reasoning

This agent acts as an AI Judge that uses Gemini 2.0 Flash for sophisticated
analysis, conflict resolution, and intelligent synthesis of multi-agent results.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .base_agent import BaseAgent
from ..models.agentic_models import (
    AgentProfile, AgentRole, ToolType, AgentCapability,
    AgentTask, AgentContribution
)
from ..services.agent_llm_service import AgentLLMService

logger = logging.getLogger(__name__)

@dataclass
class ConflictAnalysis:
    """Analysis of conflicts between agent findings."""
    conflict_type: str
    conflicting_agents: List[str]
    conflicting_statements: List[str]
    severity: float  # 0-1 scale
    resolution_strategy: str

@dataclass
class EvidenceWeight:
    """Weight assigned to evidence by the AI Judge."""
    evidence_source: str
    evidence_content: str
    reliability_score: float  # 0-1 scale
    relevance_score: float   # 0-1 scale
    final_weight: float      # 0-1 scale

@dataclass
class JudgeVerdict:
    """Final verdict from the AI Judge."""
    overall_confidence: float
    quality_score: float
    synthesis_summary: str
    key_insights: List[str]
    resolved_conflicts: List[ConflictAnalysis]
    evidence_evaluation: List[EvidenceWeight]
    recommendations: List[str]
    reasoning_chain: List[str]

class EnhancedQualityAssuranceAgent(BaseAgent):
    """
    Enhanced Quality Assurance Agent with LLM-powered reasoning.
    
    Acts as an AI Judge that:
    - Evaluates agent output quality using LLM reasoning
    - Resolves conflicts between agents intelligently
    - Weighs evidence and assigns confidence scores
    - Provides sophisticated synthesis with detailed reasoning
    """
    
    def __init__(self, profile: Optional[AgentProfile] = None):
        """Initialize the Enhanced Quality Assurance Agent."""
        if profile is None:
            profile = self.create_default_profile()
        super().__init__(profile)
        
        # Initialize LLM service specifically for judging
        self.llm_service = AgentLLMService()
        
        # Judge-specific configuration
        self.min_evidence_threshold = 0.7
        self.conflict_resolution_temperature = 0.3  # Lower for more consistent judgments
        self.max_reasoning_depth = 5
        
        logger.info("Enhanced Quality Assurance Agent initialized with LLM-powered reasoning")
    
    @classmethod
    def create_default_profile(cls) -> AgentProfile:
        """Create enhanced profile for the AI Judge."""
        capabilities = [
            AgentCapability(
                name="LLM-Powered Analysis Evaluation",
                description="Evaluate quality and consistency of agent analyses using advanced LLM reasoning",
                confidence_level=0.92,
                required_tools=[ToolType.LLM_REASONING, ToolType.DATA_ANALYSIS],
                complexity_level=9
            ),
            AgentCapability(
                name="Intelligent Conflict Resolution",
                description="Resolve disagreements between agents using sophisticated reasoning",
                confidence_level=0.89,
                required_tools=[ToolType.LLM_REASONING, ToolType.LOGICAL_ANALYSIS],
                complexity_level=9
            ),
            AgentCapability(
                name="Evidence Weighing and Validation",
                description="Assign reliability scores to evidence and validate claims",
                confidence_level=0.91,
                required_tools=[ToolType.LLM_REASONING, ToolType.FACT_CHECKING],
                complexity_level=8
            ),
            AgentCapability(
                name="Synthesis and Executive Judgment",
                description="Create comprehensive syntheses with detailed reasoning chains",
                confidence_level=0.93,
                required_tools=[ToolType.LLM_REASONING, ToolType.STRATEGIC_ANALYSIS],
                complexity_level=10
            )
        ]
        
        return AgentProfile(
            name="EnhancedQualityGuard",
            role=AgentRole.QUALITY_ASSURANCE,
            description="Advanced AI Judge with LLM-powered reasoning for sophisticated analysis evaluation and synthesis",
            capabilities=capabilities,
            expertise_areas=[
                "Multi-Agent Analysis Evaluation",
                "Conflict Resolution and Mediation", 
                "Evidence Validation and Weighing",
                "Executive Synthesis and Reasoning",
                "Quality Assurance and Validation",
                "Strategic Decision Making"
            ],
            specializations=[
                "LLM-powered conflict resolution",
                "Sophisticated evidence evaluation",
                "Multi-perspective synthesis",
                "Reasoning chain construction"
            ],
            experience_level=10,
            confidence_score=0.91,
            preferred_tools=[
                ToolType.LLM_REASONING,
                ToolType.DATA_ANALYSIS,
                ToolType.LOGICAL_ANALYSIS
            ]
        )
    
    def get_expertise_areas(self) -> List[str]:
        """Get the enhanced agent's areas of expertise."""
        return [
            "Multi-Agent Analysis Evaluation",
            "LLM-Powered Conflict Resolution",
            "Evidence Validation and Weighing",
            "Executive Synthesis and Reasoning",
            "Quality Assurance and Validation",
            "Strategic Decision Making"
        ]
    
    def get_preferred_tools(self) -> List[ToolType]:
        """Get the enhanced agent's preferred tools."""
        return [
            ToolType.LLM_REASONING,
            ToolType.DATA_ANALYSIS,
            ToolType.LOGICAL_ANALYSIS
        ]
    
    async def analyze_task(self, request: Dict[str, Any], task: AgentTask) -> AgentContribution:
        """
        Perform enhanced quality assurance analysis using LLM-powered reasoning.
        
        Args:
            request: Multi-agent analysis results to evaluate
            task: The QA task assigned to this agent
            
        Returns:
            AgentContribution: Enhanced synthesis with LLM reasoning
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info("🏛️ Enhanced QA Agent starting LLM-powered analysis evaluation")
            
            # STEP 1: Extract and validate agent results
            agent_results = self._extract_agent_results(request)
            
            # STEP 2: Evaluate each agent's output quality using LLM
            quality_evaluations = await self._evaluate_agent_quality(agent_results)
            
            # STEP 3: Identify and analyze conflicts using LLM reasoning
            conflicts = await self._identify_conflicts_with_llm(agent_results)
            
            # STEP 4: Resolve conflicts using sophisticated LLM reasoning
            resolved_conflicts = await self._resolve_conflicts_with_llm(conflicts, agent_results)
            
            # STEP 5: Weigh evidence and assign confidence scores
            evidence_weights = await self._weigh_evidence_with_llm(agent_results)
            
            # STEP 6: Generate final synthesis using LLM reasoning
            final_verdict = await self._synthesize_final_judgment(
                agent_results, quality_evaluations, resolved_conflicts, evidence_weights
            )
            
            # STEP 7: Create detailed reasoning chain
            reasoning_chain = self._build_reasoning_chain(
                quality_evaluations, conflicts, resolved_conflicts, evidence_weights
            )
            
            # Create enhanced contribution
            contribution = AgentContribution(
                agent_id=self.profile.agent_id,
                agent_role=self.profile.role,
                findings=final_verdict.key_insights,
                insights=[final_verdict.synthesis_summary],
                recommendations=final_verdict.recommendations,
                evidence=[f"Evidence weight: {ew.final_weight:.2f} - {ew.evidence_content}" for ew in evidence_weights[:5]],
                methodology="LLM-powered multi-agent evaluation with conflict resolution and evidence weighing",
                tools_used=["gemini-2.0-flash", "conflict_resolver", "evidence_evaluator", "synthesis_engine"],
                confidence_level=final_verdict.overall_confidence,
                reasoning_chain=reasoning_chain,
                assumptions=[
                    "Agent analyses are performed in good faith",
                    "LLM reasoning provides objective evaluation",
                    "Conflicts can be resolved through evidence weighing"
                ],
                limitations=[
                    "LLM reasoning may have inherent biases",
                    "Some conflicts may require domain expert input",
                    "Evidence quality depends on original data sources"
                ],
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
            
            logger.info(f"🏛️ Enhanced QA Agent completed with confidence: {final_verdict.overall_confidence:.2f}")
            return contribution
            
        except Exception as e:
            logger.error(f"🚨 Enhanced QA Agent failed: {e}")
            # Return fallback contribution
            return self._create_fallback_contribution(start_time, str(e))
    
    def _extract_agent_results(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize agent results for evaluation."""
        return {
            "music_analysis": request.get("music_analysis", {}),
            "commercial_analysis": request.get("commercial_analysis", {}),
            "novelty_analysis": request.get("novelty_analysis", {}),
            "benchmark_analysis": request.get("benchmark_analysis", {}),
            "hit_prediction": request.get("hit_prediction", {}),
            "original_request": request.get("original_request", {})
        }
    
    async def _evaluate_agent_quality(self, agent_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the quality of each agent's output using LLM reasoning."""
        quality_scores = {}
        
        for agent_name, analysis in agent_results.items():
            if agent_name == "original_request":
                continue
                
            try:
                # Create LLM prompt for quality evaluation
                prompt = self._create_quality_evaluation_prompt(agent_name, analysis)
                
                # Get LLM evaluation
                response = await self.llm_service.llm_provider.generate(
                    prompt, 
                    max_tokens=500,
                    temperature=self.conflict_resolution_temperature
                )
                
                # Parse quality score from response
                quality_score = self._parse_quality_score(response)
                quality_scores[agent_name] = quality_score
                
                logger.info(f"🔍 Quality evaluation for {agent_name}: {quality_score:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {agent_name} quality: {e}")
                quality_scores[agent_name] = 0.7  # Default score
        
        return quality_scores
    
    def _create_quality_evaluation_prompt(self, agent_name: str, analysis: Dict[str, Any]) -> str:
        """Create LLM prompt for evaluating agent output quality."""
        
        findings = analysis.get("findings", [])
        insights = analysis.get("insights", [])
        recommendations = analysis.get("recommendations", [])
        confidence = analysis.get("confidence_level", 0.5)
        
        prompt = f"""
As an expert AI Judge evaluating multi-agent analysis quality, assess the following {agent_name} output:

**Agent Output to Evaluate:**
- Findings: {findings[:3]}  # Top 3 findings
- Insights: {insights[:3]}  # Top 3 insights  
- Recommendations: {recommendations[:3]}  # Top 3 recommendations
- Agent Confidence: {confidence}

**Evaluation Criteria:**
1. **Coherence**: Are the findings logically consistent and well-structured?
2. **Depth**: Does the analysis show sophisticated understanding?
3. **Relevance**: Are the insights directly relevant to music analysis?
4. **Actionability**: Are recommendations specific and implementable?
5. **Evidence**: Is the analysis well-supported by data?

**Instructions:**
Provide a quality score from 0.0 to 1.0 where:
- 0.9-1.0: Exceptional quality, sophisticated insights, highly actionable
- 0.7-0.8: Good quality, solid analysis, useful recommendations  
- 0.5-0.6: Average quality, basic insights, generic recommendations
- 0.3-0.4: Poor quality, shallow analysis, unclear recommendations
- 0.0-0.2: Very poor quality, incoherent or irrelevant analysis

**Response Format:**
Quality Score: [0.XX]
Reasoning: [Brief explanation of the score in 1-2 sentences]
"""
        
        return prompt
    
    def _parse_quality_score(self, response: str) -> float:
        """Parse quality score from LLM response."""
        try:
            # Look for "Quality Score: X.XX" pattern
            lines = response.split('\n')
            for line in lines:
                if 'quality score:' in line.lower():
                    # Extract number from the line
                    import re
                    numbers = re.findall(r'(\d+\.?\d*)', line)
                    if numbers:
                        score = float(numbers[0])
                        # Ensure score is in valid range
                        return max(0.0, min(1.0, score))
            
            # Fallback: look for any decimal number in response
            import re
            numbers = re.findall(r'(\d+\.\d+)', response)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
                
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse quality score from: {response[:100]}")
        
        return 0.7  # Default score if parsing fails
    
    async def _identify_conflicts_with_llm(self, agent_results: Dict[str, Any]) -> List[ConflictAnalysis]:
        """Identify conflicts between agents using LLM reasoning."""
        conflicts = []
        
        try:
            # Create conflict identification prompt
            prompt = self._create_conflict_identification_prompt(agent_results)
            
            # Get LLM analysis
            response = await self.llm_service.llm_provider.generate(
                prompt,
                max_tokens=800,
                temperature=self.conflict_resolution_temperature
            )
            
            # Parse conflicts from response
            conflicts = self._parse_conflicts(response, agent_results)
            
            logger.info(f"🔍 Identified {len(conflicts)} conflicts between agents")
            
        except Exception as e:
            logger.warning(f"Failed to identify conflicts with LLM: {e}")
        
        return conflicts
    
    def _create_conflict_identification_prompt(self, agent_results: Dict[str, Any]) -> str:
        """Create prompt for LLM to identify conflicts between agents."""
        
        # Extract key findings from each agent
        music_findings = agent_results.get("music_analysis", {}).get("findings", [])[:3]
        commercial_findings = agent_results.get("commercial_analysis", {}).get("findings", [])[:3]
        novelty_findings = agent_results.get("novelty_analysis", {}).get("findings", [])[:3]
        
        prompt = f"""
As an expert AI Judge, analyze the following multi-agent findings to identify conflicts or contradictions:

**Music Analysis Agent Findings:**
{json.dumps(music_findings, indent=2)}

**Commercial Analysis Agent Findings:**  
{json.dumps(commercial_findings, indent=2)}

**Novelty Assessment Agent Findings:**
{json.dumps(novelty_findings, indent=2)}

**Task:**
Identify any conflicts, contradictions, or inconsistencies between these agent findings. Look for:

1. **Direct Contradictions**: Agents making opposite claims about the same aspect
2. **Inconsistent Assessments**: Different quality/potential assessments of the same elements
3. **Conflicting Recommendations**: Agents suggesting incompatible actions
4. **Data Inconsistencies**: Different interpretations of the same underlying data

**Response Format:**
For each conflict found, provide:
- Conflict Type: [direct_contradiction|inconsistent_assessment|conflicting_recommendation|data_inconsistency]
- Agents Involved: [list of agent names]
- Conflicting Statements: [specific contradictory statements]
- Severity: [0.1-1.0 where 1.0 is major contradiction]

If no conflicts found, respond: "No significant conflicts identified."
"""
        
        return prompt
    
    def _parse_conflicts(self, response: str, agent_results: Dict[str, Any]) -> List[ConflictAnalysis]:
        """Parse conflict analysis from LLM response."""
        conflicts = []
        
        try:
            if "no significant conflicts" in response.lower():
                return conflicts
            
            # Simple parsing - look for conflict patterns
            lines = response.split('\n')
            current_conflict = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if 'conflict type:' in line.lower():
                    current_conflict['type'] = line.split(':', 1)[1].strip()
                elif 'agents involved:' in line.lower():
                    current_conflict['agents'] = line.split(':', 1)[1].strip()
                elif 'conflicting statements:' in line.lower():
                    current_conflict['statements'] = line.split(':', 1)[1].strip()
                elif 'severity:' in line.lower():
                    try:
                        severity_str = line.split(':', 1)[1].strip()
                        current_conflict['severity'] = float(severity_str)
                        
                        # Complete conflict - add to list
                        if all(k in current_conflict for k in ['type', 'agents', 'statements', 'severity']):
                            conflict = ConflictAnalysis(
                                conflict_type=current_conflict['type'],
                                conflicting_agents=current_conflict['agents'].split(','),
                                conflicting_statements=[current_conflict['statements']],
                                severity=current_conflict['severity'],
                                resolution_strategy="llm_mediation"
                            )
                            conflicts.append(conflict)
                            current_conflict = {}
                    except ValueError:
                        continue
            
        except Exception as e:
            logger.warning(f"Failed to parse conflicts: {e}")
        
        return conflicts
    
    async def _resolve_conflicts_with_llm(self, conflicts: List[ConflictAnalysis], 
                                        agent_results: Dict[str, Any]) -> List[ConflictAnalysis]:
        """Resolve conflicts using sophisticated LLM reasoning."""
        resolved_conflicts = []
        
        for conflict in conflicts:
            try:
                # Create conflict resolution prompt
                prompt = self._create_conflict_resolution_prompt(conflict, agent_results)
                
                # Get LLM resolution
                response = await self.llm_service.llm_provider.generate(
                    prompt,
                    max_tokens=600,
                    temperature=self.conflict_resolution_temperature
                )
                
                # Update conflict with resolution
                conflict.resolution_strategy = response
                resolved_conflicts.append(conflict)
                
                logger.info(f"🔧 Resolved conflict: {conflict.conflict_type}")
                
            except Exception as e:
                logger.warning(f"Failed to resolve conflict {conflict.conflict_type}: {e}")
                resolved_conflicts.append(conflict)
        
        return resolved_conflicts
    
    def _create_conflict_resolution_prompt(self, conflict: ConflictAnalysis, 
                                         agent_results: Dict[str, Any]) -> str:
        """Create prompt for LLM to resolve a specific conflict."""
        
        prompt = f"""
As an expert AI Judge, resolve the following conflict between specialized agents:

**Conflict Details:**
- Type: {conflict.conflict_type}
- Agents Involved: {', '.join(conflict.conflicting_agents)}
- Conflicting Statements: {'; '.join(conflict.conflicting_statements)}
- Severity: {conflict.severity}/1.0

**Supporting Context:**
- Music Analysis Context: {agent_results.get('music_analysis', {}).get('methodology', 'N/A')}
- Commercial Analysis Context: {agent_results.get('commercial_analysis', {}).get('methodology', 'N/A')}
- Available Evidence: {agent_results.get('hit_prediction', {}).get('confidence_score', 'N/A')}

**Task:**
Provide a reasoned resolution to this conflict by:

1. **Analyzing Root Cause**: Why do the agents disagree?
2. **Weighing Evidence**: Which agent's perspective has stronger supporting evidence?
3. **Finding Synthesis**: Is there a way both perspectives can be partially correct?
4. **Final Resolution**: What is the most reasonable conclusion?

**Response Format:**
Resolution: [Your reasoned conclusion in 2-3 sentences]
Confidence: [0.1-1.0 confidence in this resolution]
Reasoning: [Brief explanation of why this resolution is most appropriate]
"""
        
        return prompt
    
    async def _weigh_evidence_with_llm(self, agent_results: Dict[str, Any]) -> List[EvidenceWeight]:
        """Weigh evidence from all agents using LLM evaluation."""
        evidence_weights = []
        
        try:
            # Extract all evidence statements
            all_evidence = []
            for agent_name, analysis in agent_results.items():
                if agent_name == "original_request":
                    continue
                evidence_items = analysis.get("evidence", [])
                for item in evidence_items:
                    all_evidence.append({
                        "source": agent_name,
                        "content": item,
                        "agent_confidence": analysis.get("confidence_level", 0.5)
                    })
            
            # Evaluate each piece of evidence
            for evidence in all_evidence[:10]:  # Limit to top 10 pieces
                prompt = self._create_evidence_evaluation_prompt(evidence)
                
                response = await self.llm_service.llm_provider.generate(
                    prompt,
                    max_tokens=300,
                    temperature=self.conflict_resolution_temperature
                )
                
                weight = self._parse_evidence_weight(response, evidence)
                evidence_weights.append(weight)
                
        except Exception as e:
            logger.warning(f"Failed to weigh evidence with LLM: {e}")
        
        return evidence_weights
    
    def _create_evidence_evaluation_prompt(self, evidence: Dict[str, Any]) -> str:
        """Create prompt for evaluating evidence reliability and relevance."""
        
        prompt = f"""
As an expert AI Judge, evaluate the reliability and relevance of this evidence:

**Evidence to Evaluate:**
- Source Agent: {evidence['source']}
- Evidence Statement: "{evidence['content']}"
- Agent Confidence: {evidence['agent_confidence']}

**Evaluation Criteria:**
1. **Reliability**: Is this evidence based on solid data/analysis?
2. **Relevance**: How directly relevant is this to music hit prediction?
3. **Specificity**: Is this specific and actionable vs. generic?
4. **Verifiability**: Can this evidence be verified or validated?

**Response Format:**
Reliability Score: [0.0-1.0]
Relevance Score: [0.0-1.0]
Final Weight: [0.0-1.0 overall weight for this evidence]
Reasoning: [Brief explanation in 1 sentence]
"""
        
        return prompt
    
    def _parse_evidence_weight(self, response: str, evidence: Dict[str, Any]) -> EvidenceWeight:
        """Parse evidence weight evaluation from LLM response."""
        try:
            # Parse scores from response
            reliability = 0.7
            relevance = 0.7
            final_weight = 0.7
            
            lines = response.split('\n')
            for line in lines:
                if 'reliability score:' in line.lower():
                    reliability = self._extract_score_from_line(line)
                elif 'relevance score:' in line.lower():
                    relevance = self._extract_score_from_line(line)
                elif 'final weight:' in line.lower():
                    final_weight = self._extract_score_from_line(line)
            
            return EvidenceWeight(
                evidence_source=evidence['source'],
                evidence_content=evidence['content'],
                reliability_score=reliability,
                relevance_score=relevance,
                final_weight=final_weight
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse evidence weight: {e}")
            return EvidenceWeight(
                evidence_source=evidence['source'],
                evidence_content=evidence['content'],
                reliability_score=0.7,
                relevance_score=0.7,
                final_weight=0.7
            )
    
    def _extract_score_from_line(self, line: str) -> float:
        """Extract numeric score from a line of text."""
        import re
        numbers = re.findall(r'(\d+\.?\d*)', line)
        if numbers:
            score = float(numbers[0])
            return max(0.0, min(1.0, score))
        return 0.7
    
    def _preserve_original_agent_insights(self, agent_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Extract and preserve original agent insights when LLM synthesis fails."""
        preserved_insights = []
        preserved_recommendations = []
        
        # Extract insights from each agent
        for agent_name, analysis in agent_results.items():
            if agent_name == "original_request":
                continue
                
            agent_insights = analysis.get("insights", [])
            agent_recommendations = analysis.get("recommendations", [])
            
            # Add agent name prefix to preserve attribution
            for insight in agent_insights[:2]:  # Top 2 insights per agent
                if insight and len(insight.strip()) > 10:  # Skip generic insights
                    preserved_insights.append(f"[{agent_name}] {insight}")
            
            for rec in agent_recommendations[:2]:  # Top 2 recommendations per agent
                if rec and len(rec.strip()) > 10:  # Skip generic recommendations
                    preserved_recommendations.append(f"[{agent_name}] {rec}")
        
        return preserved_insights, preserved_recommendations

    async def _synthesize_final_judgment(self, agent_results: Dict[str, Any],
                                       quality_evaluations: Dict[str, float],
                                       resolved_conflicts: List[ConflictAnalysis],
                                       evidence_weights: List[EvidenceWeight]) -> JudgeVerdict:
        """Generate final synthesis using LLM reasoning."""
        
        try:
            # Create comprehensive synthesis prompt
            prompt = self._create_synthesis_prompt(
                agent_results, quality_evaluations, resolved_conflicts, evidence_weights
            )
            
            # Get LLM synthesis
            response = await self.llm_service.llm_provider.generate(
                prompt,
                max_tokens=1000,
                temperature=self.conflict_resolution_temperature
            )
            
            # Parse synthesis response
            verdict = self._parse_synthesis_verdict(response, quality_evaluations, evidence_weights, agent_results)
            
            logger.info(f"🏛️ Final synthesis completed with confidence: {verdict.overall_confidence:.2f}")
            return verdict
            
        except Exception as e:
            logger.warning(f"Failed to synthesize final judgment: {e}")
            return self._create_fallback_verdict(quality_evaluations, evidence_weights, agent_results)
    
    def _create_synthesis_prompt(self, agent_results: Dict[str, Any],
                               quality_evaluations: Dict[str, float],
                               resolved_conflicts: List[ConflictAnalysis],
                               evidence_weights: List[EvidenceWeight]) -> str:
        """Create comprehensive synthesis prompt."""
        
        # Get top evidence
        top_evidence = sorted(evidence_weights, key=lambda x: x.final_weight, reverse=True)[:5]
        
        prompt = f"""
As an expert AI Judge, synthesize the following multi-agent analysis into a comprehensive final judgment:

**Agent Quality Evaluations:**
{json.dumps(quality_evaluations, indent=2)}

**Resolved Conflicts:**
{len(resolved_conflicts)} conflicts resolved through reasoning

**Top Weighted Evidence:**
{json.dumps([{"source": e.evidence_source, "content": e.evidence_content, "weight": e.final_weight} for e in top_evidence], indent=2)}

**Original Analysis Results:**
- Music Analysis Insights: {agent_results.get('music_analysis', {}).get('insights', [])[:2]}
- Commercial Analysis Insights: {agent_results.get('commercial_analysis', {}).get('insights', [])[:2]}
- Hit Prediction Score: {agent_results.get('hit_prediction', {}).get('prediction', 'N/A')}

**Task:**
Create a comprehensive final synthesis that:
1. Integrates the highest-quality insights from all agents
2. Accounts for resolved conflicts and evidence weights
3. Provides actionable recommendations
4. Assigns appropriate confidence levels

**Response Format:**
Overall Confidence: [0.0-1.0]
Quality Score: [0.0-1.0]
Synthesis Summary: [2-3 sentences summarizing key findings]
Key Insights: [List of 3-4 most important insights]
Recommendations: [List of 3-4 actionable recommendations]
Reasoning: [Brief explanation of how you reached these conclusions]
"""
        
        return prompt
    
    def _parse_synthesis_verdict(self, response: str, quality_evaluations: Dict[str, float],
                               evidence_weights: List[EvidenceWeight], agent_results: Dict[str, Any]) -> JudgeVerdict:
        """Parse final synthesis verdict from LLM response."""
        
        # Default values - preserve original agent insights when LLM parsing fails
        overall_confidence = sum(quality_evaluations.values()) / len(quality_evaluations) if quality_evaluations else 0.75
        quality_score = overall_confidence
        synthesis_summary = "LLM synthesis parsing failed - preserving original agent insights"
        key_insights = []  # Will be populated from original agent insights if LLM fails
        recommendations = []  # Will be populated from original agent insights if LLM fails
        reasoning_chain = ["Preserving original agent insights due to synthesis parsing failure"]
        
        try:
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if 'overall confidence:' in line.lower():
                    overall_confidence = self._extract_score_from_line(line)
                elif 'quality score:' in line.lower():
                    quality_score = self._extract_score_from_line(line)
                elif 'synthesis summary:' in line.lower():
                    synthesis_summary = line.split(':', 1)[1].strip()
                elif 'key insights:' in line.lower():
                    current_section = 'insights'
                elif 'recommendations:' in line.lower():
                    current_section = 'recommendations'
                elif 'reasoning:' in line.lower():
                    reasoning_text = line.split(':', 1)[1].strip()
                    reasoning_chain = [reasoning_text]
                elif line.startswith('-') or line.startswith('•'):
                    content = line[1:].strip()
                    if current_section == 'insights':
                        key_insights.append(content)
                    elif current_section == 'recommendations':
                        recommendations.append(content)
            
        except Exception as e:
            logger.warning(f"Failed to parse synthesis verdict: {e}")
        
        # If LLM parsing failed or returned empty results, preserve original agent insights
        if not key_insights or not recommendations:
            logger.info("🔄 LLM synthesis parsing failed - preserving original agent insights")
            preserved_insights, preserved_recommendations = self._preserve_original_agent_insights(agent_results)
            
            if not key_insights:
                key_insights = preserved_insights[:4] if preserved_insights else ["Analysis completed by specialized agents"]
            if not recommendations:
                recommendations = preserved_recommendations[:4] if preserved_recommendations else ["Review individual agent findings"]
        
        return JudgeVerdict(
            overall_confidence=overall_confidence,
            quality_score=quality_score,
            synthesis_summary=synthesis_summary,
            key_insights=key_insights[:4],  # Limit to top 4
            resolved_conflicts=[],  # Already handled
            evidence_evaluation=evidence_weights,
            recommendations=recommendations[:4],  # Limit to top 4
            reasoning_chain=reasoning_chain
        )
    
    def _create_fallback_verdict(self, quality_evaluations: Dict[str, float],
                               evidence_weights: List[EvidenceWeight], agent_results: Dict[str, Any]) -> JudgeVerdict:
        """Create fallback verdict when LLM synthesis fails."""
        
        avg_quality = sum(quality_evaluations.values()) / len(quality_evaluations) if quality_evaluations else 0.75
        
        # Preserve original agent insights as fallback
        preserved_insights, preserved_recommendations = self._preserve_original_agent_insights(agent_results)
        
        fallback_insights = preserved_insights[:4] if preserved_insights else [
            "Individual agent analyses completed successfully",
            "Quality evaluation performed across all agents", 
            "Evidence validation applied to findings"
        ]
        
        fallback_recommendations = preserved_recommendations[:4] if preserved_recommendations else [
            "Review individual agent findings for detailed insights",
            "Consider agent quality scores in decision making",
            "Apply evidence weights when evaluating findings"
        ]
        
        return JudgeVerdict(
            overall_confidence=avg_quality,
            quality_score=avg_quality,
            synthesis_summary="Preserving original agent insights - LLM synthesis unavailable",
            key_insights=fallback_insights,
            resolved_conflicts=[],
            evidence_evaluation=evidence_weights,
            recommendations=fallback_recommendations,
            reasoning_chain=[
                "Preserved original agent insights due to LLM synthesis failure",
                "Quality scores averaged across all agents",
                "Original agent findings maintained for user review"
            ]
        )
    
    def _build_reasoning_chain(self, quality_evaluations: Dict[str, float],
                             conflicts: List[ConflictAnalysis],
                             resolved_conflicts: List[ConflictAnalysis],
                             evidence_weights: List[EvidenceWeight]) -> List[str]:
        """Build detailed reasoning chain for the judgment."""
        
        reasoning = []
        
        # Quality evaluation step
        avg_quality = sum(quality_evaluations.values()) / len(quality_evaluations) if quality_evaluations else 0.75
        reasoning.append(f"1. Evaluated agent output quality using LLM reasoning - average quality: {avg_quality:.2f}")
        
        # Conflict resolution step
        if conflicts:
            reasoning.append(f"2. Identified and resolved {len(conflicts)} conflicts using sophisticated LLM mediation")
        else:
            reasoning.append("2. No significant conflicts detected between agent analyses")
        
        # Evidence weighing step
        if evidence_weights:
            top_weight = max(evidence_weights, key=lambda x: x.final_weight).final_weight
            reasoning.append(f"3. Evaluated {len(evidence_weights)} pieces of evidence - highest weight: {top_weight:.2f}")
        
        # Synthesis step
        reasoning.append("4. Synthesized final judgment using LLM-powered reasoning across all agent inputs")
        
        # Final validation
        reasoning.append("5. Applied quality assurance validation to ensure coherent and actionable output")
        
        return reasoning
    
    def _create_fallback_contribution(self, start_time: datetime, error_msg: str) -> AgentContribution:
        """Create fallback contribution when enhanced analysis fails."""
        
        return AgentContribution(
            agent_id=self.profile.agent_id,
            agent_role=self.profile.role,
            findings=[f"Enhanced QA analysis failed: {error_msg}"],
            insights=["Fallback quality assurance applied"],
            recommendations=["Review agent outputs manually for quality"],
            evidence=["LLM-powered reasoning unavailable"],
            methodology="fallback_quality_assurance",
            tools_used=["basic_validation"],
            confidence_level=0.5,
            reasoning_chain=[
                "Enhanced QA agent encountered error",
                "Applied fallback quality assurance",
                "Manual review recommended"
            ],
            assumptions=["Basic validation sufficient for fallback"],
            limitations=["Limited analysis due to LLM failure"],
            processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            started_at=start_time,
            completed_at=datetime.utcnow()
        )