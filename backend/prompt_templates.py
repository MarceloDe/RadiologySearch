"""
Default prompt templates for the Radiology AI System
These can be edited through the Prompt Manager UI
"""

DEFAULT_PROMPTS = [
    {
        "template_id": "radiology_context_extraction",
        "name": "Radiology Context Extraction",
        "version": "1.0.0",
        "description": "Extracts structured radiology information from clinical cases",
        "template_text": """You are an expert radiologist. Extract structured information from the case.
Focus on anatomical structures, imaging characteristics, and clinically relevant features.
Be specific and comprehensive.

Extract radiology context from this case:
Patient: {age} year old {sex}
Clinical History: {clinical_history}
Imaging Modality: {imaging_modality}
Anatomical Region: {anatomical_region}
Image Description: {image_description}

Provide a JSON response with these fields:
- anatomy: list of anatomical structures
- imaging_modality: the modality used
- sequences: list of sequences (for MRI)
- measurements: dict of key measurements
- morphology: list of morphological features
- location: dict of location descriptors
- signal_characteristics: list of signal/density characteristics
- enhancement_pattern: list of enhancement patterns
- associated_findings: list of associated findings
- clinical_context: summary of clinical relevance""",
        "input_variables": ["age", "sex", "clinical_history", "imaging_modality", "anatomical_region", "image_description"],
        "model_type": "claude",
        "performance_metrics": {}
    },
    {
        "template_id": "literature_search_query_generation",
        "name": "Literature Search Query Generation",
        "version": "1.0.0",
        "description": "Generates optimized search queries for medical literature",
        "template_text": """You are an expert medical literature search specialist.
Generate targeted search queries to find radiology papers and resources with IMAGES.

Focus on:
1. Papers with imaging figures, radiological images, or diagnostic images
2. Case reports with visual findings
3. Radiology atlases and visual guides
4. Papers that explicitly mention figures or images

Include terms like: "with images", "figure", "radiographic findings", "imaging features", "case report with images"

Generate 5 search queries to find radiology literature WITH IMAGES for:
Anatomy: {anatomy}
Modality: {imaging_modality}
Key features: {features}
Clinical context: {clinical_context}

Format: One query per line, each targeting papers with visual content.""",
        "input_variables": ["anatomy", "imaging_modality", "features", "clinical_context"],
        "model_type": "deepseek",
        "performance_metrics": {}
    },
    {
        "template_id": "image_relevance_scoring",
        "name": "Image Relevance Scoring",
        "version": "1.0.0",
        "description": "Scores the relevance of extracted images to the case",
        "template_text": """You are a radiologist evaluating if an image is relevant to a specific case.
Score the relevance of the image based on:
1. Matching anatomical location
2. Similar imaging modality
3. Similar pathological features
4. Educational value for the case

Provide a relevance score from 0.0 to 1.0.

Case details:
Anatomy: {anatomy}
Modality: {modality}
Features: {features}

Image caption: {caption}
Image context: {context}

Provide only a decimal score between 0.0 and 1.0.""",
        "input_variables": ["anatomy", "modality", "features", "caption", "context"],
        "model_type": "claude",
        "performance_metrics": {}
    },
    {
        "template_id": "document_relevance_analysis",
        "name": "Document Relevance Analysis",
        "version": "1.0.0",
        "description": "Analyzes medical literature relevance to a case",
        "template_text": """You are a radiologist evaluating medical literature relevance.
Score the relevance based on:
1. Matching anatomy and imaging modality
2. Similar pathological features
3. Diagnostic value
4. Presence of relevant images or figures

Provide:
1. A relevance score (0.0-1.0)
2. Brief reasoning (one sentence)

Format: SCORE|REASONING

Case context:
Anatomy: {anatomy}
Modality: {modality}
Features: {features}

Document excerpt:
{content}

Evaluate relevance.""",
        "input_variables": ["anatomy", "modality", "features", "content"],
        "model_type": "claude",
        "performance_metrics": {}
    },
    {
        "template_id": "primary_diagnosis_generation",
        "name": "Primary Diagnosis Generation",
        "version": "1.0.0",
        "description": "Generates the primary diagnosis based on all available evidence",
        "template_text": """You are an expert radiologist providing a primary diagnosis.
Consider imaging findings, clinical context, and literature evidence.
Be specific and evidence-based.

Based on this case, provide the most likely diagnosis:

Clinical Context:
{clinical_context}

Radiology Findings:
{radiology_findings}

Literature Evidence:
{literature_evidence}

Provide a JSON response with:
- diagnosis: the primary diagnosis name
- confidence_score: 0.0 to 1.0
- reasoning: detailed clinical reasoning
- icd_code: ICD-10 code if applicable
- supporting_evidence: key supporting evidence
- imaging_features: characteristic imaging features""",
        "input_variables": ["clinical_context", "radiology_findings", "literature_evidence"],
        "model_type": "claude",
        "performance_metrics": {}
    },
    {
        "template_id": "differential_diagnosis_generation",
        "name": "Differential Diagnosis Generation",
        "version": "1.0.0",
        "description": "Generates differential diagnoses based on imaging findings",
        "template_text": """You are an expert radiologist generating differential diagnoses.
Consider alternative explanations for the imaging findings.
Rank by probability.

Given this primary diagnosis and case details, provide differential diagnoses:

Primary Diagnosis: {primary_diagnosis}

Case Summary:
{case_summary}

Provide a JSON array of differential diagnoses, each with:
- diagnosis: diagnosis name
- probability: 0.0 to 1.0
- reasoning: why this is considered
- distinguishing_features: what would distinguish this
- additional_workup: tests to confirm/exclude""",
        "input_variables": ["primary_diagnosis", "case_summary"],
        "model_type": "claude",
        "performance_metrics": {}
    },
    {
        "template_id": "confidence_assessment",
        "name": "Diagnostic Confidence Assessment",
        "version": "1.0.0",
        "description": "Assesses overall confidence in the diagnosis",
        "template_text": """Assess the overall confidence in the diagnosis based on:
1. Strength of imaging findings
2. Literature support
3. Clinical correlation
4. Differential considerations

Assess confidence for:
Primary: {primary}
Differentials: {differentials}
Literature matches: {lit_count} papers with average relevance {avg_relevance}

Provide JSON with:
- overall_confidence: 0.0 to 1.0
- evidence_quality: High/Medium/Low
- clinical_correlation: assessment
- diagnostic_certainty: 0.0 to 1.0
- recommendation: next steps
- limitations: any limitations""",
        "input_variables": ["primary", "differentials", "lit_count", "avg_relevance"],
        "model_type": "claude",
        "performance_metrics": {}
    },
    {
        "template_id": "image_description_extraction",
        "name": "Medical Image Description Extraction",
        "version": "1.0.0",
        "description": "Extracts and analyzes image descriptions from medical documents",
        "template_text": """You are a medical imaging specialist extracting image descriptions from documents.
Focus on:
1. Figure captions and labels
2. Image descriptions in the text
3. References to specific imaging findings
4. Technical details about the images

Document content:
{content}

Extract all image-related information and provide:
1. List of figure references with descriptions
2. Key imaging findings mentioned
3. Technical imaging details

Format as structured text with clear sections.""",
        "input_variables": ["content"],
        "model_type": "mistral",
        "performance_metrics": {}
    }
]