"""LLM-based compliance assessment."""
import json
from typing import Dict, List
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv 



class ComplianceAssessor:
    """Assesses policy compliance against regulations using LLM."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        # Load environment variables from .env file
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        self.client = OpenAI()
        self.model = model
    
    def find_related_articles(
        self, 
        policy_section: Dict,
        candidate_articles: List[Dict]
    ) -> List[Dict]:
        """
        Identify which GDPR articles relate to a policy section.
        Uses LLM to evaluate pre-retrieved candidates (prevents hallucination).
        """
        candidate_text = "\n\n".join([
            f"**{art['metadata']['section_id']}**: {art['metadata']['section_title']}\n{art['text'][:300]}..."
            for art in candidate_articles
        ])
        
        prompt = f"""You are a GDPR compliance expert. Given a policy section and candidate GDPR articles, identify which articles are related.

**Policy Section**: {policy_section['metadata']['section_id']} - {policy_section['metadata']['section_title']}
{policy_section['text']}

**Candidate GDPR Articles**:
{candidate_text}

For each candidate article, determine:
1. **Relationship**: Direct, Indirect, or None
2. **Reasoning**: Brief explanation

Output as JSON array:
[
  {{
    "article": "Article 5",
    "relationship": "Direct",
    "reasoning": "Policy section 5.1 directly implements GDPR Article 5 principles"
  }},
  ...
]

Only include articles with Direct or Indirect relationships."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a GDPR compliance expert. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        try:
            relationships = json.loads(response.choices[0].message.content)
            return relationships
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse LLM response as JSON")
            return []
    
    def assess_compliance(
        self, 
        policy_section: Dict,
        gdpr_article: Dict
    ) -> Dict:
        """
        Assess compliance of a policy section against a GDPR article.
        Uses chain-of-thought reasoning.
        """
        prompt = f"""You are a GDPR compliance auditor. Assess whether a policy section complies with a GDPR article.

**Policy Section**: {policy_section['metadata']['section_id']} - {policy_section['metadata']['section_title']}
{policy_section['text']}

**GDPR Article**: {gdpr_article['metadata']['section_id']} - {gdpr_article['metadata']['section_title']}
{gdpr_article['text']}

Follow these steps:

**Step 1**: Extract REQUIREMENTS from the GDPR article (what MUST organizations do?)
**Step 2**: Extract IMPLEMENTATIONS from the policy (what DOES the organization do?)
**Step 3**: Map each requirement to implementations (which requirements are addressed?)
**Step 4**: Identify GAPS (requirements with no implementation)
**Step 5**: Determine compliance level: Full / Partial / Non-compliant

Output as JSON:
{{
  "step1_requirements": ["requirement 1", "requirement 2", ...],
  "step2_implementations": ["implementation 1", "implementation 2", ...],
  "step3_mapping": [
    {{"requirement": "...", "implementation": "..." or null}}
  ],
  "step4_gaps": ["gap 1", "gap 2", ...],
  "step5_compliance": "Full" | "Partial" | "Non-compliant",
  "summary": "Brief explanation of assessment"
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a GDPR compliance auditor. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        try:
            assessment = json.loads(response.choices[0].message.content)
            return {
                "policy_section": policy_section['metadata']['section_id'],
                "policy_title": policy_section['metadata']['section_title'],
                "gdpr_reference": gdpr_article['metadata']['section_id'],
                "gdpr_title": gdpr_article['metadata']['section_title'],
                "status": assessment.get('step5_compliance', 'Unknown'),
                "gaps": assessment.get('step4_gaps', []),
                "reasoning": assessment,
                "full_assessment": assessment
            }
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse assessment as JSON")
            return {
                "policy_section": policy_section['metadata']['section_id'],
                "gdpr_reference": gdpr_article['metadata']['section_id'],
                "status": "Error",
                "gaps": ["Failed to assess"],
                "reasoning": response.choices[0].message.content
            }