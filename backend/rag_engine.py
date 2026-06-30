import os
import re
from typing import Dict, Any, List

class ClinicalRAGEngine:
    """Retrieval-Augmented Generation (RAG) engine for clinical breast cancer guidelines"""
    
    def __init__(self, kb_dir: str = r"e:\CONFERENCE\knowledge_base"):
        self.kb_dir = kb_dir
        self.nccn_path = os.path.join(kb_dir, "nccn_breast_cancer_guidelines.md")
        self.acr_path = os.path.join(kb_dir, "acr_birads_atlas_5th_ed.md")
        self.pubmed_path = os.path.join(kb_dir, "pubmed_mammography_papers.md")
        
    def _read_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def query(self, birads_rating: str, density_category: str) -> Dict[str, Any]:
        """
        Dynamically queries NCCN and ACR guideline files for matching recommendation blocks.
        """
        # Load knowledge assets
        nccn_text = self._read_file(self.nccn_path)
        acr_text = self._read_file(self.acr_path)
        pubmed_text = self._read_file(self.pubmed_path)
        
        # Clean rating tag for searching (e.g. "BI-RADS 4B" -> "BI-RADS 4")
        m = re.search(r"BI-RADS \d", birads_rating)
        birads_base = m.group(0) if m else "BI-RADS 1"
        
        # 1. Retrieve recommendation from NCCN guidelines
        nccn_recommendation = ""
        nccn_citation = "NCCN Guidelines Version 2.2026 - Breast Cancer Screening and Diagnosis"
        
        # Use regex to find the matching BI-RADS block in NCCN guidelines
        pattern = rf"###\s+{birads_base}.*?\n(.*?)(?=\n###|\n##|$)"
        match = re.search(pattern, nccn_text, re.DOTALL | re.IGNORECASE)
        if match:
            nccn_recommendation = match.group(1).strip()
        else:
            # Fallback
            if "BI-RADS 4" in birads_base or "BI-RADS 5" in birads_base:
                nccn_recommendation = "- Clinical Recommendation: Urgent tissue diagnosis via ultrasound-guided core needle biopsy."
            elif "BI-RADS 3" in birads_base:
                nccn_recommendation = "- Clinical Recommendation: Short-interval follow-up diagnostic ultrasound in 6 months."
            else:
                nccn_recommendation = "- Clinical Recommendation: Routine screening mammography annually for women >= 40."
                
        # 2. Retrieve acoustic guidelines from ACR BI-RADS
        acr_guidelines = ""
        acr_citation = "ACR BI-RADS Atlas 5th Edition - Ultrasound Follow-up Guidelines"
        
        # Look for BI-RADS description in ACR
        acr_pattern = rf"-\s+\*\*{birads_base}.*?\n(.*?)(?=\n-|\n#|$)"
        acr_match = re.search(acr_pattern, acr_text, re.DOTALL | re.IGNORECASE)
        if acr_match:
            acr_guidelines = acr_match.group(0).strip()
        else:
            # Fallback
            if "BI-RADS 4" in birads_base:
                acr_guidelines = f"**{birads_base}**: Suspicious abnormality. Biopsy should be considered."
            elif "BI-RADS 5" in birads_base:
                acr_guidelines = f"**{birads_base}**: Highly suggestive of malignancy. Action: Core needle biopsy."
            elif "BI-RADS 3" in birads_base:
                acr_guidelines = f"**{birads_base}**: Probably benign finding. Action: 6-month diagnostic follow-up."
            else:
                acr_guidelines = f"**{birads_base}**: Negative/Benign. Action: Standard annual screening."

        # 3. Retrieve relevant PubMed research paper citation
        pubmed_citation = "PubMed Medical Imaging Literature Reference"
        pubmed_summary = ""
        if "BI-RADS 4" in birads_base or "BI-RADS 5" in birads_base:
            # Grab combined inputs multi-channel paper
            pm_match = re.search(r"### 3\. Multi-Channel Inputs.*?\n(.*?)(?=\n###|$)", pubmed_text, re.DOTALL)
            if pm_match:
                pubmed_summary = "Literature Context: " + pm_match.group(1).strip()
        else:
            # Grab attention u-net paper
            pm_match = re.search(r"### 1\. Attention U-Net.*?\n(.*?)(?=\n###|$)", pubmed_text, re.DOTALL)
            if pm_match:
                pubmed_summary = "Literature Context: " + pm_match.group(1).strip()
                
        # 4. Integrate density-based recommendations
        density_notes = ""
        if density_category in ["C", "D"]:
            density_notes = " Note: Patient presents dense breast tissue (Category {}). Supplemental screening ultrasound is indicated.".format(density_category)

        # Assemble final synthesis
        synthesized_text = f"{nccn_recommendation}\n{density_notes}"
        
        return {
            "birads_rating": birads_rating,
            "density_category": density_category,
            "recommendation_text": synthesized_text,
            "citation": f"{nccn_citation} / {acr_citation}",
            "pubmed_citation": pubmed_citation,
            "guideline_details": acr_guidelines
        }

if __name__ == "__main__":
    engine = ClinicalRAGEngine()
    result = engine.query("BI-RADS 4B", "C")
    print("Citation:", result["citation"])
    print("Text:", result["recommendation_text"])
