import os
from pathlib import Path
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'DeFIGuard: Project Lifecycle & Comprehensive Evaluation Report', new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, f' {title}', new_x="LMARGIN", new_y="NEXT", align='L', fill=True)
        self.ln(4)

    def chapter_subtitle(self, subtitle):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, subtitle, new_x="LMARGIN", new_y="NEXT", align='L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, body)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, f"- {text}")
        self.ln(1)


def generate_extended_pdf(output_path):
    pdf = PDFReport()
    pdf.add_page()
    
    # ---------------- 1. Project Idea ----------------
    pdf.chapter_title('1. The Project Idea: DeFIGuard')
    pdf.chapter_body(
        "Blockchain transactions are often described as transparent, but visibility does not automatically "
        "translate into understanding. Modern cryptocurrency laundering operations frequently involve "
        "multiple wallets, rapid fund transfers, transaction layering, fund splitting, intermediary wallets, "
        "and coordinated movement patterns designed to make tracing incredibly difficult."
    )
    pdf.chapter_body(
        "DeFIGuard was conceptualized to solve this tracking problem in Decentralized Finance (DeFi) ecosystems. "
        "Traditional tracing relies on simple heuristics that are easily defeated by complex laundering topologies. "
        "Our goal was to build a multi-layered machine learning architecture capable of reconstructing and "
        "detecting suspicious fund movements by combining traditional features with structural Graph Neural Networks (GNNs) "
        "and Node Temporal Spread (NTS) velocity tracking."
    )
    
    # ---------------- 2. Methodology & Architecture ----------------
    pdf.chapter_title('2. Methodology & Three-Layer Architecture')
    pdf.chapter_body(
        "Because labeled money laundering data is notoriously scarce in the crypto space, we built a robust "
        "synthetic pattern generator. We systematically injected highly realistic laundering topologies (Peel Chains, "
        "Smurfing Clusters, and Circular Rings) into real Ethereum transaction datasets. We then evaluated these datasets "
        "using a three-layered approach:"
    )
    pdf.bullet_point("Layer 1 (Baseline ML): XGBoost to analyze standard transactional features (amounts, transaction counts, balances).")
    pdf.bullet_point("Layer 2 (Structural Intelligence): GraphSAGE (a GNN) to map the entire transaction graph, allowing the model to 'see' clusters and structural topologies.")
    pdf.bullet_point("Layer 3 (Temporal Intelligence): NTS to measure the 'velocity' and burstiness of transactions, catching launderers who move funds rapidly in coordinated attacks.")
    pdf.ln(5)

    # ---------------- 3. The 5k Dataset Prototype ----------------
    pdf.chapter_title('3. The 5k Dataset Prototype (Initial Proof of Concept)')
    pdf.chapter_body(
        "The project began with a small-scale 5k transaction dataset. This phase acted as our sandbox to test "
        "the synthetic injection pipeline and the baseline machine learning setup. We injected simple Peel Chains "
        "into the data and successfully verified that our data preparation, feature engineering, and model training "
        "pipelines executed without errors. This proved our engineering foundation was solid, paving the way for scale."
    )
    
    # ---------------- 4. The 25k Dataset (Scale & Discovery) ----------------
    pdf.chapter_title('4. The 25k Dataset (Scaling Up & Critical Discoveries)')
    pdf.chapter_body(
        "We scaled the data up to 25,000 transactions and introduced both Peel Chains and Smurfing patterns. "
        "While the pipeline ran successfully, our analysis revealed two massive flaws that distorted the results:"
    )
    pdf.bullet_point("The Base Rate Illusion: Because we injected over 7,000 illicit transactions, the dataset had an unrealistic 22.4% positive rate. In real AML, illicit activity is typically under 1%. This made the detection look artificially easy (Baseline PR-AUC ~0.89).")
    pdf.bullet_point("Temporal Data Leakage: Synthetic transactions were generated with random timestamps across the dataset's entire range. This randomness artificially inflated the Temporal (NTS) spread, causing severe data leakage.")
    pdf.chapter_body(
        "The 25k dataset was an incredible learning moment, proving that functional pipelines do not always equate to "
        "mathematically sound results."
    )
    pdf.add_page()
    
    # ---------------- 5. The 100k Dataset (Finalized & Corrected) ----------------
    pdf.chapter_title('5. The 100k Dataset (Corrected & Realistic Evaluation)')
    pdf.chapter_body(
        "For our final evaluation, we generated a 100,000-transaction dataset. We dramatically diluted the illicit "
        "patterns to achieve a realistic class imbalance (6% - 9% positive rate). Crucially, we fixed the Temporal "
        "Leakage bug by strictly enforcing chronological rules: Peel chains executed with realistic 1-5 minute delays, "
        "and Smurfing clusters executed in tightly coordinated batches."
    )
    
    pdf.chapter_subtitle('Key Findings from the 100k Evaluation:')
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 6, 'A) Peel Chains (Linear Layering)', new_x="LMARGIN", new_y="NEXT")
    pdf.chapter_body(
        "GraphSAGE caused a performance drop due to oversmoothing on the linear chains. Temporal features also "
        "struggled because realistic peel chains look temporally similar to regular sequential DeFi usage. Baseline XGBoost proved best here."
    )
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 6, 'B) Smurfing (Structural Fan-Outs/Fan-Ins)', new_x="LMARGIN", new_y="NEXT")
    pdf.chapter_body(
        "GraphSAGE provided a strong lift (+0.0539 PR-AUC) because the fan-out/fan-in topology forms distinct structural clusters. "
        "NTS Temporal features provided a massive lift (+0.1064 PR-AUC) because the coordinated, batched nature of smurfing "
        "creates an unmistakable temporal burst signature."
    )

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 6, 'C) Circular Rings (Loops)', new_x="LMARGIN", new_y="NEXT")
    pdf.chapter_body(
        "GraphSAGE failed heavily on circular loops, as neighborhood aggregations get confused when funds loop back "
        "to the source. NTS provided only a slight lift."
    )

    # ---------------- 6. Current Position ----------------
    pdf.chapter_title('6. Current Position: Architecture Validated')
    pdf.chapter_body(
        "As of this moment, the Model Choice & Architecture Validation phase is fully complete. We have proven our core hypothesis: "
        "No single algorithm is a silver bullet. Traditional models excel at linear trails, GNNs excel at structural clusters, and "
        "Temporal Intelligence excels at catching rapid coordination. We know exactly what architecture is required to catch modern launderers."
    )
    
    # ---------------- 7. Next Steps & Final Deliverable ----------------
    pdf.chapter_title('7. Next Steps & Final Deliverable')
    pdf.chapter_body(
        "With the architecture validated, the remaining work transitions from experimentation to finalization:"
    )
    pdf.bullet_point("1. Master Hybrid Model: We will combine the Baseline features, the GraphSAGE embeddings, and the NTS Temporal features into a single, unified 'Master' model.")
    pdf.bullet_point("2. Final Blind Testing: We will execute a final evaluation of this Master Hybrid Model on a completely blind, held-out Test dataset to generate the definitive performance metrics for the academic paper/conference.")
    pdf.bullet_point("3. Final Deliverable: A comprehensive codebase and academic paper proving that hybrid structural-temporal tracking vastly outperforms traditional heuristics in DeFi AML investigations.")
    
    pdf.output(output_path)


if __name__ == "__main__":
    reports_dir = Path("reports/100k")
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / "DeFIGuard_Comprehensive_Timeline_Report.pdf"
    
    generate_extended_pdf(str(pdf_path))
    print(f"Created timeline PDF at {pdf_path}")
