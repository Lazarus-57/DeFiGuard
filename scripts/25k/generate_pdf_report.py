from fpdf import FPDF
from pathlib import Path

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(42, 157, 143) # teal color
        self.cell(0, 10, 'DeFIGuard - 25k Dataset & Model Evaluation Report', border=False, align='C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(38, 70, 83) # dark blue
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, text):
        self.set_font('helvetica', '', 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln()

def main():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Introduction
    pdf.chapter_title('1. The 25k Dataset: Composition')
    body_text1 = (
        "In this phase of the project, we scaled our dataset to approximately 25,000 transactions. "
        "Our goal is to build an Anti-Money Laundering (AML) detection pipeline that can accurately "
        "find suspicious transactions hiding among legitimate ones.\n\n"
        "To train and test the model, we injected two complex, real-world money laundering patterns into the data:\n"
        " - Peel Chains: A long sequence of transactions where a small amount is 'peeled' off at each step, "
        "acting like a chain of drops to slowly cash out funds.\n"
        " - Smurfing: A fan-out and fan-in pattern where a single source wallet distributes funds to multiple "
        "'mule' wallets, which then forward the money to a central collector wallet. This hides the large transfer.\n\n"
        "Dataset Breakdown:\n"
        " - Total Transactions: 26,854\n"
        " - Normal (Legitimate): 25,000 (93.1%)\n"
        " - Peel Chains: 1,050 (3.9%)\n"
        " - Smurfing: 804 (3.0%)\n"
    )
    pdf.chapter_body(body_text1)

    # Insert pie chart
    img_path = Path("reports/25k/Pipeline Overview/composition_pie_25k.png")
    if img_path.exists():
        pdf.image(str(img_path), w=120, x='C')
        pdf.ln(5)

    pdf.add_page()
    pdf.chapter_title('2. Model Evaluation and Layers')
    body_text2 = (
        "How do we know if our model is doing a good job? Since suspicious transactions only make up about 7% of the data, "
        "traditional 'Accuracy' can be misleading (a model that guesses 'Normal' every time would still be 93% accurate!). "
        "Instead, we use PR-AUC (Precision-Recall Area Under Curve), which heavily penalizes false alarms and rewards "
        "correctly catching the rare suspicious events.\n\n"
        "We tested our models in three layers of increasing complexity:\n\n"
        "Layer 1: Traditional Baselines\n"
        "We started with standard machine learning models using basic transaction features (like amounts and simple network degrees).\n"
        " - Random Forest: 0.8603 PR-AUC\n"
        " - Logistic Regression: 0.9015 PR-AUC\n"
        " - XGBoost: 0.9311 PR-AUC\n"
        "XGBoost was the clear winner here, setting a strong baseline.\n\n"
        "Layer 2: Adding Graph Context (GraphSAGE)\n"
        "Next, we gave the model 'Graph Context'. We used an advanced Graph Neural Network (GraphSAGE) to map the shape "
        "and structure of the wallets' transaction history. Did giving XGBoost this structural awareness help?\n"
        " - XGBoost Baseline: 0.9250 PR-AUC\n"
        " - XGBoost + GraphSAGE: 0.9351 PR-AUC (+0.0101)\n"
        "Yes! The graph features helped the model better recognize the structural signatures of smurfing and peel chains.\n\n"
        "Layer 3: Adding Temporal Intelligence (NTS)\n"
        "Finally, we tested 'Temporal Intelligence'. Money laundering is rarely instantaneous; it involves deliberate delays "
        "and time-spreads. We engineered Network Time Spread (NTS) features to capture the rhythm and timing of transactions.\n"
        " - XGBoost Baseline: 0.9250 PR-AUC\n"
        " - XGBoost + NTS: 0.9365 PR-AUC (+0.0114)\n"
        "This yielded the biggest jump. The model learned that the timing of transactions is just as suspicious as the structure."
    )
    pdf.chapter_body(body_text2)

    pdf.chapter_title('3. The Final Winner')
    body_text3 = (
        "The final winner is XGBoost augmented with Temporal Intelligence (NTS), achieving the highest PR-AUC score of 0.9365. \n\n"
        "Why did this win?\n"
        "While Graph Neural Networks (GraphSAGE) successfully captured the spatial layout of laundering operations, the time-based "
        "features proved to be an even stronger signal. Both peel chains and smurfing rely on specific sequence timings "
        "to move funds securely. By giving XGBoost awareness of these temporal rhythms, the model became highly effective "
        "at distinguishing complex money laundering from normal DeFi activity."
    )
    pdf.chapter_body(body_text3)

    out_path = Path("reports/25k/DeFIGuard_25k_Report.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))

if __name__ == '__main__':
    main()
