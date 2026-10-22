import React, { useEffect, useRef, useState } from 'react';
import {
  Shield,
  ArrowRight,
  Zap,
  Cpu,
  Layers,
  Network,
  Clock,
  Activity,
  FileSearch,
  CheckCircle2,
  ChevronRight,
  Database,
  Upload,
  BarChart3,
  Sliders,
  HelpCircle,
  Eye,
  GitBranch,
  FileText,
  AlertTriangle,
} from 'lucide-react';

// Ambient particle constellation background
function AmbientConstellationCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let width = (canvas.width = canvas.offsetWidth);
    let height = (canvas.height = canvas.offsetHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = canvas.offsetWidth;
      height = canvas.height = canvas.offsetHeight;
    };
    window.addEventListener('resize', handleResize);

    const nodeCount = Math.floor(Math.min(width, 1200) / 32);
    const nodes = Array.from({ length: nodeCount }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      radius: Math.random() * 1.4 + 1,
      isFlagged: Math.random() < 0.1,
    }));

    let mouse = { x: -1000, y: -1000 };
    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    };
    window.addEventListener('mousemove', handleMouseMove);

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Soft ambient radial illumination
      const grad = ctx.createRadialGradient(width * 0.7, height * 0.35, 10, width * 0.7, height * 0.35, Math.max(width, height) * 0.5);
      grad.addColorStop(0, 'rgba(95, 208, 255, 0.025)');
      grad.addColorStop(1, 'rgba(5, 7, 11, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(width * 0.7, height * 0.35, Math.max(width, height) * 0.5, 0, Math.PI * 2);
      ctx.fill();

      // Connect nodes
      for (let i = 0; i < nodes.length; i++) {
        const n1 = nodes[i];
        n1.x += n1.vx;
        n1.y += n1.vy;
        if (n1.x < 0 || n1.x > width) n1.vx *= -1;
        if (n1.y < 0 || n1.y > height) n1.vy *= -1;

        // Subtle mouse pull
        const dxMouse = mouse.x - n1.x;
        const dyMouse = mouse.y - n1.y;
        const distMouse = Math.sqrt(dxMouse * dxMouse + dyMouse * dyMouse);
        if (distMouse < 120) {
          n1.x += (dxMouse / distMouse) * 0.25;
          n1.y += (dyMouse / distMouse) * 0.25;
        }

        for (let j = i + 1; j < nodes.length; j++) {
          const n2 = nodes[j];
          const dx = n1.x - n2.x;
          const dy = n1.y - n2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 100) {
            const alpha = (1 - dist / 100) * 0.12;
            ctx.strokeStyle = n1.isFlagged || n2.isFlagged
              ? `rgba(255, 84, 104, ${alpha * 1.2})`
              : `rgba(95, 208, 255, ${alpha})`;
            ctx.lineWidth = 0.75;
            ctx.beginPath();
            ctx.moveTo(n1.x, n1.y);
            ctx.lineTo(n2.x, n2.y);
            ctx.stroke();
          }
        }

        ctx.fillStyle = n1.isFlagged ? 'rgba(255, 84, 104, 0.65)' : 'rgba(154, 163, 178, 0.45)';
        ctx.beginPath();
        ctx.arc(n1.x, n1.y, n1.radius, 0, Math.PI * 2);
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return <canvas ref={canvasRef} className="landing-canvas-bg" />;
}

// Interactive Pattern Preview tabs
const PATTERN_TABS = [
  {
    id: 'smurfing',
    title: 'Structuring (Smurfing)',
    badge: 'High-Fanout Pattern',
    simpleTitle: 'Splitting Large Funds Across Multiple Accounts',
    description:
      'Instead of moving 100 ETH at once, an entity splits the deposit into dozens of small transactions across intermediate wallets to stay below monitoring thresholds, then recombines them at a destination wallet.',
    flow: [
      { step: '1. Source Deposit', detail: 'Single wallet initiates high-value outflow' },
      { step: '2. Dispersion', detail: 'Divided into 10+ temporary mule wallets' },
      { step: '3. Recombination', detail: 'Consolidated into single exit account' },
    ],
    detection:
      'Our Graph Neural Network detects the sudden fan-out and fan-in structure by analyzing 2-hop wallet connectivity, even if transaction amounts vary slightly.',
  },
  {
    id: 'peel',
    title: 'Peel Chains',
    badge: 'Sequential Hop Pattern',
    simpleTitle: 'Hopping Through Wallets While Peeling Small Amounts',
    description:
      'Funds are sent through a continuous chain of newly created addresses. At each hop, a small amount is peeled off to an exchange or private address, while the remainder continues hopping to blur the origin.',
    flow: [
      { step: '1. Initial Transfer', detail: 'Funds move to temporary Address B' },
      { step: '2. Continuous Peeling', detail: 'Small cut peeled; remainder forwarded to Address C' },
      { step: '3. Rapid Cadence', detail: 'Transfers occur seconds or minutes apart' },
    ],
    detection:
      'Temporal analysis measures the inter-transaction delay and pass-through velocity, flagging addresses that immediately forward funds with near-zero holding time.',
  },
  {
    id: 'circular',
    title: 'Circular Wash Loops',
    badge: 'Cyclic Transfer Pattern',
    simpleTitle: 'Routing Funds in Closed Loops',
    description:
      'Transactions are routed in circular cycles through several wallets back to the original owner or affiliated addresses. This is commonly used to fabricate trading volume or artificially clean fund history.',
    flow: [
      { step: '1. Wallet A sends to B', detail: 'Transfer through intermediary' },
      { step: '2. Wallet B sends to C', detail: 'Secondary transfer to affiliate' },
      { step: '3. Wallet C returns to A', detail: 'Closed loop transaction completed' },
    ],
    detection:
      'Graph cycle tracking calculates node centrality and recurrent loops, uncovering wallets that repeatedly transact with their own cluster.',
  },
  {
    id: 'shap',
    title: 'Why Was It Flagged? (SHAP)',
    badge: 'Transparent Explanation',
    simpleTitle: 'Clear, Human-Readable Risk Drivers',
    description:
      'Every alert in DeFIGuard comes with a breakdown of exactly why the model made its decision. Compliance officers and investigators can see which factors contributed to the score rather than trusting a black box.',
    flow: [
      { step: 'Rapid Forwarding', detail: 'Inter-transfer interval under 45s (+0.34 risk)' },
      { step: 'Graph Centrality', detail: 'Wallet connects to known hub accounts (+0.28 risk)' },
      { step: 'Balance Drain', detail: '98% of wallet funds moved in single block (+0.16 risk)' },
    ],
    detection:
      'Uses TreeSHAP feature attributions so every decision can be cited in compliance reports and regulatory filings.',
  },
];

export default function LandingPage({ onNavigate, onQuickDemo, loadingDemo }) {
  const [activePattern, setActivePattern] = useState(PATTERN_TABS[0].id);

  const selectedPattern = PATTERN_TABS.find((p) => p.id === activePattern) || PATTERN_TABS[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 40 }}>
      {/* ── 1. Hero Section ── */}
      <section className="landing-hero">
        <AmbientConstellationCanvas />

        <div className="landing-hero-content">
          <div className="landing-telemetry-badge">
            <div className="landing-telemetry-dot" />
            <span>ETHEREUM AML DETECTION · GRAPH AI &amp; GRADIENT BOOSTING</span>
          </div>

          <h1 className="landing-hero-title">
            Detect Ethereum transactions with clarity.
          </h1>

          <p className="landing-hero-sub">
            DeFIGuard analyzes blockchain transactions to flag money laundering patterns like
            smurfing, peel chains, and circular trading — and shows you exactly why each alert was raised.
          </p>

          {/* Quick-Action Entry Point */}
          <div className="landing-quick-card">
            <div className="landing-quick-card__info">
              <span className="landing-quick-card__label">GET STARTED IN ONE CLICK</span>
              <p className="landing-quick-card__text">
                Upload your own Ethereum transaction CSV, or run our sample benchmark dataset to inspect the live dashboard, interactive wallet graph, and risk explanations.
              </p>
            </div>

            <div className="landing-actions">
              <button
                className="btn btn--primary"
                onClick={() => onNavigate('upload')}
              >
                <Upload size={14} />
                <span>Upload CSV Dataset</span>
                <ArrowRight size={14} />
              </button>

              <button
                className="btn btn--secondary"
                onClick={onQuickDemo}
                disabled={loadingDemo}
              >
                <Zap size={14} style={{ color: 'var(--accent-cyan)' }} />
                <span>{loadingDemo ? 'Running Pipeline...' : 'Try Demo Dataset (120 Txns)'}</span>
              </button>

              <button
                className="btn btn--ghost"
                onClick={() => onNavigate('model')}
              >
                <span>How the Model Works</span>
                <ChevronRight size={14} />
              </button>
            </div>
          </div>

          {/* Key Benchmark Metrics Strip */}
          <div className="landing-kpi-strip">
            <div className="landing-kpi-item">
              <span className="landing-kpi-val" style={{ color: 'var(--accent-emerald)' }}>98.6%</span>
              <span className="landing-kpi-label">Test Set Recall</span>
              <span className="landing-kpi-hint">Catches 98.6% of laundering flows in benchmark tests</span>
            </div>

            <div className="landing-kpi-item">
              <span className="landing-kpi-val" style={{ color: 'var(--accent-cyan)' }}>75.0%</span>
              <span className="landing-kpi-label">Precision @ Top 0.1%</span>
              <span className="landing-kpi-hint">3 in 4 top-priority alerts are true positives</span>
            </div>

            <div className="landing-kpi-item">
              <span className="landing-kpi-val" style={{ color: 'var(--text-primary)' }}>51 Signals</span>
              <span className="landing-kpi-label">Multi-Modal Features</span>
              <span className="landing-kpi-hint">Graph connections, transfer timing, and balance dynamics</span>
            </div>

            <div className="landing-kpi-item">
              <span className="landing-kpi-val" style={{ color: 'var(--accent-purple)' }}>Explainable</span>
              <span className="landing-kpi-label">SHAP Feature Attribution</span>
              <span className="landing-kpi-hint">Clear reason breakdown for every flagged transaction</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── 2. What You Can Do (Core Capabilities) ── */}
      <section>
        <div className="section-header">
          <div>
            <h2 className="section-title">Built for thorough investigations, simplified for everyday use</h2>
            <p className="section-sub">
              Everything you need to audit transaction ledgers, investigate suspicious wallets, and explain findings to stakeholders.
            </p>
          </div>
        </div>

        <div className="capabilities-grid">
          <div className="capability-card">
            <div className="capability-card__icon" style={{ color: 'var(--accent-rose)' }}>
              <Shield size={20} />
            </div>
            <h3 className="capability-card__title">Automated Risk Scoring</h3>
            <p className="capability-card__desc">
              Every transaction receives a calibrated risk score between 0.00 and 1.00 based on its graph connectivity, transfer intervals, and balance flows.
            </p>
            <div className="capability-card__tag">Calibrated XGBoost Classifier</div>
          </div>

          <div className="capability-card">
            <div className="capability-card__icon" style={{ color: 'var(--accent-cyan)' }}>
              <GitBranch size={20} />
            </div>
            <h3 className="capability-card__title">Interactive Wallet Network</h3>
            <p className="capability-card__desc">
              Visualize transaction relationships in an interactive wallet graph. Trace how funds hop between wallets, highlight mule hubs, and isolate high-risk clusters.
            </p>
            <div className="capability-card__tag">Force-Directed Graph Visualization</div>
          </div>

          <div className="capability-card">
            <div className="capability-card__icon" style={{ color: 'var(--accent-purple)' }}>
              <HelpCircle size={20} />
            </div>
            <h3 className="capability-card__title">Transparent Explanations</h3>
            <p className="capability-card__desc">
              No black boxes. Clicking any flagged transaction opens a plain-English explanation of the specific factors that contributed to its risk score.
            </p>
            <div className="capability-card__tag">TreeSHAP Feature Attributions</div>
          </div>

          <div className="capability-card">
            <div className="capability-card__icon" style={{ color: 'var(--accent-amber)' }}>
              <BarChart3 size={20} />
            </div>
            <h3 className="capability-card__title">Triage &amp; Filtering</h3>
            <p className="capability-card__desc">
              Filter alerts by risk severity, pattern type, or specific wallet addresses to prioritize your investigation queue and reduce manual review time.
            </p>
            <div className="capability-card__tag">Risk-Tiered Alert Queuing</div>
          </div>
        </div>
      </section>

      {/* ── 3. Interactive Pattern Walkthrough (Explains Laundering Simply) ── */}
      <section className="card">
        <div className="card__header">
          <div>
            <span className="card__title">
              <Layers /> How DeFIGuard Identifies Laundering Patterns
            </span>
            <p className="card__desc" style={{ marginTop: 4 }}>
              Click any pattern below to see how funds move and how our detection pipeline catches them.
            </p>
          </div>
        </div>

        <div className="card__body" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Pattern Selector Tabs */}
          <div className="pattern-tabs">
            {PATTERN_TABS.map((tab) => (
              <button
                key={tab.id}
                className={`pattern-tab-btn ${activePattern === tab.id ? 'pattern-tab-btn--active' : ''}`}
                onClick={() => setActivePattern(tab.id)}
              >
                <span>{tab.title}</span>
              </button>
            ))}
          </div>

          {/* Active Pattern Display Box */}
          <div className="pattern-detail-box">
            <div className="pattern-detail-header">
              <div>
                <span className="badge badge--cyan" style={{ fontSize: '0.65rem' }}>
                  {selectedPattern.badge}
                </span>
                <h4 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: 8 }}>
                  {selectedPattern.simpleTitle}
                </h4>
              </div>
            </div>

            <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              {selectedPattern.description}
            </p>

            {/* Step-by-Step Flow */}
            <div className="pattern-flow-grid">
              {selectedPattern.flow.map((item, idx) => (
                <div key={idx} className="pattern-flow-card">
                  <span className="pattern-flow-step">{item.step}</span>
                  <span className="pattern-flow-detail">{item.detail}</span>
                </div>
              ))}
            </div>

            {/* How we catch it */}
            <div className="pattern-catch-box">
              <CheckCircle2 size={16} style={{ color: 'var(--accent-emerald)', flexShrink: 0, marginTop: 2 }} />
              <div>
                <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.78rem' }}>
                  Detection Approach:{' '}
                </span>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
                  {selectedPattern.detection}
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── 4. The 3-Step Pipeline (Under the Hood) ── */}
      <section>
        <div className="section-header">
          <div>
            <h2 className="section-title">How It Works Under the Hood</h2>
            <p className="section-sub">A simple, transparent 3-step pipeline from raw CSV to explained alerts.</p>
          </div>
        </div>

        <div className="pipeline-steps-grid">
          <div className="pipeline-step-card">
            <div className="pipeline-step-num">STEP 01</div>
            <div className="pipeline-step-title">
              <Network size={16} style={{ color: 'var(--accent-rose)' }} />
              <span>Wallet Graph Construction</span>
            </div>
            <p className="pipeline-step-desc">
              Ingests raw transaction pairs, builds a wallet-to-wallet network, and runs GraphSAGE to generate 32-dimensional structural embeddings that summarize each wallet's 2-hop neighborhood.
            </p>
          </div>

          <div className="pipeline-step-card">
            <div className="pipeline-step-num">STEP 02</div>
            <div className="pipeline-step-title">
              <Clock size={16} style={{ color: 'var(--accent-amber)' }} />
              <span>Timing &amp; Velocity Dynamics</span>
            </div>
            <p className="pipeline-step-desc">
              Calculates Normalized Transaction Scores (NTS) measuring inter-block transfer delay, transaction frequency bursts, and rapid pass-through holding periods.
            </p>
          </div>

          <div className="pipeline-step-card">
            <div className="pipeline-step-num">STEP 03</div>
            <div className="pipeline-step-title">
              <Cpu size={16} style={{ color: 'var(--accent-cyan)' }} />
              <span>Scoring &amp; Risk Explanation</span>
            </div>
            <p className="pipeline-step-desc">
              Evaluates all 51 features with calibrated XGBoost to assign a risk score. Calculates exact SHAP values so every flag has an understandable explanation for compliance audits.
            </p>
          </div>
        </div>
      </section>

      {/* ── 5. Simple, Inviting Footer CTA ── */}
      <section className="landing-footer-cta">
        <div>
          <h3 className="landing-footer-title">Ready to analyze Ethereum transactions?</h3>
          <p className="landing-footer-desc">
            Upload an exported transaction CSV, or run our sample benchmark dataset to test the full pipeline in seconds.
          </p>
        </div>

        <div className="landing-actions">
          <button
            className="btn btn--primary"
            onClick={() => onNavigate('upload')}
          >
            <Upload size={14} />
            <span>Upload CSV Dataset</span>
            <ArrowRight size={14} />
          </button>

          <button
            className="btn btn--secondary"
            onClick={onQuickDemo}
            disabled={loadingDemo}
          >
            <Zap size={14} style={{ color: 'var(--accent-cyan)' }} />
            <span>{loadingDemo ? 'Running Pipeline...' : 'Run Benchmark Dataset'}</span>
          </button>
        </div>
      </section>
    </div>
  );
}
