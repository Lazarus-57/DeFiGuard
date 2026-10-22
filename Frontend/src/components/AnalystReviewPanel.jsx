import React, { useState, useEffect } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  FileText,
  Clock,
  User,
  Save,
} from 'lucide-react';

const REASON_PRESETS = {
  cleared: [
    'Verified Exchange / Liquidity Router (e.g. Uniswap/Binance)',
    'Standard High-Volume Merchant / OTC Trading Desk',
    'Routine Internal Wallet Rebalancing (Self-Funded)',
    'Normal Payroll / Automated Batch Dispersion',
    'Low Confidence False Alarm (Standard Network Pattern)',
    'Other (Detailed in notes below)',
  ],
  escalated: [
    'Confirmed Smurfing Structuring (Fan-out / Fan-in Mule Network)',
    'Rapid Peel Chain Hop (Minimal Holding Time & Systematic Peeling)',
    'Circular Loop / Artificial Volume Wash Trading Cycle',
    'Known Exploiter / Stolen Funds Provenance Trail',
    'Immediate Asset Freezing & Suspicious Activity Report (SAR) Required',
    'Other (Detailed in notes below)',
  ],
};

export default function AnalystReviewPanel({ transaction, currentReview, onSaveReview, onClearReview }) {
  const [decision, setDecision] = useState(currentReview?.status || null);
  const [reason, setReason] = useState(currentReview?.reason || '');
  const [note, setNote] = useState(currentReview?.note || '');
  const [isSaved, setIsSaved] = useState(Boolean(currentReview));

  // Sync with currentReview when transaction changes
  useEffect(() => {
    if (currentReview) {
      setDecision(currentReview.status);
      setReason(currentReview.reason || '');
      setNote(currentReview.note || '');
      setIsSaved(true);
    } else {
      setDecision(null);
      setReason('');
      setNote('');
      setIsSaved(false);
    }
  }, [transaction?.tx_hash, currentReview]);

  if (!transaction) return null;

  const score = transaction.suspicion_score ?? 0;
  const isBorderline = transaction.aml_flag === 1 && score < 0.45;
  const isCritical = score >= 0.45;

  const handleDecisionSelect = (newDecision) => {
    setDecision(newDecision);
    setIsSaved(false);
    // Set default reason preset if empty
    if (!reason || !REASON_PRESETS[newDecision]?.includes(reason)) {
      setReason(REASON_PRESETS[newDecision][0]);
    }
  };

  const handleSave = () => {
    if (!decision) return;
    const reviewData = {
      status: decision,
      reason: reason || (decision === 'cleared' ? 'Cleared by Analyst' : 'Escalated by Analyst'),
      note: note.trim(),
      reviewedAt: new Date().toISOString(),
      analyst: 'Compliance Analyst (Local)',
    };
    onSaveReview(transaction.tx_hash, reviewData);
    setIsSaved(true);
  };

  const handleReset = () => {
    onClearReview(transaction.tx_hash);
    setDecision(null);
    setReason('');
    setNote('');
    setIsSaved(false);
  };

  return (
    <div className="analyst-panel card" style={{ marginTop: 14 }}>
      <div className="card__header" style={{ padding: '12px 18px', borderBottom: '1px solid var(--border-subtle)' }}>
        <div className="flex items-center gap-2">
          <User size={15} style={{ color: 'var(--accent-cyan)' }} />
          <span style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Analyst Triage &amp; Human-in-the-Loop Review
          </span>
        </div>

        <div>
          {isSaved && currentReview?.status === 'cleared' && (
            <span className="badge badge--safe" style={{ gap: 5 }}>
              <CheckCircle2 size={12} /> Cleared by Analyst
            </span>
          )}
          {isSaved && currentReview?.status === 'escalated' && (
            <span className="badge badge--danger" style={{ gap: 5 }}>
              <ShieldAlert size={12} /> Escalated (SAR Required)
            </span>
          )}
          {!isSaved && isBorderline && (
            <span className="badge badge--warn" style={{ gap: 5 }}>
              <AlertTriangle size={12} /> Triage Needed (Score: {score.toFixed(3)})
            </span>
          )}
          {!isSaved && isCritical && (
            <span className="badge badge--danger" style={{ gap: 5 }}>
              <ShieldAlert size={12} /> High Risk Alert
            </span>
          )}
          {!isSaved && !isBorderline && !isCritical && (
            <span className="badge badge--neutral">
              Auto-Cleared by Model (&lt; 0.200)
            </span>
          )}
        </div>
      </div>

      <div className="card__body" style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
        {/* Triage Guidance Text */}
        <p className="text-secondary" style={{ fontSize: '0.78rem', lineHeight: 1.5, margin: 0 }}>
          {isBorderline
            ? 'The model flagged this transaction due to moderate topological or velocity indicators, but confidence is borderline. Review the SHAP drivers above to approve or confirm escalation.'
            : 'Review the transaction details and SHAP feature breakdown to record an official compliance determination.'}
        </p>

        {/* Action Toggle Buttons */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <button
            type="button"
            className={`triage-btn triage-btn--clear ${decision === 'cleared' ? 'triage-btn--active' : ''}`}
            onClick={() => handleDecisionSelect('cleared')}
          >
            <div className="flex items-center gap-2">
              <ShieldCheck size={16} />
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontWeight: 600, fontSize: '0.8rem' }}>Clear / False Positive</div>
                <div style={{ fontSize: '0.68rem', opacity: 0.8 }}>Verify benign business activity</div>
              </div>
            </div>
          </button>

          <button
            type="button"
            className={`triage-btn triage-btn--escalate ${decision === 'escalated' ? 'triage-btn--active' : ''}`}
            onClick={() => handleDecisionSelect('escalated')}
          >
            <div className="flex items-center gap-2">
              <ShieldAlert size={16} />
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontWeight: 600, fontSize: '0.8rem' }}>Confirm / Escalate</div>
                <div style={{ fontSize: '0.68rem', opacity: 0.8 }}>Flag for SAR filing &amp; wallet freeze</div>
              </div>
            </div>
          </button>
        </div>

        {/* Dynamic Preset Reason Dropdown & Notes (When Decision Selected) */}
        {decision && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 4 }}>
            <div>
              <label className="label" style={{ marginBottom: 4, display: 'block', fontSize: '0.72rem' }}>
                Compliance Determination Reason
              </label>
              <select
                className="select-input w-full font-mono text-xs"
                value={reason}
                onChange={(e) => {
                  setReason(e.target.value);
                  setIsSaved(false);
                }}
                style={{ padding: '6px 10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}
              >
                {REASON_PRESETS[decision].map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="label" style={{ marginBottom: 4, display: 'block', fontSize: '0.72rem' }}>
                Investigator Case Notes (Optional)
              </label>
              <textarea
                className="textarea-input w-full font-mono text-xs"
                rows={2}
                placeholder="Enter forensic justification, related wallet clusters, or exchange ticket references..."
                value={note}
                onChange={(e) => {
                  setNote(e.target.value);
                  setIsSaved(false);
                }}
                style={{
                  padding: '8px 10px',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-default)',
                  borderRadius: 'var(--radius-sm)',
                  color: 'var(--text-primary)',
                  resize: 'vertical',
                }}
              />
            </div>

            {/* Bottom Actions Bar */}
            <div className="flex items-center justify-between" style={{ marginTop: 4 }}>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className="btn btn--primary btn--sm"
                  onClick={handleSave}
                  disabled={isSaved}
                  style={{ gap: 6 }}
                >
                  <Save size={13} />
                  <span>{isSaved ? 'Decision Recorded' : 'Save Determination'}</span>
                </button>

                {currentReview && (
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={handleReset}
                    style={{ gap: 6, opacity: 0.75 }}
                  >
                    <RotateCcw size={12} />
                    <span>Reset Review</span>
                  </button>
                )}
              </div>

              {isSaved && currentReview?.reviewedAt && (
                <div className="flex items-center gap-1.5 text-mono text-muted" style={{ fontSize: '0.68rem' }}>
                  <Clock size={11} />
                  <span>Recorded {new Date(currentReview.reviewedAt).toLocaleTimeString()}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
