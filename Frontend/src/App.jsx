import React, { useState } from 'react';
import {
  Shield,
  Upload,
  LayoutDashboard,
  Table2,
  GitBranch,
  Cpu,
  FileSearch,
} from 'lucide-react';

import UploadPanel from './components/UploadPanel';
import DashboardOverview from './components/DashboardOverview';
import AlertsTable from './components/AlertsTable';
import NetworkGraph from './components/NetworkGraph';
import ModelInfoPanel from './components/ModelInfoPanel';

const TABS = [
  { id: 'upload', label: 'Upload', icon: Upload },
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'alerts', label: 'Alerts', icon: Table2 },
  { id: 'network', label: 'Network Graph', icon: GitBranch },
  { id: 'model', label: 'Model Info', icon: Cpu },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null); // full API response

  const handleResults = (apiResponse) => {
    setData(apiResponse);
    setActiveTab('dashboard');
  };

  const hasResults = data && data.results && data.results.length > 0;

  // Restrict tabs to only upload + model when no data
  const availableTabs = hasResults
    ? TABS
    : TABS.filter((t) => t.id === 'upload' || t.id === 'model');

  return (
    <>
      {/* ── Top Navigation ── */}
      <nav className="topnav">
        <div className="topnav__brand">
          <Shield className="topnav__brand-icon" />
          <span className="topnav__brand-name">DeFIGuard</span>
        </div>

        <div className="topnav__tabs">
          {availableTabs.map((tab) => (
            <button
              key={tab.id}
              className={`topnav__tab ${activeTab === tab.id ? 'topnav__tab--active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <tab.icon />
              {tab.label}
              {tab.id === 'alerts' && hasResults && (
                <span className="badge badge--danger" style={{ marginLeft: 4, padding: '1px 6px', fontSize: '0.65rem' }}>
                  {data.summary?.flagged_transactions ?? 0}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="topnav__status">
          {hasResults && (
            <>
              <FileSearch size={14} />
              <span>{data.count} txns analyzed</span>
              <span style={{ margin: '0 4px', opacity: 0.3 }}>|</span>
            </>
          )}
          <div className="topnav__status-dot" />
          <span>API Connected</span>
        </div>
      </nav>

      {/* ── Main Content Area ── */}
      <main className="main-content">
        {activeTab === 'upload' && (
          <div>
            <div className="section-header mb-6">
              <div>
                <h1 className="section-title" style={{ fontSize: '1.4rem', marginBottom: 4 }}>
                  AML Transaction Investigation
                </h1>
                <p className="text-sm text-muted">
                  Upload a CSV of DeFi transactions for real-time ML-powered money laundering detection with explainable AI
                </p>
              </div>
            </div>
            <UploadPanel
              onResults={handleResults}
              loading={loading}
              setLoading={setLoading}
            />
          </div>
        )}

        {activeTab === 'dashboard' && hasResults && (
          <div>
            <div className="section-header mb-6">
              <div>
                <h1 className="section-title">Investigation Dashboard</h1>
                <p className="text-sm text-muted">
                  {data.filename ? `File: ${data.filename} · ` : ''}
                  {data.count} transactions analyzed
                </p>
              </div>
              <button
                className="btn btn--secondary btn--sm"
                onClick={() => { setData(null); setActiveTab('upload'); }}
              >
                <Upload size={14} /> New Upload
              </button>
            </div>
            <DashboardOverview
              summary={data.summary}
              resultCount={data.count}
            />
          </div>
        )}

        {activeTab === 'alerts' && hasResults && (
          <div>
            <div className="section-header mb-6">
              <div>
                <h1 className="section-title">Transaction Alerts</h1>
                <p className="text-sm text-muted">
                  Click any row to inspect its SHAP explanation and risk decomposition
                </p>
              </div>
            </div>
            <AlertsTable results={data.results} />
          </div>
        )}

        {activeTab === 'network' && hasResults && (
          <div>
            <div className="section-header mb-6">
              <div>
                <h1 className="section-title">Wallet Network Graph</h1>
                <p className="text-sm text-muted">
                  Interactive visualization of wallet-to-wallet transaction flows. Red = suspicious, Cyan = clean.
                </p>
              </div>
            </div>
            <div className="card">
              <div className="card__body">
                <NetworkGraph graphData={data.graph} />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'model' && (
          <div>
            <div className="section-header mb-6">
              <div>
                <h1 className="section-title">Model Architecture & Performance</h1>
                <p className="text-sm text-muted">
                  Phase 1 trained hybrid model — XGBoost + GraphSAGE GNN + NTS Temporal Intelligence
                </p>
              </div>
            </div>
            <ModelInfoPanel />
          </div>
        )}
      </main>
    </>
  );
}
