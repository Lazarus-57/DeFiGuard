import React, { useState } from 'react';
import {
  Shield,
  Upload,
  LayoutDashboard,
  Table2,
  GitBranch,
  Cpu,
  FileSearch,
  RefreshCw,
  Terminal,
} from 'lucide-react';

import LandingPage from './components/LandingPage';
import UploadPanel from './components/UploadPanel';
import DashboardOverview from './components/DashboardOverview';
import AlertsTable from './components/AlertsTable';
import NetworkGraph from './components/NetworkGraph';
import ModelInfoPanel from './components/ModelInfoPanel';
import { loadSample } from './api';

const TABS = [
  { id: 'overview', label: 'Overview', icon: Shield },
  { id: 'upload', label: 'Analyze', icon: Upload },
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'alerts', label: 'Alerts', icon: Table2 },
  { id: 'network', label: 'Network', icon: GitBranch },
  { id: 'model', label: 'Model', icon: Cpu },
];

function PageHeader({ title, subtitle, action }) {
  return (
    <div className="section-header">
      <div>
        <h1 className="section-title">{title}</h1>
        {subtitle && <p className="section-sub">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(false);
  const [loadingDemo, setLoadingDemo] = useState(false);
  const [data, setData] = useState(null); // full API response
  const [drillDownFilter, setDrillDownFilter] = useState(null); // { type: 'pattern' | 'score' | 'triage', value: string }
  const [analystReviews, setAnalystReviews] = useState({}); // { [txHash]: { status: 'cleared'|'escalated', reason, notes, timestamp, analyst } }

  const handleSaveReview = (txHash, reviewData) => {
    setAnalystReviews((prev) => ({
      ...prev,
      [txHash]: reviewData,
    }));
  };

  const handleClearReview = (txHash) => {
    setAnalystReviews((prev) => {
      const next = { ...prev };
      delete next[txHash];
      return next;
    });
  };

  const handleResults = (apiResponse) => {
    setData(apiResponse);
    setDrillDownFilter(null);
    setAnalystReviews({});
    setActiveTab('dashboard');
  };

  const handleDrillDown = (filterObj) => {
    setDrillDownFilter(filterObj);
    setActiveTab('alerts');
  };

  const handleQuickDemo = async () => {
    setLoadingDemo(true);
    try {
      const res = await loadSample('sample_100k_test');
      handleResults(res);
    } catch (err) {
      console.error('Failed to load sample benchmark dataset', err);
    } finally {
      setLoadingDemo(false);
    }
  };

  const hasResults = data && data.results && data.results.length > 0;

  // Overview, Upload, and Model are available until a dataset is analyzed
  const availableTabs = hasResults
    ? TABS
    : TABS.filter((t) => t.id === 'overview' || t.id === 'upload' || t.id === 'model');

  return (
    <>
      <nav className="topnav">
        <div
          className="topnav__brand"
          onClick={() => setActiveTab('overview')}
          style={{ cursor: 'pointer' }}
          role="button"
          tabIndex={0}
        >
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
              <span>{tab.label}</span>
              {tab.id === 'alerts' && hasResults && (
                <span className="badge badge--danger">
                  {(data.summary?.flagged_transactions ?? 0).toLocaleString()}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="topnav__status">
          {hasResults && (
            <div className="active-dataset-chip">
              <FileSearch size={13} />
              <span>{data.count.toLocaleString()} txns</span>
            </div>
          )}
          <div className="flex items-center gap-2">
            <div className="topnav__status-dot" />
            <span>API Online</span>
          </div>
        </div>
      </nav>

      <main className="main-content">
        {activeTab === 'overview' && (
          <LandingPage
            onNavigate={(tab) => setActiveTab(tab)}
            onQuickDemo={handleQuickDemo}
            loadingDemo={loadingDemo}
          />
        )}

        {activeTab === 'upload' && (
          <div>
            <PageHeader
              title="Analyze Transactions"
              subtitle="Upload an Ethereum transaction CSV to detect laundering patterns."
            />
            <UploadPanel
              onResults={handleResults}
              loading={loading}
              setLoading={setLoading}
            />
          </div>
        )}

        {activeTab === 'dashboard' && hasResults && (
          <div>
            <PageHeader
              title="Dashboard"
              subtitle={`${data.filename ? `${data.filename} · ` : ''}${data.count.toLocaleString()} transactions analyzed`}
              action={
                <button
                  className="btn btn--secondary btn--sm"
                  onClick={() => {
                    setData(null);
                    setDrillDownFilter(null);
                    setAnalystReviews({});
                    setActiveTab('upload');
                  }}
                >
                  <RefreshCw /> New Dataset
                </button>
              }
            />
            <DashboardOverview
              summary={data.summary}
              results={data.results}
              resultCount={data.count}
              normalization={data.normalization}
              reviews={analystReviews}
              onDrillDown={handleDrillDown}
            />
          </div>
        )}

        {activeTab === 'alerts' && hasResults && (
          <div>
            <PageHeader
              title="Alerts"
              subtitle="Select a row to see its SHAP explanation."
            />
            <AlertsTable
              results={data.results}
              activeDrillDown={drillDownFilter}
              onClearDrillDown={() => setDrillDownFilter(null)}
              reviews={analystReviews}
              onSaveReview={handleSaveReview}
              onClearReview={handleClearReview}
            />
          </div>
        )}

        {activeTab === 'network' && hasResults && (
          <div>
            <PageHeader
              title="Wallet Network"
              subtitle="Select a wallet to inspect its risk profile."
            />
            <NetworkGraph graphData={data.graph} results={data.results} />
          </div>
        )}

        {activeTab === 'model' && (
          <div>
            <PageHeader
              title="Model"
              subtitle="Architecture, features, and evaluation metrics."
            />
            <ModelInfoPanel />
          </div>
        )}
      </main>
    </>
  );
}
