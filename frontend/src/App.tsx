import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/layout/Navbar';
import { Sidebar } from './components/layout/Sidebar';
import { Footer } from './components/layout/Footer';
import { ErrorBoundary } from './components/common/ErrorBoundary';

// Pages
import { Dashboard } from './pages/Dashboard';
import { TextScanner } from './pages/TextScanner';
import { ImageScanner } from './pages/ImageScanner';
import { CrossModalScanner } from './pages/CrossModalScanner';
import { Explainability } from './pages/Explainability';
import { Analytics } from './pages/Analytics';
import { Reports } from './pages/Reports';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-cyan-500 selection:text-slate-950">
        <Navbar />

        <div className="flex-1 flex w-full">
          <Sidebar />

          <main className="flex-1 p-4 sm:p-8 max-w-7xl mx-auto w-full overflow-y-auto">
            <ErrorBoundary fallbackTitle="SentinelGuard AI Page Failure">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/text-scanner" element={<TextScanner />} />
                <Route path="/image-scanner" element={<ImageScanner />} />
                <Route path="/cross-modal-scanner" element={<CrossModalScanner />} />
                <Route path="/explainability" element={<Explainability />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/reports" element={<Reports />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </ErrorBoundary>
          </main>
        </div>

        <Footer />
      </div>
    </BrowserRouter>
  );
};

export default App;
