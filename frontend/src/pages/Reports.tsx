import React, { useState, useMemo } from 'react';
import {
  History,
  Search,
  Trash2,
  Eye,
  X,
  Printer,
  FileSpreadsheet,
} from 'lucide-react';
import { getScanRecords, clearScanRecords } from '../services/storage';
import { VerdictBadge } from '../components/common/VerdictBadge';
import type { ScanRecord, Verdict } from '../types/api';

export const Reports: React.FC = () => {
  const [records, setRecords] = useState<ScanRecord[]>(getScanRecords());
  const [searchQuery, setSearchQuery] = useState('');
  const [verdictFilter, setVerdictFilter] = useState<'ALL' | Verdict>('ALL');
  const [categoryFilter, setCategoryFilter] = useState('ALL');
  const [selectedScan, setSelectedScan] = useState<ScanRecord | null>(null);

  // Available unique attack categories
  const categories = useMemo(() => {
    const set = new Set<string>();
    records.forEach((r) => {
      if (r.attackCategory) set.add(r.attackCategory);
    });
    return Array.from(set);
  }, [records]);

  // Filtered scan records
  const filteredRecords = useMemo(() => {
    return records.filter((record) => {
      // Verdict filter
      if (verdictFilter !== 'ALL' && record.verdict !== verdictFilter) return false;
      // Category filter
      if (categoryFilter !== 'ALL' && record.attackCategory !== categoryFilter) return false;
      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesPrompt = record.inputPrompt?.toLowerCase().includes(q);
        const matchesImage = record.imageName?.toLowerCase().includes(q);
        const matchesCategory = record.attackCategory.toLowerCase().includes(q);
        const matchesReason = record.reason?.toLowerCase().includes(q);
        const matchesOcr = record.ocrText?.toLowerCase().includes(q);
        if (!matchesPrompt && !matchesImage && !matchesCategory && !matchesReason && !matchesOcr) {
          return false;
        }
      }
      return true;
    });
  }, [records, verdictFilter, categoryFilter, searchQuery]);

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear scan logs and restore defaults?')) {
      clearScanRecords();
      setRecords(getScanRecords());
    }
  };

  const handleExportCSV = () => {
    const headers = [
      'ID',
      'Timestamp',
      'Modality',
      'Verdict',
      'Confidence',
      'RiskScore',
      'Category',
      'LatencyMs',
      'Prompt',
      'ImageName',
      'Reason',
    ];
    const rows = filteredRecords.map((r) => [
      r.id,
      r.timestamp,
      r.type,
      r.verdict,
      r.confidence,
      r.riskScore,
      `"${r.attackCategory.replace(/"/g, '""')}"`,
      r.latencyMs,
      `"${(r.inputPrompt || '').replace(/"/g, '""')}"`,
      `"${(r.imageName || '').replace(/"/g, '""')}"`,
      `"${(r.reason || '').replace(/"/g, '""')}"`,
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute(
      'download',
      `cmjd_scan_report_${new Date().toISOString().slice(0, 10)}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportPDF = () => {
    window.print();
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-300 print:bg-white print:text-black">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800 print:hidden">
        <div>
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400">
              <History className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                Scan Audit Reports & Log History
              </h1>
              <p className="text-xs text-slate-400">
                Searchable, Filterable Security Telemetry with PDF and CSV Export Capabilities
              </p>
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleExportCSV}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 text-xs font-semibold transition-all"
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
            <span>Export CSV</span>
          </button>
          <button
            type="button"
            onClick={handleExportPDF}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-800/60 text-xs font-semibold transition-all"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>Print / PDF</span>
          </button>
          <button
            type="button"
            onClick={handleClearHistory}
            className="p-2 rounded-xl bg-slate-900 hover:bg-red-950 text-slate-400 hover:text-red-400 border border-slate-800 transition-all"
            title="Reset to Benchmark Records"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4 print:hidden">
        <div className="grid grid-cols-1 sm:grid-cols-12 gap-3">
          {/* Search Box */}
          <div className="sm:col-span-6 relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
            <input
              type="text"
              placeholder="Search prompts, image filenames, or fusion rationale..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>

          {/* Verdict Filter */}
          <div className="sm:col-span-3">
            <select
              value={verdictFilter}
              onChange={(e) => setVerdictFilter(e.target.value as 'ALL' | Verdict)}
              className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-sans"
            >
              <option value="ALL">All Verdicts (All Status)</option>
              <option value="SAFE">SAFE Only</option>
              <option value="JAILBREAK">JAILBREAK (Threats) Only</option>
            </select>
          </div>

          {/* Category Filter */}
          <div className="sm:col-span-3">
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-sans"
            >
              <option value="ALL">All Categories</option>
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Filter Summary */}
        <div className="flex items-center justify-between text-xs text-slate-400 pt-1 border-t border-slate-800/80">
          <span>
            Showing <strong>{filteredRecords.length}</strong> of <strong>{records.length}</strong>{' '}
            records
          </span>
          {(searchQuery || verdictFilter !== 'ALL' || categoryFilter !== 'ALL') && (
            <button
              type="button"
              onClick={() => {
                setSearchQuery('');
                setVerdictFilter('ALL');
                setCategoryFilter('ALL');
              }}
              className="text-cyan-400 hover:text-cyan-300 font-medium"
            >
              Reset Filters
            </button>
          )}
        </div>
      </div>

      {/* Reports Table */}
      <div className="rounded-2xl overflow-hidden bg-slate-900/80 border border-slate-800 shadow-xl print:border-none print:shadow-none">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/70 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="py-3.5 px-4">Timestamp</th>
                <th className="py-3.5 px-4">Modality</th>
                <th className="py-3.5 px-4">Input Target</th>
                <th className="py-3.5 px-4">Category</th>
                <th className="py-3.5 px-4">Risk</th>
                <th className="py-3.5 px-4">Verdict</th>
                <th className="py-3.5 px-4 text-right">Latency</th>
                <th className="py-3.5 px-4 text-center print:hidden">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredRecords.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-500">
                    No scan logs matched your search or filter criteria.
                  </td>
                </tr>
              ) : (
                filteredRecords.map((scan) => (
                  <tr
                    key={scan.id}
                    className="hover:bg-slate-800/40 transition-colors group cursor-pointer"
                    onClick={() => setSelectedScan(scan)}
                  >
                    <td className="py-3 px-4 font-mono text-slate-400 whitespace-nowrap">
                      {new Date(scan.timestamp).toLocaleString([], {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-300 border border-slate-700">
                        {scan.type}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-200 max-w-xs truncate font-mono">
                      {scan.inputPrompt || scan.imageName || scan.ocrText || 'Benchmark Sample'}
                    </td>
                    <td className="py-3 px-4 text-slate-400">{scan.attackCategory}</td>
                    <td className="py-3 px-4">
                      <span
                        className={`font-mono font-bold ${
                          (scan.riskScore ?? 0) > 50 ? 'text-red-400' : 'text-emerald-400'
                        }`}
                      >
                        {(scan.riskScore ?? 0).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <VerdictBadge verdict={scan.verdict} size="sm" />
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-slate-400">
                      {(scan.latencyMs ?? 0).toFixed(1)}ms
                    </td>
                    <td className="py-3 px-4 text-center print:hidden">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedScan(scan);
                        }}
                        className="p-1.5 rounded-lg bg-slate-800 text-cyan-400 hover:bg-slate-700 transition-colors"
                        title="View Full Scan Telemetry"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detailed Scan Modal */}
      {selectedScan && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
          <div className="w-full max-w-2xl rounded-2xl bg-slate-900 border border-cyan-500/40 shadow-2xl p-6 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-3">
                <VerdictBadge verdict={selectedScan.verdict} size="md" />
                <div>
                  <h3 className="text-sm font-bold text-white font-mono">{selectedScan.id}</h3>
                  <p className="text-[11px] text-slate-400">
                    {new Date(selectedScan.timestamp).toLocaleString()} | {selectedScan.type} Scan
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSelectedScan(null)}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-3 gap-3">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 uppercase text-[10px]">Risk Score</span>
                  <p className="text-lg font-mono font-bold text-cyan-400 mt-1">
                    {(selectedScan.riskScore ?? 0).toFixed(1)}%
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 uppercase text-[10px]">Confidence</span>
                  <p className="text-lg font-mono font-bold text-white mt-1">
                    {((selectedScan.confidence ?? 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 uppercase text-[10px]">Latency</span>
                  <p className="text-lg font-mono font-bold text-purple-400 mt-1">
                    {(selectedScan.latencyMs ?? 0).toFixed(1)} ms
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-slate-400 uppercase text-[10px] font-bold">
                  Attack Category
                </span>
                <p className="text-slate-200 font-semibold">{selectedScan.attackCategory}</p>
              </div>

              {selectedScan.inputPrompt && (
                <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <span className="text-slate-400 uppercase text-[10px] font-bold">
                    Evaluated Text Prompt
                  </span>
                  <p className="text-slate-200 font-mono text-[11px] whitespace-pre-wrap">
                    {selectedScan.inputPrompt}
                  </p>
                </div>
              )}

              {selectedScan.ocrText && (
                <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <span className="text-slate-400 uppercase text-[10px] font-bold">
                    OCR Extracted Typography
                  </span>
                  <p className="text-slate-200 font-mono text-[11px] whitespace-pre-wrap">
                    {selectedScan.ocrText}
                  </p>
                </div>
              )}

              {selectedScan.reason && (
                <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <span className="text-slate-400 uppercase text-[10px] font-bold">
                    Decision Logic / Fusion Reasoning
                  </span>
                  <p className="text-cyan-300 font-mono text-[11px] leading-relaxed">
                    {selectedScan.reason}
                  </p>
                </div>
              )}
            </div>

            <div className="pt-2 flex justify-end">
              <button
                type="button"
                onClick={() => setSelectedScan(null)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
