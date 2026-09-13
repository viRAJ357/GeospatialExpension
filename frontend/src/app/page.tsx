"use client";

import React, { useState, useEffect } from "react";
import {
  Map,
  LayoutDashboard,
  Sprout,
  FlaskConical,
  CloudSun,
  CloudRain,
  TrendingUp,
  AlertTriangle,
  ListChecks,
  FileText,
  Activity,
  Database,
  Settings,
  ChevronDown,
  Clock,
  User,
  Download,
  SlidersHorizontal,
  ThermometerSun,
  Droplets,
  Beaker,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
} from "recharts";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────
interface FormData {
  crop_type: string;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
}

interface Predictions {
  crop_yield_tons_ha: number;
  derived_ndvi: number;
  disease_name: string;
  irrigation_needed: string;
  nitrogen_recommendation_kg_ha: number;
  soil_fertility_index: number;
  water_stress_level: string;
}

// ─────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────
function NavItem({
  icon,
  label,
  active = false,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className={`flex items-center gap-3 px-3 py-2 rounded cursor-pointer transition-colors ${
        active
          ? "bg-[#1c2a1e] text-emerald-400 border border-emerald-900/40"
          : "text-neutral-500 hover:text-neutral-200 hover:bg-[#1a1a1a]"
      }`}
    >
      <div className="w-4 h-4 shrink-0">{icon}</div>
      <span className="text-xs font-medium">{label}</span>
    </div>
  );
}

function MetricRow({
  label,
  value,
  highlight = false,
  statusLevel = null,
}: {
  label: string;
  value: string;
  highlight?: boolean;
  statusLevel?: string | null;
}) {
  let color = "text-neutral-200";
  if (statusLevel) {
    const lvl = statusLevel.toLowerCase();
    color =
      lvl === "high" || lvl === "severe"
        ? "text-red-400"
        : lvl === "moderate" || lvl === "medium"
        ? "text-amber-400"
        : "text-emerald-400";
  }
  return (
    <div className="flex justify-between items-center py-1">
      <span className="text-xs text-neutral-500">{label}</span>
      <span className={`text-sm font-semibold ${highlight ? "text-emerald-400" : color}`}>
        {value}
      </span>
    </div>
  );
}

function RiskRow({
  label,
  level,
  note,
}: {
  label: string;
  level: string;
  note: string;
}) {
  const lvl = level.toLowerCase();
  const badgeClass =
    lvl === "high" || lvl === "critical" || lvl === "elevated"
      ? "text-red-400 bg-red-950/30 border-red-900/40"
      : lvl === "moderate" || lvl === "medium"
      ? "text-amber-400 bg-amber-950/30 border-amber-900/40"
      : lvl === "unknown"
      ? "text-neutral-500 bg-neutral-800/30 border-neutral-700/40"
      : "text-emerald-400 bg-emerald-950/30 border-emerald-900/40";

  return (
    <div className="border-b border-[#1e1e1e] last:border-b-0 px-3 py-3 hover:bg-[#151515] transition-colors">
      <div className="flex justify-between items-start mb-0.5">
        <span className="text-xs font-medium text-neutral-300">{label}</span>
        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${badgeClass}`}>
          {level.toUpperCase()}
        </span>
      </div>
      <p className="text-[11px] text-neutral-500 leading-relaxed">{note}</p>
    </div>
  );
}

// ─────────────────────────────────────────────
// Main Dashboard
// ─────────────────────────────────────────────
export default function AgriVisionEnterprise() {
  const [mounted, setMounted] = useState(false);
  const [activeTab, setActiveTab] = useState("Dashboard");
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Predictions | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState<FormData>({
    crop_type: "Maize",
    nitrogen: 59,
    phosphorus: 60,
    potassium: 60,
    temperature: 40,
    humidity: 89,
    ph: 2.7,
    rainfall: 105,
  });

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === "crop_type" ? value : parseFloat(value),
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("https://geospatialexpension-3.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (data.success) {
        setResults(data.predictions);
        setIsPanelOpen(false);
      } else {
        setError(data.error ?? "Prediction failed.");
      }
    } catch {
      setError("Cannot connect to the prediction API. Is the Flask server running?");
    } finally {
      setLoading(false);
    }
  };

  // Chart data
  const yieldHistory = [
    { name: "W-4", value: results ? +(results.crop_yield_tons_ha * 0.72).toFixed(2) : 2.54 },
    { name: "W-3", value: results ? +(results.crop_yield_tons_ha * 0.82).toFixed(2) : 2.89 },
    { name: "W-2", value: results ? +(results.crop_yield_tons_ha * 0.90).toFixed(2) : 3.18 },
    { name: "W-1", value: results ? +(results.crop_yield_tons_ha * 0.96).toFixed(2) : 3.39 },
    { name: "Forecast", forecast: results ? +results.crop_yield_tons_ha.toFixed(2) : 3.53 },
  ];

  const fieldHealth = results
    ? Math.min(100, Math.round(results.soil_fertility_index * 100 * 0.6 + 40))
    : 78;

  // Risk calculations
  const phLevel = formData.ph;
  const phRisk = phLevel < 5.5 || phLevel > 7.5 ? "High" : "Optimal";
  const phNote =
    phLevel < 5.5
      ? `pH ${phLevel} — Highly acidic. Nutrient uptake severely impacted. Lime application recommended.`
      : phLevel > 7.5
      ? `pH ${phLevel} — Alkaline soil. Micronutrient deficiency likely.`
      : `pH ${phLevel} — Within acceptable range for ${formData.crop_type}.`;

  const waterStress = results?.water_stress_level ?? "Low";
  const waterNote =
    waterStress === "High"
      ? "Critical moisture deficit. Immediate irrigation required."
      : waterStress === "Moderate"
      ? "Below optimal moisture. Schedule irrigation within 48 hours."
      : "Moisture levels within acceptable range. No action needed.";

  const tempRisk = formData.temperature > 37 ? "Moderate" : "Low";
  const tempNote =
    formData.temperature > 37
      ? `${formData.temperature}C detected. Potential crop stress during peak afternoon hours.`
      : `Temperature within optimal range for ${formData.crop_type}.`;

  const diseaseLevel =
    results?.disease_name === "None"
      ? "Low"
      : results?.disease_name
      ? "High"
      : "Unknown";
  const diseaseNote =
    results?.disease_name === "None"
      ? "No significant disease indicators detected in model output."
      : results?.disease_name
      ? `CRITICAL: ${results.disease_name} outbreak predicted. Immediate agronomic intervention required.`
      : "Disease assessment requires field-specific pathogen dataset, which is not available for this field configuration.";

  // Dynamic Actions
  const actions = [];
  
  if (results?.disease_name && results.disease_name !== "None") {
    actions.push({
      title: "Disease Control",
      severity: "CRITICAL",
      color: "red",
      desc: `High risk of ${results.disease_name} detected for ${formData.crop_type}. Immediate fungicide application or agronomic intervention required.`
    });
  }

  if (phRisk === "High") {
    actions.push({
      title: "Soil Treatment Review",
      severity: "HIGH",
      color: "red",
      desc: phNote
    });
  }

  if (waterStress === "High" || results?.irrigation_needed === "Yes") {
    actions.push({
      title: "Irrigation Schedule",
      severity: waterStress === "High" ? "HIGH" : "MEDIUM",
      color: waterStress === "High" ? "red" : "amber",
      desc: waterNote
    });
  }

  actions.push({
    title: "Nitrogen Application",
    severity: "MEDIUM",
    color: "amber",
    desc: results ? `Model recommends ${results.nitrogen_recommendation_kg_ha.toFixed(1)} kg/ha for optimal yield.` : "Run analysis to get nitrogen recommendation."
  });

  if (actions.length < 3) {
    actions.push({
      title: "Routine Monitoring",
      severity: "LOW",
      color: "neutral",
      desc: "All parameters stable. Continue standard monitoring protocol."
    });
  }
  
  const topActions = actions.slice(0, 3);

  const crops = [
    "Rice","Maize","Chickpea","Kidneybeans","Pigeonpeas","Mothbeans",
    "Mungbean","Blackgram","Lentil","Pomegranate","Banana","Mango",
    "Grapes","Watermelon","Muskmelon","Apple","Orange","Papaya",
    "Coconut","Cotton","Jute","Coffee",
  ];

  return (
    <div className="app-shell flex h-screen bg-[#0f0f0f] text-neutral-200 overflow-hidden" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>

      {/* ── SIDEBAR ── */}
      <aside className="sidebar-desktop w-56 shrink-0 bg-[#0a0a0a] border-r border-[#1e1e1e] flex flex-col">
        {/* Logo */}
        <div className="h-12 flex items-center px-4 border-b border-[#1e1e1e]">
          <div className="flex items-center gap-2">
            <Sprout className="w-4 h-4 text-emerald-500" />
            <span className="text-sm font-bold tracking-tight text-neutral-100">AgriVision Pro</span>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto py-4 px-2 space-y-5 custom-scrollbar">
          <div className="space-y-0.5">
            <p className="sidebar-section-title px-3 mb-1.5 text-[10px] font-semibold text-neutral-600 uppercase tracking-widest">Overview</p>
            <NavItem icon={<LayoutDashboard className="w-4 h-4" />} label="Dashboard" active={activeTab === "Dashboard"} onClick={() => setActiveTab("Dashboard")} />
            <NavItem icon={<Map className="w-4 h-4" />} label="Field Intelligence" />
          </div>
          <div className="space-y-0.5">
            <p className="sidebar-section-title px-3 mb-1.5 text-[10px] font-semibold text-neutral-600 uppercase tracking-widest">Analytics</p>
            <NavItem icon={<Sprout className="w-4 h-4" />} label="Crop Health" />
            <NavItem icon={<FlaskConical className="w-4 h-4" />} label="Soil Analytics" />
            <NavItem icon={<CloudSun className="w-4 h-4" />} label="Weather" />
            <NavItem icon={<TrendingUp className="w-4 h-4" />} label="Yield Forecast" />
            <NavItem icon={<AlertTriangle className="w-4 h-4" />} label="Risk Analysis" />
          </div>
          <div className="space-y-0.5">
            <p className="sidebar-section-title px-3 mb-1.5 text-[10px] font-semibold text-neutral-600 uppercase tracking-widest">Operations</p>
            <NavItem icon={<ListChecks className="w-4 h-4" />} label="Recommendations" />
            <NavItem icon={<FileText className="w-4 h-4" />} label="Reports" />
          </div>
          <div className="space-y-0.5">
            <p className="sidebar-section-title px-3 mb-1.5 text-[10px] font-semibold text-neutral-600 uppercase tracking-widest">System</p>
            <NavItem icon={<Activity className="w-4 h-4" />} label="Model Monitor" />
            <NavItem icon={<Database className="w-4 h-4" />} label="Data Quality" />
            <NavItem icon={<Settings className="w-4 h-4" />} label="Settings" />
          </div>
        </nav>

        {/* Status footer */}
        <div className="p-3 border-t border-[#1e1e1e]">
          <div className="bg-[#111] rounded p-2.5 text-[10px] space-y-1.5">
            <div className="flex justify-between">
              <span className="text-neutral-600">Models</span>
              <span className="text-emerald-400 font-medium">6 / 6 Active</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-600">Data Quality</span>
              <span className="text-neutral-300 font-medium">95%</span>
            </div>
          </div>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Top bar */}
        <header className="h-12 bg-[#0a0a0a] border-b border-[#1e1e1e] flex items-center justify-between px-5 shrink-0">
          <button className="topbar-field-selector flex items-center gap-2 bg-[#161616] border border-[#272727] px-3 py-1.5 rounded text-xs font-medium hover:bg-[#1c1c1c] transition-colors">
            <span>Field 07 &mdash; Odisha Farm Cluster</span>
            <ChevronDown className="w-3.5 h-3.5 text-neutral-500" />
          </button>
          <div className="topbar-meta flex items-center gap-5 text-[11px] text-neutral-500">
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              <span className="text-emerald-500 font-medium">Live</span>
            </div>
            <div className="border-l border-[#272727] pl-4">
              <span className="text-neutral-600">Models&nbsp;</span>
              <span className="text-neutral-300 font-medium">6 / 6</span>
            </div>
            <div className="border-l border-[#272727] pl-4 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              <span>13 Sep 2026</span>
            </div>
            <div className="border-l border-[#272727] pl-4">
              <div className="w-6 h-6 rounded-full bg-[#1e1e1e] border border-[#2e2e2e] flex items-center justify-center">
                <User className="w-3.5 h-3.5" />
              </div>
            </div>
          </div>
        </header>

        {/* Workspace */}
        <div className="main-workspace flex-1 overflow-y-auto custom-scrollbar">
          <div className="p-5 space-y-5">

            {/* Page header */}
            <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 pb-4 border-b border-[#1e1e1e]">
              <div>
                <h1 className="text-xl font-semibold text-neutral-100 tracking-tight">Field Intelligence</h1>
                <div className="flex flex-wrap items-center gap-1.5 mt-1 text-xs text-neutral-500">
                  <span>Field 07</span>
                  <span>&#183;</span>
                  <span className="text-emerald-500">{formData.crop_type}</span>
                  <span>&#183;</span>
                  <span>42.6 ha</span>
                  <span>&#183;</span>
                  <span>Monitoring: 13 Sep 2026</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => setIsPanelOpen(true)} className="flex items-center gap-1.5 bg-[#161616] hover:bg-[#1e1e1e] border border-[#272727] text-xs font-medium px-3 py-1.5 rounded transition-colors">
                  <SlidersHorizontal className="w-3.5 h-3.5" /> Configure
                </button>
                <button onClick={handlePredict} disabled={loading} className="flex items-center gap-1.5 bg-emerald-800 hover:bg-emerald-700 text-white text-xs font-semibold px-4 py-1.5 rounded transition-colors disabled:opacity-50">
                  {loading ? <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Activity className="w-3.5 h-3.5" />}
                  Run Analysis
                </button>
                <button className="flex items-center gap-1.5 bg-[#161616] hover:bg-[#1e1e1e] border border-[#272727] text-xs font-medium px-3 py-1.5 rounded transition-colors">
                  <Download className="w-3.5 h-3.5" /> Export
                </button>
              </div>
            </div>

            {/* Error banner */}
            {error && (
              <div className="bg-red-950/40 border border-red-900/50 text-red-300 text-xs px-4 py-2.5 rounded flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                {error}
              </div>
            )}

            {/* ── HERO ── */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">

              {/* Left: Map + Forecast + Model Monitor */}
              <div className="xl:col-span-8 space-y-5">

                {/* GIS Map + Field Health */}
                <div className="hero-card border border-[#1e1e1e] rounded-lg overflow-hidden flex flex-col md:flex-row" style={{ minHeight: 360 }}>

                  {/* Map */}
                  <div className="hero-map flex-1 relative bg-[#0c120e] overflow-hidden">
                    {/* Faux satellite zones */}
                    <div className="absolute inset-0">
                      <div className="absolute top-[20%] left-[15%] w-[45%] h-[40%] bg-emerald-700/25 blur-3xl rounded-full" />
                      <div className="absolute top-[45%] right-[10%] w-[30%] h-[30%] bg-amber-700/20 blur-2xl rounded-full" />
                      <div className="absolute bottom-[15%] left-[30%] w-[25%] h-[25%] bg-red-800/20 blur-2xl rounded-full" />
                    </div>
                    {/* Field boundary SVG */}
                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 200 200" preserveAspectRatio="none">
                      <polygon points="20,18 178,22 175,178 25,182" fill="none" stroke="#16a34a" strokeWidth="0.6" strokeDasharray="4,2" opacity="0.5" />
                      <polygon points="45,45 155,48 152,100 48,102" fill="none" stroke="#d97706" strokeWidth="0.4" opacity="0.4" />
                      <text x="30" y="14" fontSize="6" fill="#16a34a" opacity="0.6">Field 07 Boundary</text>
                      <text x="50" y="42" fontSize="5" fill="#d97706" opacity="0.5">Zone A</text>
                      <text x="50" y="118" fontSize="5" fill="#6b7280" opacity="0.5">Zone B</text>
                    </svg>
                    {/* Grid overlay */}
                    <svg className="absolute inset-0 w-full h-full opacity-5" viewBox="0 0 200 200">
                      {Array.from({ length: 10 }).map((_, i) => (
                        <React.Fragment key={i}>
                          <line x1={i * 20} y1="0" x2={i * 20} y2="200" stroke="#fff" strokeWidth="0.5" />
                          <line x1="0" y1={i * 20} x2="200" y2={i * 20} stroke="#fff" strokeWidth="0.5" />
                        </React.Fragment>
                      ))}
                    </svg>
                    {/* Controls */}
                    <div className="absolute top-3 right-3 flex flex-col gap-1.5">
                      <div className="bg-[#0f0f0f]/90 border border-[#2e2e2e] rounded text-neutral-400 text-xs">
                        <div className="px-2 py-1 hover:bg-[#1e1e1e] cursor-pointer text-center border-b border-[#2e2e2e]">+</div>
                        <div className="px-2 py-1 hover:bg-[#1e1e1e] cursor-pointer text-center">&#8722;</div>
                      </div>
                      <div className="bg-[#0f0f0f]/90 border border-[#2e2e2e] rounded p-1.5 hover:bg-[#1e1e1e] cursor-pointer">
                        <Map className="w-3.5 h-3.5 text-neutral-400" />
                      </div>
                    </div>
                    {/* Layer label */}
                    <div className="absolute top-3 left-3 bg-[#0f0f0f]/80 border border-[#2e2e2e] rounded px-2 py-1 text-[10px] text-neutral-400">
                      NDVI Composite
                    </div>
                    {/* Legend */}
                    <div className="absolute bottom-3 left-3 bg-[#0a0a0a]/90 border border-[#2e2e2e] rounded px-3 py-2">
                      <p className="text-[9px] text-neutral-600 font-semibold uppercase mb-1 tracking-wider">NDVI</p>
                      <div className="w-28 h-1.5 rounded-full bg-gradient-to-r from-red-700 via-amber-600 to-emerald-600 mb-1" />
                      <div className="flex justify-between text-[9px] text-neutral-600">
                        <span>0.0</span><span>1.0</span>
                      </div>
                    </div>
                  </div>

                  {/* Field Health KPIs */}
                  <div className="hero-metrics w-full md:w-52 bg-[#0d0d0d] border-t md:border-t-0 md:border-l border-[#1e1e1e] p-4 flex flex-col gap-4">
                    <div>
                      <p className="text-[10px] text-neutral-600 uppercase font-semibold tracking-wider mb-2">Field Health</p>
                      <div className="flex items-baseline gap-1.5 mb-2">
                        <span className="text-3xl font-light text-emerald-400">{fieldHealth}</span>
                        <span className="text-xs text-neutral-600">/ 100</span>
                      </div>
                      <div className="w-full bg-[#1e1e1e] h-1 rounded-full">
                        <div className="bg-emerald-600 h-1 rounded-full transition-all" style={{ width: `${fieldHealth}%` }} />
                      </div>
                    </div>
                    <div className="border-t border-[#1e1e1e] pt-3 space-y-2">
                      <MetricRow label="NDVI" value={results ? results.derived_ndvi.toFixed(3) : "—"} />
                      <MetricRow label="Yield" value={results ? `${results.crop_yield_tons_ha.toFixed(2)} t/ha` : "—"} highlight />
                      <MetricRow label="Soil Index" value={results ? `${(results.soil_fertility_index * 100).toFixed(0)} / 100` : "—"} />
                      <MetricRow label="Water Stress" value={results?.water_stress_level ?? "—"} statusLevel={results?.water_stress_level} />
                      <MetricRow label="Irrigation" value={results?.irrigation_needed ?? "—"} statusLevel={results?.irrigation_needed === "Yes" ? "moderate" : "low"} />
                      <MetricRow label="N Rec." value={results ? `${results.nitrogen_recommendation_kg_ha.toFixed(1)} kg/ha` : "—"} />
                    </div>
                    {!results && (
                      <p className="text-[10px] text-neutral-600 text-center border border-[#1e1e1e] rounded p-2">
                        Run Analysis to populate metrics
                      </p>
                    )}
                  </div>
                </div>

                {/* Yield Forecast + Model Monitor */}
                <div className="analytics-grid grid grid-cols-1 md:grid-cols-2 gap-5">

                  {/* Yield Forecast */}
                  <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-4">
                    <p className="text-[10px] text-neutral-600 uppercase font-semibold tracking-wider mb-3 flex items-center gap-1.5">
                      <TrendingUp className="w-3.5 h-3.5" /> Yield Forecast
                    </p>
                    <div className="chart-container" style={{ height: 180 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={yieldHistory} margin={{ top: 8, right: 4, left: -22, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="2 2" stroke="#1e1e1e" vertical={false} />
                          <XAxis dataKey="name" stroke="#333" tick={{ fill: "#555", fontSize: 10 }} axisLine={false} tickLine={false} />
                          <YAxis stroke="#333" tick={{ fill: "#555", fontSize: 10 }} axisLine={false} tickLine={false} domain={[2, "auto"]} />
                          <Tooltip
                            contentStyle={{ backgroundColor: "#111", border: "1px solid #222", fontSize: 11, borderRadius: 6 }}
                            labelStyle={{ color: "#aaa" }}
                            itemStyle={{ color: "#10b981" }}
                          />
                          <Line type="monotone" dataKey="value" name="Historical" stroke="#4b5563" strokeWidth={1.5} dot={{ r: 3, fill: "#4b5563", strokeWidth: 0 }} connectNulls />
                          <Line type="monotone" dataKey="forecast" name="Forecast" stroke="#10b981" strokeWidth={2} dot={{ r: 4, fill: "#10b981", strokeWidth: 0 }} strokeDasharray="4 2" connectNulls />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="flex gap-3 mt-2 text-[10px] text-neutral-500">
                      <span className="flex items-center gap-1"><span className="w-5 border-t border-[#4b5563]" /> Historical</span>
                      <span className="flex items-center gap-1"><span className="w-5 border-t border-dashed border-emerald-600" /> Forecast</span>
                    </div>
                  </div>

                  {/* Model Monitor */}
                  <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-4">
                    <div className="flex justify-between items-center mb-3">
                      <p className="text-[10px] text-neutral-600 uppercase font-semibold tracking-wider flex items-center gap-1.5">
                        <Database className="w-3.5 h-3.5" /> Model Monitor
                      </p>
                      <span className="text-[10px] text-emerald-400 bg-emerald-950/50 border border-emerald-900/40 px-1.5 py-0.5 rounded font-medium">
                        Healthy
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      {[
                        { label: "Models Active", value: "6 / 6 (Valid)" },
                        { label: "Last Inference", value: loading ? "Running..." : results ? "&#10003; Done" : "Not run" },
                        { label: "Yield Confidence", value: "R² 0.89" },
                        { label: "Disease Accuracy", value: "88.2%" },
                      ].map((m) => (
                        <div key={m.label} className="bg-[#111] border border-[#1e1e1e] rounded p-2.5">
                          <p className="text-[9px] text-neutral-600 mb-0.5">{m.label}</p>
                          <p className="text-xs font-medium text-neutral-200" dangerouslySetInnerHTML={{ __html: m.value }} />
                        </div>
                      ))}
                    </div>
                    <div>
                      <p className="text-[9px] text-neutral-600 uppercase tracking-wider mb-1.5">Input Telemetry</p>
                      <div className="flex flex-wrap gap-1">
                        {["Weather", "Soil NPK", "Crop Type", "pH", "Rainfall"].map((tag) => (
                          <span key={tag} className="text-[10px] text-neutral-500 bg-[#141414] border border-[#222] px-2 py-0.5 rounded">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

              </div>

              {/* Right column: Risk + Actions + Weather + Soil */}
              <div className="xl:col-span-4 space-y-5">

                {/* Risk Assessment */}
                <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg overflow-hidden">
                  <div className="px-4 py-3 border-b border-[#1e1e1e]">
                    <p className="text-[10px] text-neutral-600 uppercase font-semibold tracking-wider flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" /> Field Risk Assessment
                    </p>
                  </div>
                  <div>
                    <RiskRow label="Disease Risk" level={diseaseLevel} note={diseaseNote} />
                    <RiskRow label="Water Stress" level={waterStress} note={waterNote} />
                    <RiskRow label="Heat Stress" level={tempRisk} note={tempNote} />
                    <RiskRow label="Soil Acidity" level={phRisk} note={phNote} />
                  </div>
                </div>

                {/* Priority Actions */}
                <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg overflow-hidden">
                  <div className="px-4 py-3 border-b border-[#1e1e1e]">
                    <p className="text-[10px] text-neutral-600 uppercase font-semibold tracking-wider flex items-center gap-1.5">
                      <ListChecks className="w-3.5 h-3.5" /> Priority Actions
                    </p>
                  </div>
                  <div className="divide-y divide-[#1e1e1e]">
                    {topActions.map((action, idx) => {
                      const colorClass = 
                        action.color === 'red' ? 'text-red-400 border-red-800/60 bg-red-950/30' :
                        action.color === 'amber' ? 'text-amber-400 border-amber-800/60 bg-amber-950/30' :
                        'text-neutral-600 border-[#2e2e2e] bg-[#161616]';
                      
                      const badgeClass =
                        action.color === 'red' ? 'text-red-400 border-red-900/40 bg-red-950/30' :
                        action.color === 'amber' ? 'text-amber-400 border-amber-900/40 bg-amber-950/30' :
                        'text-neutral-500 border-[#2e2e2e] bg-[#141414]';

                      return (
                        <div key={idx} className="px-4 py-3 flex gap-3 hover:bg-[#111] transition-colors">
                          <div className={`shrink-0 w-5 h-5 rounded border flex items-center justify-center text-[9px] font-bold mt-0.5 ${colorClass}`}>
                            0{idx + 1}
                          </div>
                          <div>
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className={`text-xs font-medium ${action.color === 'neutral' ? 'text-neutral-500' : 'text-neutral-200'}`}>
                                {action.title}
                              </span>
                              <span className={`text-[9px] px-1 rounded border font-bold tracking-wider ${badgeClass}`}>
                                {action.severity}
                              </span>
                            </div>
                            <p className="text-[11px] text-neutral-500 leading-relaxed">{action.desc}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Weather & Soil */}
                <div className="bg-[#0d0d0d] border border-[#1e1e1e] rounded-lg p-4">
                  <p className="text-[10px] text-neutral-600 uppercase font-semibold tracking-wider mb-3 flex items-center gap-1.5">
                    <CloudSun className="w-3.5 h-3.5" /> Environmental Parameters
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-[#111] border border-[#1e1e1e] rounded p-2.5">
                      <p className="text-[9px] text-neutral-600 flex items-center gap-1 mb-1">
                        <ThermometerSun className="w-3 h-3 text-orange-700" /> Temperature
                      </p>
                      <p className="text-sm font-semibold text-neutral-200">{formData.temperature}&#176;C</p>
                    </div>
                    <div className="bg-[#111] border border-[#1e1e1e] rounded p-2.5">
                      <p className="text-[9px] text-neutral-600 flex items-center gap-1 mb-1">
                        <Droplets className="w-3 h-3 text-cyan-700" /> Humidity
                      </p>
                      <p className="text-sm font-semibold text-neutral-200">{formData.humidity}%</p>
                    </div>
                    <div className="bg-[#111] border border-[#1e1e1e] rounded p-2.5">
                      <p className="text-[9px] text-neutral-600 flex items-center gap-1 mb-1">
                        <CloudRain className="w-3 h-3 text-blue-700" /> Rainfall
                      </p>
                      <p className="text-sm font-semibold text-neutral-200">{formData.rainfall} mm</p>
                    </div>
                    <div className="bg-[#111] border border-[#1e1e1e] rounded p-2.5">
                      <p className="text-[9px] text-neutral-600 flex items-center gap-1 mb-1">
                        <Beaker className="w-3 h-3 text-emerald-700" /> Soil pH
                      </p>
                      <p className={`text-sm font-semibold ${phLevel < 5.5 || phLevel > 7.5 ? "text-red-400" : "text-neutral-200"}`}>{formData.ph}</p>
                    </div>
                    <div className="bg-[#111] border border-[#1e1e1e] rounded p-2.5 col-span-2">
                      <p className="text-[9px] text-neutral-600 flex items-center gap-1 mb-1">
                        <FlaskConical className="w-3 h-3 text-purple-700" /> NPK (kg/ha)
                      </p>
                      <p className="text-sm font-semibold text-neutral-200">N:{formData.nitrogen} &nbsp; P:{formData.phosphorus} &nbsp; K:{formData.potassium}</p>
                    </div>
                  </div>
                </div>

              </div>
            </div>

          </div>
        </div>
      </div>

      {/* ── MOBILE BOTTOM NAV ── */}
      <nav className="mobile-nav hidden fixed bottom-0 left-0 right-0 h-14 bg-[#0a0a0a] border-t border-[#1e1e1e] z-30 items-center justify-around px-2">
        <MobileNavBtn icon={<LayoutDashboard className="w-5 h-5" />} label="Dashboard" active />
        <MobileNavBtn icon={<Map className="w-5 h-5" />} label="Field" />
        <MobileNavBtn icon={<AlertTriangle className="w-5 h-5" />} label="Risk" />
        <MobileNavBtn icon={<Activity className="w-5 h-5" />} label="Models" />
        <MobileNavBtn icon={<Settings className="w-5 h-5" />} label="Settings" />
      </nav>


      {isPanelOpen && (
        <>
          <div className="absolute inset-0 bg-black/60 z-40" onClick={() => setIsPanelOpen(false)} />
          <div className="config-panel absolute inset-y-0 right-0 w-72 bg-[#0a0a0a] border-l border-[#1e1e1e] z-50 flex flex-col">
            <div className="h-12 flex items-center justify-between px-4 border-b border-[#1e1e1e] shrink-0">
              <span className="text-sm font-semibold text-neutral-200">Simulation Parameters</span>
              <button onClick={() => setIsPanelOpen(false)} className="text-xs text-neutral-500 hover:text-white transition-colors">Close</button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-5 text-xs">

              {/* Field */}
              <div className="space-y-2">
                <p className="text-[10px] font-semibold text-neutral-600 uppercase tracking-wider">Field Context</p>
                <div>
                  <label className="text-neutral-500 block mb-1">Crop Type</label>
                  <select name="crop_type" value={formData.crop_type} onChange={handleChange}
                    className="w-full bg-[#111] border border-[#272727] rounded px-3 py-2 text-neutral-200 focus:border-emerald-700 outline-none">
                    {crops.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              </div>

              {/* Soil */}
              <div className="space-y-2">
                <p className="text-[10px] font-semibold text-neutral-600 uppercase tracking-wider">Soil Composition</p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { label: "N (kg/ha)", name: "nitrogen" },
                    { label: "P (kg/ha)", name: "phosphorus" },
                    { label: "K (kg/ha)", name: "potassium" },
                    { label: "pH", name: "ph", step: "0.1" },
                  ].map((f) => (
                    <div key={f.name}>
                      <label className="text-neutral-500 block mb-1">{f.label}</label>
                      <input type="number" name={f.name} step={f.step ?? "1"}
                        value={formData[f.name as keyof FormData]} onChange={handleChange}
                        className="w-full bg-[#111] border border-[#272727] rounded px-2 py-1.5 text-neutral-200 focus:border-emerald-700 outline-none" />
                    </div>
                  ))}
                </div>
              </div>

              {/* Meteorological */}
              <div className="space-y-3">
                <p className="text-[10px] font-semibold text-neutral-600 uppercase tracking-wider">Meteorological</p>
                {[
                  { label: "Temperature", name: "temperature", min: 0, max: 50, unit: "°C" },
                  { label: "Humidity", name: "humidity", min: 0, max: 100, unit: "%" },
                  { label: "Rainfall", name: "rainfall", min: 0, max: 300, unit: "mm" },
                ].map((f) => (
                  <div key={f.name}>
                    <div className="flex justify-between mb-1">
                      <span className="text-neutral-500">{f.label}</span>
                      <span className="text-neutral-300 font-medium">{formData[f.name as keyof FormData]}{f.unit}</span>
                    </div>
                    <input type="range" name={f.name} min={f.min} max={f.max}
                      value={formData[f.name as keyof FormData]} onChange={handleChange}
                      className="w-full h-1 accent-emerald-700 cursor-pointer" />
                    <div className="flex justify-between text-[9px] text-neutral-600 mt-0.5">
                      <span>{f.min}{f.unit}</span><span>{f.max}{f.unit}</span>
                    </div>
                  </div>
                ))}
              </div>

            </div>
            <div className="p-4 border-t border-[#1e1e1e] shrink-0 space-y-2">
              <button onClick={handlePredict} disabled={loading}
                className="w-full bg-emerald-800 hover:bg-emerald-700 text-white font-semibold py-2 rounded text-sm transition-colors flex justify-center items-center gap-2 disabled:opacity-50">
                {loading ? <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : null}
                {loading ? "Analyzing..." : "Apply & Run Analysis"}
              </button>
              {error && <p className="text-red-400 text-[10px] text-center">{error}</p>}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function MobileNavBtn({ icon, label, active = false }: { icon: React.ReactNode; label: string; active?: boolean }) {
  return (
    <div className={`flex flex-col items-center gap-0.5 px-3 py-1.5 rounded cursor-pointer ${active ? 'text-emerald-400' : 'text-neutral-500 hover:text-neutral-200'}`}>
      <div className="w-5 h-5">{icon}</div>
      <span className="text-[9px] font-medium">{label}</span>
    </div>
  );
}
