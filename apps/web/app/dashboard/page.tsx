'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const regions = [
  { code: 'EU', name: 'Europe' },
  { code: 'US', name: 'USA' },
  { code: 'CA', name: 'Canada' },
  { code: 'AU', name: 'Australia' },
  { code: 'NC', name: 'Arctic' },
  { code: 'SC', name: 'Antarctic' },
];

export default function PrivateDashboard() {
  const [token, setToken] = useState('');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedRegion, setSelectedRegion] = useState('EU');

  useEffect(() => {
    const saved = localStorage.getItem('access_token');
    if (saved) {
      setToken(saved);
      loadImpact(saved, selectedRegion);
    }
  }, []);

  const loadImpact = async (accessToken: string, region: string) => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`http://localhost:8001/api/private/impact/powergrids?region=${region}`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });

      if (res.status === 401) {
        setError('Incorrect token');
        localStorage.removeItem('access_token');
        return;
      }

      const result = await res.json();
      setData(result);
      localStorage.setItem('access_token', accessToken);
    } catch (e) {
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  const handleRegionChange = (region: string) => {
    setSelectedRegion(region);
    if (token) loadImpact(token, region);
  };

  // For risk chart
  const chartData = data?.impact_data?.map((item: any, index: number) => ({
    hour: index,
    time: new Date(item.timestamp).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric' }),
    gic_risk: item.gic_risk,
    speed: item.solar_wind_speed,
    bz: item.IMF_Bz,
  })) || [];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between mb-10">
          <h1 className="text-4xl font-bold">Impact Intelligence - Hourly Forecast</h1>
          {data && <button onClick={() => { localStorage.removeItem('access_token'); setData(null); }} className="text-red-400">Logout</button>}
        </div>

        {!data ? (
          <div className="max-w-lg mx-auto bg-zinc-900 p-10 rounded-3xl">
            <h2 className="text-2xl mb-8">Enter access token</h2>
            <form onSubmit={(e) => { e.preventDefault(); if (token) loadImpact(token, selectedRegion); }}>
              <textarea value={token} onChange={e => setToken(e.target.value)} className="w-full h-40 bg-zinc-950 border border-zinc-700 rounded-2xl p-5" placeholder="Enter token..." />
              <button type="submit" className="mt-6 w-full py-4 bg-white text-black rounded-2xl font-semibold">Open dashboard</button>
            </form>
            {error && <p className="text-red-500 mt-4">{error}</p>}
          </div>
        ) : (
          <>
            <div className="flex gap-3 mb-8">
              {regions.map(r => (
                <button key={r.code} onClick={() => handleRegionChange(r.code)}
                  className={`px-6 py-3 rounded-2xl ${selectedRegion === r.code ? 'bg-white text-black' : 'bg-zinc-900 hover:bg-zinc-800'}`}>
                  {r.name}
                </button>
              ))}
            </div>

            <div className="bg-zinc-900 rounded-3xl p-8 mb-8">
              <h2 className="text-2xl mb-6">Hourly GIC Risk (96 hours)</h2>
              <ResponsiveContainer width="100%" height={380}>
                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="time" 
                    tick={{ fontSize: 11, fill: '#71717a' }}
                  />
                  <YAxis 
                    domain={[0, 1]} 
                    tick={{ fontSize: 11, fill: '#71717a' }}
                  />
                  <Tooltip 
                    cursor={true}
                    animationDuration={0}
                    contentStyle={{
                      backgroundColor: '#18181b',
                      border: '1px solid #3f3f46',
                      borderRadius: '8px',
                      padding: '10px 14px',
                      color: '#e4e4e7',
                      fontSize: '13px',
                    }}
                  />
                  <defs>
                    <linearGradient id="colorRisk" x1="0" y1="1" x2="0" y2="0">
                      <stop offset="0%" stopColor="#22c55e" />
                      <stop offset="50%" stopColor="#22c55e" />
                      <stop offset="75%" stopColor="#f59e0b" />
                      <stop offset="100%" stopColor="#ef4444" />
                    </linearGradient>
                  </defs>

                  <Line
                    dataKey="gic_risk"
                    type="natural"
                    strokeWidth={3.5} 
                    stroke="url(#colorRisk)"
                    dot={false}
                    activeDot={{ r: 6, fill: '#f97316', stroke: '#fff', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-zinc-900 rounded-3xl overflow-hidden">
              <div className="p-6 border-b border-zinc-800 flex justify-between">
                <h3 className="font-semibold">Hourly detalization</h3>
                <span className="text-zinc-400">Total of {data.forecast_period_hours} hours</span>
              </div>
              <div className="max-h-[600px] overflow-auto">
                <table className="w-full">
                  <thead className="bg-zinc-950 sticky top-0">
                    <tr>
                      <th className="p-4 text-left">Time</th>
                      <th className="p-4 text-left">SW Speed</th>
                      <th className="p-4 text-left">Bz (nT)</th>
                      <th className="p-4 text-left">GIC Risk</th>
                      <th className="p-4 text-left">Expected problems</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.impact_data.map((item: any, i: number) => (
                      <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/50">
                        <td className="p-4">{new Date(item.timestamp).toLocaleString('en-US', { hour: '2-digit', minute: '2-digit', day: 'numeric', month: 'short' })}</td>
                        <td className="p-4">{item.solar_wind_speed} km/s</td>
                        <td className="p-4">{item.IMF_Bz}</td>
                        <td className="p-4 font-medium" style={{ color: item.gic_risk > 0.7 ? '#f87171' : item.gic_risk > 0.4 ? '#fbbf24' : '#4ade80' }}>
                          {item.gic_risk}
                        </td>
                        <td className="p-4">{item.estimated_outages}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}