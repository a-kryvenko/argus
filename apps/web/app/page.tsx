'use client';

import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function PublicForecast() {
  const [forecast, setForecast] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/forecast/solar-wind?hours=72');
      const data = await res.json();
      setForecast(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 300000); // Each 5 minutes
    return () => clearInterval(interval);
  }, []);

  if (loading || !forecast) {
    return <div className="min-h-screen bg-zinc-950 flex items-center justify-center text-xl text-zinc-400">Forecast loading...</div>;
  }

  //ts["timestamp"]: new Date(ts["timestamp"]).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric' }),
  const chartData = forecast.forecast;

  const latest = chartData[0];
  const isStormRisk = latest.bz < -5;
  const isModerateRisk = latest.bz < -2 && latest.bz >= -5;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 pb-12">
      <div className="pt-4">
        <div className="max-w-7xl mx-auto px-6 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-10">
            <div>
              <h1 className="text-5xl font-bold tracking-tight">Argus Sunwatch</h1>
              <p className="text-zinc-400 mt-2 text-lg">Solar Wind and Magnetic Field forecast</p>
            </div>
            <div className="text-sm text-zinc-500 mt-4 md:mt-0">
              Updated: {new Date(forecast.forecast_generated_at).toLocaleString('en-US')}
            </div>
          </div>

          {/* Status Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-12">
            <div className="bg-zinc-900 rounded-3xl p-6">
              <p className="text-zinc-400 text-sm">Solar Wind Speed</p>
              <p className="text-5xl font-semibold mt-3">{latest.V} <span className="text-2xl">km/s</span></p>
            </div>

            <div className={`rounded-3xl p-6 ${isStormRisk ? 'bg-red-950 border border-red-500/50' : isModerateRisk ? 'bg-yellow-950 border border-yellow-500/50' : 'bg-zinc-900'}`}>
              <p className="text-zinc-400 text-sm">IMF Bz</p>
              <p className={`text-5xl font-semibold mt-3 ${isStormRisk ? 'text-red-400' : isModerateRisk ? 'text-yellow-400' : 'text-emerald-400'}`}>
                {latest.bz} <span className="text-2xl">nT</span>
              </p>
              {isStormRisk && <p className="text-red-400 text-sm mt-2">High magnetic storm risk</p>}
            </div>

            <div className="bg-zinc-900 rounded-3xl p-6">
              <p className="text-zinc-400 text-sm">Kp-index</p>
              <p className="text-5xl font-semibold mt-3">{latest.KP}</p>
            </div>

            <div className="bg-zinc-900 rounded-3xl p-6">
              <p className="text-zinc-400 text-sm">Forecast for</p>
              <p className="text-5xl font-semibold mt-3">72 hours</p>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-zinc-900 rounded-3xl p-7">
              <h2 className="text-lg mb-5">Solar wind speed (km/s)</h2>
              <ResponsiveContainer width="100%" height={380}>
                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="timestamp" 
                    tick={{ fontSize: 11, fill: '#71717a' }}
                  />
                  <YAxis
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
                  <Line 
                    type="natural" 
                    dataKey="V" 
                    stroke="#f97316" 
                    strokeWidth={3.5} 
                    dot={false}
                    activeDot={{ r: 6, fill: '#f97316', stroke: '#fff', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-zinc-900 rounded-3xl p-7">
              <h2 className="text-lg mb-5">IMF Bz (nT) - southward is dangerous</h2>
              <ResponsiveContainer width="100%" height={380}>
                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="timestamp" 
                    tick={{ fontSize: 11, fill: '#71717a' }}
                  />
                  <YAxis
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
                  <Line 
                    type="natural" 
                    dataKey="BZ" 
                    stroke="#f87171" 
                    strokeWidth={3.5} 
                    dot={false}
                    activeDot={{ r: 6, fill: '#f97316', stroke: '#fff', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="text-center text-xs text-zinc-500 mt-10">
            Model: Enchanced Surya-1.0 - Data updated automaticly
          </div>
        </div>
      </div>
    </div>
  );
}