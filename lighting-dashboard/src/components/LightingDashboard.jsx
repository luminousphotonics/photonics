import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const LightingDashboard = () => {
  const [layer, setLayer] = useState(19);
  const [height, setHeight] = useState(11.68);
  const [ppfd, setPPFD] = useState(1500);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [rawLog, setRawLog] = useState('');
  const [loading, setLoading] = useState(false);

  const runScenario = async () => {
    setLoading(true);
    setResult(null);
    setRawLog('');

    try {
      const response = await fetch("http://localhost:8000/api/run_scenario", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layer, height, ppfd })
      });

      const data = await response.json();
      setRawLog(data.logs || '');
      setResult(data);
      setLogs(prev => [
        ...prev,
        {
          ...data,
          timestamp: new Date().toISOString(),
          layer,
          height,
          ppfd,
          status: (data.action === 'Added' || data.action === 'Updated') ? 'success' : (data.action === 'Skipped' ? 'neutral' : 'fail')
        }
      ]);
    } catch (err) {
      console.error('Run failed:', err);
      alert('Failed to run scenario.');
    }

    setLoading(false);
  };

  const StatusBadge = ({ status }) => {
    const style = {
      success: 'bg-green-200 text-green-800',
      neutral: 'bg-gray-200 text-gray-800',
      fail: 'bg-red-200 text-red-800'
    }[status] || 'bg-gray-100 text-gray-600';
    return <span className={`px-2 py-1 rounded text-xs font-medium ${style}`}>{status}</span>;
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8">
      <h1 className="text-3xl font-bold">Lighting Optimization Dashboard</h1>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="block mb-1 font-medium">Layer Count</label>
          <input type="number" value={layer} onChange={e => setLayer(+e.target.value)} className="border rounded p-2 w-full" />
        </div>
        <div>
          <label className="block mb-1 font-medium">Height (ft)</label>
          <input type="number" value={height} onChange={e => setHeight(+e.target.value)} className="border rounded p-2 w-full" />
        </div>
        <div>
          <label className="block mb-1 font-medium">Target PPFD</label>
          <input type="number" value={ppfd} onChange={e => setPPFD(+e.target.value)} className="border rounded p-2 w-full" />
        </div>
        <div className="col-span-3">
          <button onClick={runScenario} disabled={loading} className="bg-blue-600 text-white px-4 py-2 rounded">
            {loading ? 'Running...' : 'Run Scenario'}
          </button>
        </div>
        <div className="col-span-3">
        <label className="block mb-1 font-medium">Optimizer Log</label>
        <textarea
            value={rawLog}
            readOnly
            rows={16}
            className="w-full p-2 font-mono text-sm bg-black text-green-300 rounded border border-gray-700"
        />
        </div>

      </div>

      {rawLog && rawLog.trim() !== '' && (
        <div className="bg-black text-green-300 p-4 rounded max-h-96 overflow-auto text-sm whitespace-pre-wrap font-mono">
            <h2 className="text-white text-lg mb-2">Optimizer Log</h2>
            <pre>{rawLog}</pre>
        </div>
      )}


      {result && (
        <div className="bg-white shadow rounded p-4">
          <h2 className="text-xl font-semibold mb-2">Latest Result</h2>
          <p><strong>Action:</strong> {result.action || 'N/A'}</p>
          <p><strong>Reason:</strong> {result.reason || 'N/A'}</p>
          <p><strong>Final PPFD:</strong> {result.final_ppfd || 'N/A'}</p>
          <p><strong>mDOU:</strong> {result.mDOU || 'N/A'}</p>
          <p><strong>Loss:</strong> {result.loss || 'N/A'}</p>
        </div>
      )}

      {logs.length > 0 && (
        <div className="bg-white shadow rounded p-4">
          <h2 className="text-xl font-semibold mb-4">Live Chart</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={logs}>
              <XAxis dataKey="timestamp" hide />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="mDOU" stroke="green" name="mDOU" />
              <Line type="monotone" dataKey="loss" stroke="red" name="Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {logs.length > 0 && (
        <div className="bg-white shadow rounded p-4 mt-4">
          <h2 className="text-xl font-semibold mb-4">Logs</h2>
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left">
                  <th>Time</th>
                  <th>Layer</th>
                  <th>Height</th>
                  <th>PPFD</th>
                  <th>mDOU</th>
                  <th>Loss</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log, i) => (
                  <tr key={i} className="border-t">
                    <td>{new Date(log.timestamp).toLocaleTimeString()}</td>
                    <td>{log.layer}</td>
                    <td>{log.height}</td>
                    <td>{log.ppfd}</td>
                    <td>{log.mDOU}</td>
                    <td>{log.loss}</td>
                    <td><StatusBadge status={log.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default LightingDashboard;