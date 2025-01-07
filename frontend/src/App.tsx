// frontend/src/App.tsx
import React from 'react';
import SimulationForm from './components/SimulationForm'; // Correct path

function App() {
  return (
    <div className="App">
      <SimulationForm onSimulationComplete={() => {}} />
    </div>
  );
}

export default App;