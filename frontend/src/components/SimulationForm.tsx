import React, { useState } from "react";
import { SimulationData } from "../types";
import ModularVisualization from "../components/ModularVisualization";

export const triggerSimulation = async (
  payload: { floor_width: number; floor_length: number; target_ppfd: number },
  csrftoken: string
) => {
  const response = await fetch("/api/ml_simulation/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrftoken,
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to run simulation");
  }
  const data = await response.json();
  return data as SimulationData;
};

interface SimulationFormProps {
  onSimulationComplete: (data: SimulationData) => void;
}

const SimulationForm: React.FC<SimulationFormProps> = ({ onSimulationComplete }) => {
  const [formData, setFormData] = useState({
    floor_width: "",
    floor_length: "",
    target_ppfd: "",
  });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [simulationResult, setSimulationResult] = useState<SimulationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const runSimulation = async (
    payload: { floor_width: number; floor_length: number; target_ppfd: number },
    csrftoken: string | null
  ): Promise<SimulationData> => {
    try {
      if (!csrftoken) {
        throw new Error("CSRF token not found.");
      }
      const backendSimulationData = await triggerSimulation(payload, csrftoken);
      return {
        optimized_lumens_by_layer: backendSimulationData.optimized_lumens_by_layer,
        mad: backendSimulationData.mad,
        optimized_ppfd: backendSimulationData.optimized_ppfd,
        floor_width: payload.floor_width,
        floor_length: payload.floor_length,
        target_ppfd: payload.target_ppfd,
        floor_height: backendSimulationData.floor_height,
        fixtures: [],
      };
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "An unknown error occurred during simulation."
      );
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSimulationResult(null);

    const floorWidthFeet = parseFloat(formData.floor_width);
    const floorLengthFeet = parseFloat(formData.floor_length);
    const targetPPFD = parseFloat(formData.target_ppfd);

    if (isNaN(floorWidthFeet) || isNaN(floorLengthFeet) || isNaN(targetPPFD)) {
      setError("Please enter valid numerical values for all fields.");
      return;
    }

    setIsLoading(true);
    try {
      const csrftoken = getCookie("csrftoken");
      const payload = {
        floor_width: floorWidthFeet,
        floor_length: floorLengthFeet,
        target_ppfd: targetPPFD,
      };

      runSimulation(payload, csrftoken)
        .then((simulationData) => {
          setSimulationResult(simulationData);
          onSimulationComplete(simulationData);
        })
        .catch((error) => {
          setError(error.message || "An unknown error occurred during simulation.");
        })
        .finally(() => {
          setIsLoading(false);
        });
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "An unknown error occurred in handleSubmit."
      );
      setIsLoading(false);
    }
  };

  // Utility function to get a cookie by name
  function getCookie(name: string) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
      const cookies = document.cookie.split(";");
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === name + "=") {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  return (
    <div className="form-and-visualization-container">
      <div className="simulation-form-container">
        <h2>Run Simulation</h2>
        <form onSubmit={handleSubmit}>
          <div>
            <label htmlFor="floor_width">Floor Width (feet):</label>
            <input
              type="number"
              id="floor_width"
              name="floor_width"
              value={formData.floor_width}
              onChange={handleChange}
              required
              step="0.01"
              min="0"
            />
          </div>
          <div>
            <label htmlFor="floor_length">Floor Length (feet):</label>
            <input
              type="number"
              id="floor_length"
              name="floor_length"
              value={formData.floor_length}
              onChange={handleChange}
              required
              step="0.01"
              min="0"
            />
          </div>
          <div>
            <label htmlFor="target_ppfd">Target PPFD (µmol/m²/s):</label>
            <input
              type="number"
              id="target_ppfd"
              name="target_ppfd"
              value={formData.target_ppfd}
              onChange={handleChange}
              required
              step="0.01"
              min="0"
            />
          </div>
          <button type="submit" disabled={isLoading}>
            {isLoading ? "Running Simulation..." : "Run Simulation"}
          </button>
        </form>
        {error && <p className="error-message">{error}</p>}
        {isLoading && <div>Loading...</div>}
        {simulationResult && (
          <div className="simulation-results-container">
            <h2>Simulation Results</h2>
            <div className="simulation-metrics">
              <p>
                <strong>Minimized MAD:</strong> {simulationResult.mad.toFixed(2)}
              </p>
              <p>
                <strong>Optimized PPFD:</strong> {simulationResult.optimized_ppfd.toFixed(2)} µmol/m²/s
              </p>
            </div>
            <div className="optimized-lumens">
              <h3>Optimized Lumens by Layer</h3>
              <ul>
                {simulationResult.optimized_lumens_by_layer.map((lumens, index) => (
                  <li key={index}>
                    Layer {index + 1}: {lumens.toFixed(2)} lumens
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
      {simulationResult && (
        <div className="modular-visualization-wrapper">
          <ModularVisualization
            mad={simulationResult.mad}
            optimized_ppfd={simulationResult.optimized_ppfd}
            floorWidth={simulationResult.floor_width}
            floorLength={simulationResult.floor_length}
            floorHeight={simulationResult.floor_height}
            optimizedLumensByLayer={simulationResult.optimized_lumens_by_layer}
            simulationResult={simulationResult}
          />
        </div>
      )}
    </div>
  );
};

export default SimulationForm;
