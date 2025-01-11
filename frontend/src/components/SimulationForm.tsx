import React, { useState } from "react";
import { SimulationData } from "../types";
import ModularVisualization from "../components/ModularVisualization";

interface SimulationFormProps {
  onSimulationComplete: (data: SimulationData) => void;
}

export const triggerSimulation = async (
  payload: {
    floor_width: number;
    floor_length: number;
    target_ppfd: number;
    floor_height: number;
    perimeter_reflectivity: number;
    min_int: number;
    max_int: number;
  },
  csrftoken: string
) => {
  console.log("triggerSimulation called with payload:", payload); // Log payload

  const response = await fetch("/api/simulation/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": csrftoken,
    },
    body: JSON.stringify(payload),
  });

  console.log("triggerSimulation response status:", response.status); // Log status

  if (!response.ok) {
    const errorData = await response.json();
    console.error("Error data:", errorData); // Log error data
    throw new Error(errorData.detail || "Failed to run simulation");
  }

  const data = await response.json();
  console.log("Raw response data:", data); // Log raw data
  return data as SimulationData;
};

const SimulationForm: React.FC<SimulationFormProps> = ({
  onSimulationComplete,
}) => {
  const [formData, setFormData] = useState({
    floor_width: "",
    floor_length: "",
    target_ppfd: "",
  });

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [simulationResult, setSimulationResult] =
    useState<SimulationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Constants based on backend configuration
  const FLOOR_HEIGHT = 2.0; // Assuming a constant height as per simulation.py
  const MIN_INT = 133.33; // Derived from MIN_LUMENS / LUMENS_TO_PPFD_CONVERSION
  const MAX_INT = 933.33; // Derived from MAX_LUMENS / LUMENS_TO_PPFD_CONVERSION

  // Handle input changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // Function to run the simulation
  const runSimulation = async (
    payload: {
        floor_width: number;
        floor_length: number;
        target_ppfd: number;
        floor_height: number;
        perimeter_reflectivity: number;
        min_int: number;
        max_int: number;
    },
    csrftoken: string | null
): Promise<SimulationData> => { // Explicitly return a Promise of type SimulationData
    try {
        if (!csrftoken) {
            throw new Error("CSRF token not found.");
        }

        // Trigger the simulation API call
        const backendSimulationData = await triggerSimulation(payload, csrftoken);
        console.log("Backend simulation data:", backendSimulationData);

        // Prepare the simulation result
        const simulationData: SimulationData = {
            optimized_lumens_by_layer: backendSimulationData.optimized_lumens_by_layer,
            mad: backendSimulationData.mad,
            optimized_ppfd: backendSimulationData.optimized_ppfd,
            floor_width: payload.floor_width,
            floor_length: payload.floor_length,
            target_ppfd: payload.target_ppfd,
            floor_height: payload.floor_height,
            fixtures: [],
        };

        // Return the simulation data
        return simulationData;

    } catch (error) {
        console.error("Simulation error:", error);
        setError(
            error instanceof Error
                ? error.message
                : "An unknown error occurred during simulation."
        );
        throw error; // Re-throw the error to be handled by handleSubmit
    } finally {
        setIsLoading(false);
    }
};

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSimulationResult(null);
  
    // Parse user inputs which are in feet
    const floorWidthFeet = parseFloat(formData.floor_width);
    const floorLengthFeet = parseFloat(formData.floor_length);
    const targetPPFD = parseFloat(formData.target_ppfd);
  
    // Convert feet to meters
    const floorWidthMeters = floorWidthFeet * 0.3048;
    const floorLengthMeters = floorLengthFeet * 0.3048;
  
    // Basic validation
    if (isNaN(floorWidthFeet) || isNaN(floorLengthFeet) || isNaN(targetPPFD)) {
      setError("Please enter valid numerical values for all fields.");
      return;
    }
  
    setIsLoading(true);
  
    try {
      const csrftoken = getCookie("csrftoken");
      console.log("CSRF Token:", csrftoken);
  
      // Prepare the payload using feet-converted values
      const payload = {
        floor_width: floorWidthFeet,
        floor_length: floorLengthFeet,
        target_ppfd: targetPPFD,
        floor_height: FLOOR_HEIGHT,
        min_int: MIN_INT,
        max_int: MAX_INT,
        perimeter_reflectivity: 0.3,
      };
  
      console.log("Submitting form with payload:", payload);
  
      runSimulation(payload, csrftoken)
        .then((simulationData) => {
          console.log("Simulation completed successfully:", simulationData);
          setSimulationResult(simulationData);
          onSimulationComplete(simulationData);
        })
        .catch((error) => {
          console.error("Simulation error:", error);
          setError(error.message || "An unknown error occurred during simulation.");
        })
        .finally(() => {
          setIsLoading(false);
        });
    } catch (error) {
      console.error("Error in handleSubmit:", error);
      setError(
        error instanceof Error
          ? error.message
          : "An unknown error occurred in handleSubmit."
      );
      setIsLoading(false);
    }
  };

  // Define getCookie here:
  function getCookie(name: string) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
      const cookies = document.cookie.split(";");
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + "=")) {
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
          {/* Form inputs for floor_width, floor_length, target_ppfd */}
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

          {/* Submit button */}
          <button type="submit" disabled={isLoading}>
            {isLoading ? "Running Simulation..." : "Run Simulation"}
          </button>
        </form>

        {/* Error and loading messages */}
        {error && <p className="error-message">{error}</p>}
        {isLoading && <div>Loading...</div>}

        {/* Simulation Results */}
        {simulationResult && (
          <div className="simulation-results-container">
            <h2>Simulation Results</h2>
            <div className="simulation-metrics">
              <p>
                <strong>Minimized MAD:</strong>{" "}
                {simulationResult.mad.toFixed(2)}
              </p>
              <p>
                <strong>Optimized PPFD:</strong>{" "}
                {simulationResult.optimized_ppfd.toFixed(2)} µmol/m²/s
              </p>
            </div>
            <div className="optimized-lumens">
              <h3>Optimized Lumens by Layer</h3>
              <ul>
                {simulationResult.optimized_lumens_by_layer.map(
                  (lumens, index) => (
                    <li key={index}>
                      Layer {index + 1}: {lumens.toFixed(2)} lumens
                    </li>
                  )
                )}
              </ul>
            </div>
          </div>
        )}
      </div>
      {/* MODULAR VISUALIZATION MOVED UP */}
      {/* Modular Visualization */}
      {simulationResult && (
        <div className="modular-visualization-wrapper">
          <ModularVisualization
            mad={simulationResult.mad}
            optimized_ppfd={simulationResult.optimized_ppfd}
            floorWidth={simulationResult.floor_width}
            floorLength={simulationResult.floor_length}
            floorHeight={simulationResult.floor_height}
            optimizedLumensByLayer={
              simulationResult.optimized_lumens_by_layer
            }
            simulationResult={simulationResult}
          />
        </div>
      )}
    </div>
  );
};

export default SimulationForm;