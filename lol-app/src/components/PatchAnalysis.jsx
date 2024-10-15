import React, { useState, useEffect } from "react";
import { Tabs, TabList, Tab, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";

const PatchAnalysis = () => {
  const [patches, setPatches] = useState([]);
  const [selectedPatch, setSelectedPatch] = useState(null);
  const [patchData, setPatchData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/api/patches")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (Array.isArray(data)) {
          setPatches(data);
          if (data.length > 0) {
            setSelectedPatch(data[0]);
          }
        } else {
          throw new Error("Received data is not an array");
        }
      })
      .catch((err) => {
        console.error("Error fetching patches:", err);
        setError(`Failed to fetch patches: ${err.message}`);
      });
  }, []);

  useEffect(() => {
    if (selectedPatch) {
      fetch(`/api/patch-data/${selectedPatch}`)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then((data) => {
          setPatchData(data);
        })
        .catch((err) => {
          console.error(`Error fetching data for patch ${selectedPatch}:`, err);
          setError(
            `Failed to fetch data for patch ${selectedPatch}: ${err.message}`
          );
        });
    }
  }, [selectedPatch]);

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="patch-analysis">
      <h2>Patch Analysis</h2>
      {patches.length === 0 ? (
        <p>Loading patches...</p>
      ) : (
        <Tabs>
          <TabList>
            {patches.map((patch) => (
              <Tab key={patch} onClick={() => setSelectedPatch(patch)}>
                {patch}
              </Tab>
            ))}
          </TabList>

          {patches.map((patch) => (
            <TabPanel key={patch}>
              {selectedPatch === patch && patchData ? (
                <div>
                  <h3>Analysis for Patch {patch}</h3>
                  <pre>{JSON.stringify(patchData, null, 2)}</pre>
                </div>
              ) : (
                <p>Loading patch data for {patch}...</p>
              )}
            </TabPanel>
          ))}
        </Tabs>
      )}
    </div>
  );
};

export default PatchAnalysis;
