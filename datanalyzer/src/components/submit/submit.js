import React, { useState } from 'react';
import './Submit.css'; // Import your CSS file for styling

function Submit() {
  const [csvFile, setCsvFile] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [responseValue, setResponseValue] = useState('');
  const [latestImageUrl, setLatestImageUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false); // Loading state

  const handleFileChange = (e) => {
    setCsvFile(e.target.files[0]);
  };

  const handlePromptChange = (e) => {
    setPrompt(e.target.value);
  };

  const handleCsvSubmit = async () => {
    if (csvFile) {
      const formData = new FormData();
      formData.append('file', csvFile);

      try {
        setIsLoading(true); // Start loading
        const response = await fetch('http://localhost:5003/files/upload', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          console.log('CSV file uploaded successfully');
          alert('CSV file uploaded successfully');
        } else {
          alert('Failed to upload CSV file');
          console.error('Failed to upload CSV file');
        }
      } catch (error) {
        alert('Error uploading CSV file');
        console.error('Error uploading CSV file:', error);
      } finally {
        setIsLoading(false); // Stop loading
      }
    } else {
      alert('Please upload a CSV file.');
    }
  };

  const handlePromptSubmit = async () => {
    if (prompt) {
      setIsLoading(true); // Start loading
      console.log('Sending prompt:', prompt);

      try {
        const response = await fetch('http://localhost:5000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ prompt }),
        });

        if (response.ok) {
          const data = await response.json();
          console.log('Prompt response:', data);
          setResponseValue(data.response); // Set response value
          setLatestImageUrl(data.latest_image_url); // Set latest image URL
        } else {
          console.error('Failed to send prompt');
        }
      } catch (error) {
        console.error('Error sending prompt:', error);
      } finally {
        setIsLoading(false); // Stop loading
      }
    } else {
      alert('Please enter a prompt.');
    }
  };

  const renderResponse = () => {
    if (isLoading) {
      return <p>Loading...</p>; // Show loading indicator
    } else {
      return (
        <div className="response-container">
          <h3>Text Response:</h3>
          <p>{responseValue}</p>
          {latestImageUrl && (
            <div className="image-response">
              <h3>Image Response:</h3>
              <img src={latestImageUrl} alt="Response Image" className="response-image" />
            </div>
          )}
        </div>
      );
    }
  };

  return (
    <div className="container">
      <div className="form-wrapper">
        <h2>Upload CSV and Enter Prompt</h2>
        <div className="input-group">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="input-file"
          />
          <button onClick={handleCsvSubmit} className="submit-button">
            Submit CSV
          </button>
        </div>
        <div className="input-group">
          <input
            type="text"
            value={prompt}
            onChange={handlePromptChange}
            placeholder="Enter your prompt"
            className="input-text"
          />
          <button onClick={handlePromptSubmit} className="submit-button">
            Submit Prompt
          </button>
        </div>
      </div>

      {/* Render response */}
      {renderResponse()}
    </div>
  );
}

export default Submit;
