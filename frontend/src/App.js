import React, { useState, useEffect } from 'react';
import api from './api';
import ImageCarousel from './ImageCarousel';
import './App.css';

// Main App Component
function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [currentView, setCurrentView] = useState('dashboard');
  const [caseData, setCaseData] = useState({
    case_id: '',
    patient_age: '',
    patient_sex: 'Male',
    clinical_history: '',
    imaging_modality: 'MRI',
    anatomical_region: 'Brain',
    image_description: ''
  });
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [prompts, setPrompts] = useState([]);
  const [selectedPrompt, setSelectedPrompt] = useState(null);

  // Check system status on load
  useEffect(() => {
    checkSystemStatus();
    loadPrompts();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await api.get('/health');
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Failed to check system status:', error);
      setSystemStatus({ status: 'error', error: error.message });
    }
  };

  const loadPrompts = async () => {
    try {
      const response = await api.get('/api/prompts');
      setPrompts(response.data.prompts);
    } catch (error) {
      console.error('Failed to load prompts:', error);
    }
  };

  const handleCaseSubmit = async (e) => {
    e.preventDefault();
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const caseWithId = {
        ...caseData,
        case_id: `case_${Date.now()}`
      };

      const response = await api.post('/api/analyze-case', caseWithId);
      setAnalysisResult(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
      let errorMessage = 'Unknown error occurred';
      
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        errorMessage = 'Request timed out. The analysis is taking longer than expected. This usually happens when the AI models are processing complex cases. Please try again with a simpler case or contact support.';
      } else if (error.response) {
        errorMessage = error.response.data?.detail || error.response.statusText || 'Server error';
      } else if (error.request) {
        errorMessage = 'No response from server. Please check your connection.';
      } else {
        errorMessage = error.message;
      }
      
      setAnalysisResult({
        error: true,
        message: errorMessage,
        details: {
          code: error.code,
          timeout: api.defaults.timeout,
          actualError: error.message
        }
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setCaseData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Dashboard Component
  const Dashboard = () => (
    <div className="dashboard">
      <h2>🏥 Radiology AI System Dashboard</h2>
      
      <div className="status-grid">
        <div className="status-card">
          <h3>System Status</h3>
          <div className={`status-indicator ${systemStatus?.status}`}>
            {systemStatus?.status || 'Unknown'}
          </div>
          {systemStatus && (
            <div className="status-details">
              <p>LangSmith: {systemStatus.langsmith_enabled ? '✅ Enabled' : '❌ Disabled'}</p>
              <p>Project: {systemStatus.langsmith_project}</p>
              <p>Models: {systemStatus.models_available?.join(', ')}</p>
              <p>Database: {systemStatus.database_connected ? '✅ Connected' : '❌ Disconnected'}</p>
            </div>
          )}
        </div>

        <div className="status-card">
          <h3>LangSmith Integration</h3>
          <div className="langsmith-info">
            <p>🔍 <strong>Observability:</strong> Full tracing enabled</p>
            <p>📊 <strong>Project:</strong> {systemStatus?.langsmith_project}</p>
            <p>🔗 <strong>Dashboard:</strong> 
              <a href={`https://smith.langchain.com/projects/${systemStatus?.langsmith_project}`} 
                 target="_blank" rel="noopener noreferrer">
                View in LangSmith
              </a>
            </p>
          </div>
        </div>

        <div className="status-card">
          <h3>Available Models</h3>
          <div className="models-list">
            <div className="model-item">🧠 Claude (Medical Reasoning)</div>
            <div className="model-item">📄 Mistral (Document Processing)</div>
            <div className="model-item">🔍 DeepSeek (Search Optimization)</div>
          </div>
        </div>

        <div className="status-card">
          <h3>Quick Actions</h3>
          <div className="quick-actions">
            <button onClick={() => setCurrentView('analyze')} className="action-btn primary">
              🔬 New Case Analysis
            </button>
            <button onClick={() => setCurrentView('prompts')} className="action-btn secondary">
              ⚙️ Manage Prompts
            </button>
            <button onClick={checkSystemStatus} className="action-btn secondary">
              🔄 Refresh Status
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  // Case Analysis Component
  const CaseAnalysis = () => (
    <div className="case-analysis">
      <h2>🔬 AI-Powered Case Analysis</h2>
      
      <form onSubmit={handleCaseSubmit} className="case-form">
        <div className="form-grid">
          <div className="form-group">
            <label>Patient Age</label>
            <input
              type="number"
              name="patient_age"
              value={caseData.patient_age}
              onChange={handleInputChange}
              required
              min="0"
              max="120"
            />
          </div>

          <div className="form-group">
            <label>Patient Sex</label>
            <select
              name="patient_sex"
              value={caseData.patient_sex}
              onChange={handleInputChange}
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div className="form-group">
            <label>Imaging Modality</label>
            <select
              name="imaging_modality"
              value={caseData.imaging_modality}
              onChange={handleInputChange}
            >
              <option value="MRI">MRI</option>
              <option value="CT">CT</option>
              <option value="X-ray">X-ray</option>
              <option value="Ultrasound">Ultrasound</option>
              <option value="PET">PET</option>
              <option value="Nuclear Medicine">Nuclear Medicine</option>
            </select>
          </div>

          <div className="form-group">
            <label>Anatomical Region</label>
            <select
              name="anatomical_region"
              value={caseData.anatomical_region}
              onChange={handleInputChange}
            >
              <option value="Brain">Brain</option>
              <option value="Spine">Spine</option>
              <option value="Chest">Chest</option>
              <option value="Abdomen">Abdomen</option>
              <option value="Pelvis">Pelvis</option>
              <option value="Extremities">Extremities</option>
              <option value="Head and Neck">Head and Neck</option>
            </select>
          </div>
        </div>

        <div className="form-group full-width">
          <label>Clinical History</label>
          <textarea
            name="clinical_history"
            value={caseData.clinical_history}
            onChange={handleInputChange}
            placeholder="Enter patient's clinical history, symptoms, and relevant medical background..."
            rows="4"
            required
          />
        </div>

        <div className="form-group full-width">
          <label>Image Description</label>
          <textarea
            name="image_description"
            value={caseData.image_description}
            onChange={handleInputChange}
            placeholder="Detailed description of imaging findings, including sequences, enhancement patterns, measurements, etc..."
            rows="6"
            required
          />
        </div>

        <button 
          type="submit" 
          className="analyze-btn"
          disabled={isAnalyzing}
        >
          {isAnalyzing ? '🔄 Analyzing with AI...' : '🚀 Start AI Analysis'}
        </button>
      </form>

      {isAnalyzing && (
        <div className="analysis-progress">
          <div className="progress-steps">
            <div className="step active">🧠 Extracting Radiology Context</div>
            <div className="step active">🔍 Searching Medical Literature</div>
            <div className="step active">📄 Processing Documents</div>
            <div className="step active">⚕️ Generating Diagnosis</div>
          </div>
          <p>AI agents are working on your case... This may take 60-90 seconds for complex analysis.</p>
        </div>
      )}

      {analysisResult && (
        <div className="analysis-results">
          {analysisResult.error ? (
            <div className="error-result">
              <h3>❌ Analysis Failed</h3>
              <p>{analysisResult.message}</p>
              {analysisResult.details && (
                <div className="error-details">
                  <p><small>Error code: {analysisResult.details.code}</small></p>
                  <p><small>Configured timeout: {analysisResult.details.timeout}ms</small></p>
                  <p><small>Actual error: {analysisResult.details.actualError}</small></p>
                </div>
              )}
            </div>
          ) : (
            <div className="success-result">
              <h3>✅ Analysis Complete</h3>
              
              <div className="results-grid">
                <div className="result-section">
                  <h4>🎯 Primary Diagnosis</h4>
                  <div className="diagnosis-card">
                    <h5>{analysisResult.diagnosis_result?.primary_diagnosis?.diagnosis}</h5>
                    <p><strong>Confidence:</strong> {(analysisResult.diagnosis_result?.primary_diagnosis?.confidence_score * 100).toFixed(1)}%</p>
                    <p><strong>Reasoning:</strong> {analysisResult.diagnosis_result?.primary_diagnosis?.reasoning}</p>
                  </div>
                </div>

                <div className="result-section">
                  <h4>🔍 Radiology Context</h4>
                  <div className="context-details">
                    <p><strong>Anatomy:</strong> {analysisResult.radiology_context?.anatomy?.join(', ')}</p>
                    <p><strong>Modality:</strong> {analysisResult.radiology_context?.imaging_modality}</p>
                    <p><strong>Enhancement:</strong> {analysisResult.radiology_context?.enhancement_pattern?.join(', ')}</p>
                  </div>
                </div>

                <div className="result-section full-width">
                  <h4>📚 Literature Evidence ({analysisResult.literature_matches?.length || 0} papers with images)</h4>
                  <div className="literature-list enhanced">
                    {analysisResult.literature_matches?.map((match, index) => (
                      <div key={index} className="literature-item enhanced">
                        <div className="literature-header">
                          <h6>{index + 1}. {match.title}</h6>
                          <a href={match.url} target="_blank" rel="noopener noreferrer" className="literature-link">
                            🔗 View Paper
                          </a>
                        </div>
                        <div className="literature-meta">
                          <span className="journal">{match.journal}</span>
                          <span className="year">{match.year}</span>
                          <span className="relevance">Relevance: {(match.relevance_score * 100).toFixed(0)}%</span>
                          {match.extracted_images?.length > 0 && (
                            <span className="image-count">📷 {match.extracted_images.length} images</span>
                          )}
                        </div>
                        <p className="match-reasoning">{match.match_reasoning}</p>
                        {match.relevant_sections?.length > 0 && (
                          <div className="relevant-sections">
                            <strong>Key Findings:</strong>
                            <ul>
                              {match.relevant_sections.slice(0, 2).map((section, idx) => (
                                <li key={idx}>{section.substring(0, 150)}...</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {match.extracted_images?.length > 0 && (
                          <ImageCarousel 
                            images={match.extracted_images} 
                            paperTitle={match.title}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="result-section">
                  <h4>🔬 Differential Diagnoses</h4>
                  <div className="differential-list">
                    {analysisResult.diagnosis_result?.differential_diagnoses?.map((diff, index) => (
                      <div key={index} className="differential-item">
                        <h6>{diff.diagnosis}</h6>
                        <p><strong>Probability:</strong> {(diff.probability * 100).toFixed(1)}%</p>
                        <p>{diff.reasoning}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="metadata">
                <h4>📊 Processing Metadata</h4>
                <p><strong>Case ID:</strong> {analysisResult.case_id}</p>
                <p><strong>Models Used:</strong> {analysisResult.processing_metadata?.models_used?.join(', ')}</p>
                <p><strong>Literature Sources:</strong> {analysisResult.processing_metadata?.literature_sources}</p>
                <p><strong>LangSmith Project:</strong> {analysisResult.processing_metadata?.langsmith_project}</p>
                <p><strong>Timestamp:</strong> {new Date(analysisResult.processing_metadata?.timestamp).toLocaleString()}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );

  // Prompt Management Component
  const PromptManagement = () => {
    const [isSaving, setIsSaving] = useState(false);
    const [saveMessage, setSaveMessage] = useState('');

    const handleSavePrompt = async () => {
      if (!selectedPrompt) return;
      
      setIsSaving(true);
      setSaveMessage('');
      
      try {
        const response = await api.put(`/api/prompts/${selectedPrompt.template_id}`, selectedPrompt);
        
        if (response.data.success) {
          setSaveMessage(`✅ ${response.data.message}`);
          // Reload prompts to show new version
          await loadPrompts();
          // Update selected prompt with new version
          setSelectedPrompt({
            ...selectedPrompt,
            version: response.data.version
          });
        }
      } catch (error) {
        console.error('Failed to save prompt:', error);
        setSaveMessage('❌ Failed to save prompt: ' + (error.response?.data?.detail || error.message));
      } finally {
        setIsSaving(false);
      }
    };

    return (
      <div className="prompt-management">
        <h2>⚙️ Prompt Management</h2>
        
        <div className="prompt-info-banner">
          <p>💡 <strong>Tip:</strong> Edit these prompts to customize how the AI agents behave. Changes take effect immediately for new analyses.</p>
        </div>
        
        {prompts.length === 0 ? (
          <div className="no-prompts">
            <p>No prompts available. Initialize default prompts by running:</p>
            <code>docker exec radiology-backend python initialize_prompts.py</code>
          </div>
        ) : (
          <div className="prompts-grid">
            {prompts.map((prompt, index) => (
              <div key={prompt.template_id} className="prompt-card">
                <h4>{prompt.name}</h4>
                <p><strong>ID:</strong> {prompt.template_id}</p>
                <p><strong>Version:</strong> {prompt.version}</p>
                <p><strong>Model:</strong> <span className={`model-badge ${prompt.model_type}`}>{prompt.model_type}</span></p>
                <p><strong>Description:</strong> {prompt.description}</p>
                <p><strong>Variables:</strong> {prompt.input_variables?.join(', ')}</p>
                <button 
                  onClick={() => setSelectedPrompt(prompt)}
                  className="view-prompt-btn"
                >
                  View/Edit
                </button>
              </div>
            ))}
          </div>
        )}

        {selectedPrompt && (
          <div className="prompt-editor">
            <div className="editor-header">
              <h3>Editing: {selectedPrompt.name}</h3>
              <span className="version-badge">v{selectedPrompt.version}</span>
            </div>
            
            <div className="editor-info">
              <p><strong>Model:</strong> {selectedPrompt.model_type}</p>
              <p><strong>Variables:</strong> {selectedPrompt.input_variables?.map(v => `{${v}}`).join(', ')}</p>
            </div>
            
            <textarea
              value={selectedPrompt.template_text}
              onChange={(e) => setSelectedPrompt({
                ...selectedPrompt,
                template_text: e.target.value
              })}
              rows="20"
              className="prompt-textarea"
              placeholder="Enter your prompt template here..."
            />
            
            {saveMessage && (
              <div className={`save-message ${saveMessage.startsWith('✅') ? 'success' : 'error'}`}>
                {saveMessage}
              </div>
            )}
            
            <div className="prompt-actions">
              <button 
                className="save-btn"
                onClick={handleSavePrompt}
                disabled={isSaving}
              >
                {isSaving ? '💾 Saving...' : '💾 Save New Version'}
              </button>
              <button 
                className="cancel-btn"
                onClick={() => {
                  setSelectedPrompt(null);
                  setSaveMessage('');
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Navigation
  const Navigation = () => (
    <nav className="navigation">
      <div className="nav-brand">
        <h1>🏥 Radiology AI</h1>
        <span className="nav-subtitle">LangChain + LangSmith</span>
      </div>
      <div className="nav-links">
        <button 
          className={currentView === 'dashboard' ? 'nav-link active' : 'nav-link'}
          onClick={() => setCurrentView('dashboard')}
        >
          📊 Dashboard
        </button>
        <button 
          className={currentView === 'analyze' ? 'nav-link active' : 'nav-link'}
          onClick={() => setCurrentView('analyze')}
        >
          🔬 Analyze Case
        </button>
        <button 
          className={currentView === 'prompts' ? 'nav-link active' : 'nav-link'}
          onClick={() => setCurrentView('prompts')}
        >
          ⚙️ Prompts
        </button>
      </div>
    </nav>
  );

  // Main render
  return (
    <div className="App">
      <Navigation />
      <main className="main-content">
        {currentView === 'dashboard' && <Dashboard />}
        {currentView === 'analyze' && <CaseAnalysis />}
        {currentView === 'prompts' && <PromptManagement />}
      </main>
    </div>
  );
}

export default App;

