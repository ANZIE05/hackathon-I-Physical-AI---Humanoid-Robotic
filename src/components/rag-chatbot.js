import React, { useState, useEffect } from 'react';

const RAGChatbot = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [selectedText, setSelectedText] = useState('');

  // Function to get selected text from the page
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('selectionchange', handleSelection);
    return () => {
      document.removeEventListener('selectionchange', handleSelection);
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);

    try {
      // Prepare the request payload
      const requestBody = {
        query: query,
        selected_text: selectedText || null,
        top_k: 5,
        temperature: 0.7
      };

      // Call the RAG API - adjust the URL based on your deployment
      const response = await fetch('/rag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();

      setResponse(data.response);

      // Add to history
      setHistory(prev => [
        ...prev,
        { query, response: data.response, sources: data.sources }
      ]);

      // Clear the input
      setQuery('');
    } catch (error) {
      console.error('Error calling RAG API:', error);
      setResponse('Sorry, there was an error processing your request.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="rag-chatbot-container" style={{
      border: '1px solid #ccc',
      borderRadius: '8px',
      padding: '16px',
      marginTop: '16px',
      backgroundColor: '#f9f9f9'
    }}>
      <h3>Textbook Assistant</h3>

      {selectedText && (
        <div style={{
          backgroundColor: '#e3f2fd',
          padding: '8px',
          borderRadius: '4px',
          marginBottom: '8px',
          fontSize: '0.9em'
        }}>
          <strong>Selected Text:</strong> {selectedText.substring(0, 100)}...
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about this textbook..."
            style={{
              flex: 1,
              padding: '8px',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            style={{
              padding: '8px 16px',
              backgroundColor: isLoading ? '#ccc' : '#007cba',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isLoading ? 'not-allowed' : 'pointer'
            }}
          >
            {isLoading ? 'Asking...' : 'Ask'}
          </button>
        </div>
      </form>

      {response && (
        <div style={{
          padding: '12px',
          backgroundColor: 'white',
          borderRadius: '4px',
          border: '1px solid #eee'
        }}>
          <h4>Answer:</h4>
          <p>{response}</p>
        </div>
      )}

      {history.length > 0 && (
        <div style={{ marginTop: '16px' }}>
          <h4>Chat History:</h4>
          <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
            {history.map((item, index) => (
              <div key={index} style={{
                marginBottom: '12px',
                paddingBottom: '12px',
                borderBottom: '1px solid #eee'
              }}>
                <div><strong>Q:</strong> {item.query}</div>
                <div><strong>A:</strong> {item.response}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default RAGChatbot;