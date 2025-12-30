import React, { useState } from 'react';
import { useHistory } from '@docusaurus/router';

export default function RAGChatbot() {
  const [question, setQuestion] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const history = useHistory();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError('');
    setResults([]);

    try {
      const res = await fetch("https://web-production-1b36.up.railway.app/query", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          top_k: 5,
        }),
      });

      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }

      const data = await res.json();
      setResults(data);
      setQuestion('');
    } catch (err) {
      console.error(err);
      setError('Failed to query the textbook assistant.');
    } finally {
      setLoading(false);
    }
  };

  const openDoc = (page) => {
    const docPath = page.replace('.md', '');
    history.push(`/docs/${docPath}`);
  };

  return (
    <div
      style={{
        border: '1px solid #ddd',
        borderRadius: '8px',
        padding: '16px',
        marginTop: '24px',
        background: '#fafafa',
      }}
    >
      <h3>📘 Textbook Assistant</h3>

      <form onSubmit={handleSubmit} style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question from the textbook…"
            style={{
              flex: 1,
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid #ccc',
            }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              cursor: 'pointer',
              padding: '8px 16px',
              borderRadius: '4px',
              border: 'none',
              background: loading ? '#ccc' : '#007acc',
              color: '#fff',
            }}
          >
            {loading ? 'Searching…' : 'Ask'}
          </button>
        </div>
      </form>

      {error && <div style={{ color: 'red' }}>{error}</div>}

      {results.length > 0 && (
        <div>
          <h4>Relevant Sections</h4>

          {results.map((item, index) => (
            <div
              key={index}
              onClick={() => openDoc(item.page)}
              style={{
                cursor: 'pointer',
                marginBottom: '12px',
                padding: '10px',
                border: '1px solid #eee',
                borderRadius: '4px',
                background: '#fff',
              }}
            >
              <div>
                📄 <strong>{item.page.replace('.md', '')}</strong>
              </div>

              <div style={{ fontSize: '0.85em', color: '#666' }}>
                Score: {item.score.toFixed(3)}
              </div>

              <div style={{ fontSize: '0.8em', color: '#007acc' }}>
                Click to open →
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
