import React from 'react';
import Layout from '@theme/Layout';

function RAGIntegration() {
  return (
    <Layout title="Textbook Assistant" description="RAG-powered Q&A for the Physical AI & Humanoid Robotics textbook">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Textbook Assistant</h1>
            <p>
              Use this AI-powered assistant to ask questions about the Physical AI & Humanoid Robotics textbook content.
              The assistant will provide answers based on the textbook material.
            </p>

            <div className="margin-vert--lg">
              <h3>How It Works</h3>
              <ul>
                <li>Select text on any textbook page to ask specific questions about that content</li>
                <li>Ask general questions about Physical AI and Humanoid Robotics concepts</li>
                <li>The assistant retrieves relevant information from the textbook to answer your questions</li>
                <li>All answers are grounded in the textbook content to ensure accuracy</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default RAGIntegration;