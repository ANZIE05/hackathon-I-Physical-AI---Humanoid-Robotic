// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
        'why-physical-ai-matters',
        'learning-outcomes',
        'how-to-use',
        'tooling-environment',
        'weekly-breakdown'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1/ros2-fundamentals',
        'module-1/ros2-architecture',
        'module-1/urdf-basics',
        'module-1/rclpy-integration',
        'module-1/ros2-practical-labs',
        'module-1/ros2-review-questions'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2/gazebo-fundamentals',
        'module-2/physics-simulation',
        'module-2/sensor-simulation',
        'module-2/gazebo-workflows',
        'module-2/unity-visualization',
        'module-2/digital-twin-practical-labs',
        'module-2/digital-twin-review-questions'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3/isaac-sim-introduction',
        'module-3/isaac-ros-integration',
        'module-3/vslam-systems',
        // 'module-3/nav2-humanoid-navigation',
        'module-3/sim-to-real-principles',
        // 'module-3/isaac-practical-labs',
        // 'module-3/isaac-review-questions'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4/vla-fundamentals',
        // 'module-4/whisper-voice-action',
        // 'module-4/llm-ros-integration',
        // 'module-4/multimodal-interaction',
        // 'module-4/vla-practical-labs',
        // 'module-4/vla-review-questions'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone/capstone-architecture',
        'capstone/end-to-end-pipeline',
        'capstone/capstone-evaluation'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Assessment Projects',
      items: [
        'assessments/ros2-project',
        'assessments/gazebo-project',
        'assessments/isaac-project',
        'assessments/capstone-project'
      ],
      collapsed: true
    },
    {
      type: 'category',
      label: 'Hardware & Lab Architecture',
      items: [
        'hardware/workstation-requirements',
        'hardware/edge-ai-kits',
        'hardware/sensor-stack',
        'hardware/robot-options',
        'hardware/lab-models'
      ],
      collapsed: true
    },
    {
      type: 'category',
      label: 'RAG Chatbot Integration',
      items: [
        'rag/chatbot-overview',
        'rag/content-chunking',
        'rag/embedding-strategy',
        'rag/response-logic'
      ],
      collapsed: true
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/glossary',
        'appendices/tool-references',
        'appendices/further-learning',
        'appendices/troubleshooting'
      ],
      collapsed: true
    },
    {
      type: 'doc',
      id: 'textbook-assistant',
      label: 'Textbook Assistant',
    },
  ],
};

module.exports = sidebars;