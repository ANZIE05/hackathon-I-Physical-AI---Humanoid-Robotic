import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '8b1'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '31b'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '385'),
            routes: [
              {
                path: '/docs/appendices/further-learning',
                component: ComponentCreator('/docs/appendices/further-learning', 'e1b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendices/glossary',
                component: ComponentCreator('/docs/appendices/glossary', 'aed'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendices/tool-references',
                component: ComponentCreator('/docs/appendices/tool-references', 'a8a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/appendices/troubleshooting',
                component: ComponentCreator('/docs/appendices/troubleshooting', 'e74'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/assessments/capstone-project',
                component: ComponentCreator('/docs/assessments/capstone-project', '569'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/assessments/gazebo-project',
                component: ComponentCreator('/docs/assessments/gazebo-project', '106'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/assessments/isaac-project',
                component: ComponentCreator('/docs/assessments/isaac-project', 'a0a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/assessments/ros2-project',
                component: ComponentCreator('/docs/assessments/ros2-project', 'f21'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/capstone/capstone-architecture',
                component: ComponentCreator('/docs/capstone/capstone-architecture', '5e0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/capstone/capstone-evaluation',
                component: ComponentCreator('/docs/capstone/capstone-evaluation', 'efd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/capstone/end-to-end-pipeline',
                component: ComponentCreator('/docs/capstone/end-to-end-pipeline', 'f51'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/hardware/edge-ai-kits',
                component: ComponentCreator('/docs/hardware/edge-ai-kits', '63f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/hardware/lab-models',
                component: ComponentCreator('/docs/hardware/lab-models', '160'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/hardware/robot-options',
                component: ComponentCreator('/docs/hardware/robot-options', 'd5f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/hardware/sensor-stack',
                component: ComponentCreator('/docs/hardware/sensor-stack', '6f1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/hardware/workstation-requirements',
                component: ComponentCreator('/docs/hardware/workstation-requirements', '345'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/how-to-use',
                component: ComponentCreator('/docs/how-to-use', 'f96'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/learning-outcomes',
                component: ComponentCreator('/docs/learning-outcomes', '769'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/rclpy-integration',
                component: ComponentCreator('/docs/module-1/rclpy-integration', '24c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/ros2-architecture',
                component: ComponentCreator('/docs/module-1/ros2-architecture', '58b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/ros2-fundamentals',
                component: ComponentCreator('/docs/module-1/ros2-fundamentals', 'd9b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/ros2-practical-labs',
                component: ComponentCreator('/docs/module-1/ros2-practical-labs', 'a6a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/ros2-review-questions',
                component: ComponentCreator('/docs/module-1/ros2-review-questions', '67d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1/urdf-basics',
                component: ComponentCreator('/docs/module-1/urdf-basics', 'c35'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/digital-twin-practical-labs',
                component: ComponentCreator('/docs/module-2/digital-twin-practical-labs', '090'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/digital-twin-review-questions',
                component: ComponentCreator('/docs/module-2/digital-twin-review-questions', 'ac1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/gazebo-fundamentals',
                component: ComponentCreator('/docs/module-2/gazebo-fundamentals', '8d4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/gazebo-workflows',
                component: ComponentCreator('/docs/module-2/gazebo-workflows', '2e2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/physics-simulation',
                component: ComponentCreator('/docs/module-2/physics-simulation', 'e1f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/sensor-simulation',
                component: ComponentCreator('/docs/module-2/sensor-simulation', '6e6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2/unity-visualization',
                component: ComponentCreator('/docs/module-2/unity-visualization', '84b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3/isaac-ros-integration',
                component: ComponentCreator('/docs/module-3/isaac-ros-integration', '3ba'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3/isaac-sim-fundamentals',
                component: ComponentCreator('/docs/module-3/isaac-sim-fundamentals', 'e12'),
                exact: true
              },
              {
                path: '/docs/module-3/isaac-sim-introduction',
                component: ComponentCreator('/docs/module-3/isaac-sim-introduction', 'ef7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3/isaac-sim-workflows',
                component: ComponentCreator('/docs/module-3/isaac-sim-workflows', 'e9e'),
                exact: true
              },
              {
                path: '/docs/module-3/sim-to-real-principles',
                component: ComponentCreator('/docs/module-3/sim-to-real-principles', '9a1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3/vslam-practical-labs',
                component: ComponentCreator('/docs/module-3/vslam-practical-labs', '271'),
                exact: true
              },
              {
                path: '/docs/module-3/vslam-systems',
                component: ComponentCreator('/docs/module-3/vslam-systems', '11f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4/vla-fundamentals',
                component: ComponentCreator('/docs/module-4/vla-fundamentals', '0ad'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/rag/chatbot-overview',
                component: ComponentCreator('/docs/rag/chatbot-overview', 'b93'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/rag/content-chunking',
                component: ComponentCreator('/docs/rag/content-chunking', '807'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/rag/embedding-strategy',
                component: ComponentCreator('/docs/rag/embedding-strategy', '7be'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/rag/response-logic',
                component: ComponentCreator('/docs/rag/response-logic', 'a05'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/textbook-assistant',
                component: ComponentCreator('/docs/textbook-assistant', '9b8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tooling-environment',
                component: ComponentCreator('/docs/tooling-environment', 'd31'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/weekly-breakdown',
                component: ComponentCreator('/docs/weekly-breakdown', '776'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/why-physical-ai-matters',
                component: ComponentCreator('/docs/why-physical-ai-matters', '2f4'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
