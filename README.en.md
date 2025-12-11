<div align="center">
   
# Awesome C++ Autonomous Driving
<p align="center">
  <img src="https://img.shields.io/github/stars/0voice/Awesome-CPP-Autonomous-Driving?style=flat-square&label=Stars&color=FFCA28&logo=star&labelColor=000" alt="stars" />
  <img src="https://img.shields.io/github/forks/0voice/Awesome-CPP-Autonomous-Driving?style=flat-square&label=Forks&color=ff6b6b" alt="forks" />
  <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2F0voice%2FAwesome-CPP-Autonomous-Driving&label=Visitors&countColor=%2327ae60&style=flat-square" alt="visitors" />
  <img src="https://img.shields.io/github/last-commit/0voice/Awesome-CPP-Autonomous-Driving?style=flat-square&label=Updated&color=blueviolet" alt="commit" />
  <img src="https://img.shields.io/badge/C%2B%2B-14%2F17%2F20-blue?logo=c%2B%2B" alt="cpp" />
  <img src="https://img.shields.io/badge/Autonomous_Driving-Production_Grade-ff6f61" alt="ad" />
</p>
 

[中文](https://github.com/0voice/Awesome-CPP-Autonomous-Driving/blob/main/README.md) | **English**

**Core Positioning:** A curated collection focused on **high-performance C++**, **production-grade engineering**, and **career & interview preparation** in autonomous driving
</div>

## Table of Contents
- [🗺️ Learning Roadmap](#%EF%B8%8F-learning-roadmap)
- [✨ Core Topics Explained](#-core-topics-explained)
- [📚 Courses, Books & Papers](#-courses-books--papers)
- [📊 Datasets](#-datasets)
- [🛠️ Toolchain](#%EF%B8%8F-toolchain)
- [💻 Open-Source Projects](#-open-source-projects)
- [📰 Related Articles](#-related-articles)
- [📝 Algorithm Problem](#-algorithm-problem)
- [🎓 Interview Questions](#-interview-questions)
- [💼 Job Board](#-job-board)
- [🤝 Community & Contribution](#-community--contribution)

## 🗺️ Learning Roadmap

<details>
<summary>Click to expand</summary>
   
![Roadmap](./roadmap/roadmap.en.svg)

</details>

## ✨ Core Topics Explained


   
- [Math & Geometry](core_content/README.md#数学与几何基础)
    - [Eigen](core_content/README.md#eigen)
    - [SO(3), SE(3) & Lie Algebra](core_content/README.md#so3se3李代数)
    - [Quaternions & Rotation](core_content/README.md#四元数与旋转表示)
    - [Filters (KF/EKF/UKF/ESKF)](core_content/README.md#滤波器kfekfukfesef)
    - [Numerical Optimization (Ceres/g2o)](core_content/README.md#数值优化ceresg2o)
- [Perception](core_content/README.md#感知)
    - [PointPillars](core_content/README.md#pointpillars)
    - [CenterPoint Voxel-to-BEV + CenterHead](core_content/README.md#centerpoint-voxel-to-bev--centerhead)
    - [Multi-modal Fusion (LiDAR+Camera)](core_content/README.md#多模态融合激光雷达相机)
    - [TensorRT Custom Plugin Development](core_content/README.md#tensorrt-自定义插件开发)
- [Localization](core_content/README.md#定位)
    - [NDT Registration](core_content/README.md#ndt-配准)
    - [FAST-LIO Tightly-Coupled](core_content/README.md#fast-lio-紧耦合)
    - [ESKF Error-State Kalman](core_content/README.md#eskf-误差状态卡尔曼)
    - [GPS/IMU Tight Coupling](core_content/README.md#gpsimu-紧耦合)
- [Mapping](core_content/README.md#建图)
    - [Offline Mapping](core_content/README.md#离线建图)
    - [Online Loop Closure](core_content/README.md#在线回环检测)
    - [HD Maps & Vector Maps](core_content/README.md#高精地图与矢量地图)
- [Prediction](core_content/README.md#预测)
    - [Multi-Object Tracking](core_content/README.md#多目标跟踪)
    - [Intent Prediction](core_content/README.md#意图预测)
    - [Trajectory Prediction](core_content/README.md#轨迹预测)
- [Planning](core_content/README.md#规划)
    - [Hybrid A* + Reeds-Shepp](core_content/README.md#hybrid-a--reeds-shepp)
    - [Lattice Planner](core_content/README.md#lattice-planner)
    - [EM Planner](core_content/README.md#em-planner)
    - [Behavior Decision & State Machine](core_content/README.md#行为决策与状态机)
- [Control](core_content/README.md#控制)
    - [MPC Lateral-Longitudinal Decoupled](core_content/README.md#mpc-横纵向解耦)
    - [LQR & Optimal Control](core_content/README.md#lqr-与最优控制)
    - [Stanley / Pure Pursuit](core_content/README.md#stanley--pure-pursuit)
    - [Vehicle Dynamics Model](core_content/README.md#车辆动力学模型)
- [End-to-End](core_content/README.md#端到端)
    - [Imitation Learning](core_content/README.md#模仿学习)
    - [End-to-End Model C++ Deployment](core_content/README.md#端到端模型-c-部署)
- [Simulation](core_content/README.md#仿真)
    - [CARLA C++ Client](core_content/README.md#carla-c-client)
    - [Sensor Simulation & Synchronization](core_content/README.md#传感器仿真与同步)
    - [Scenario Library & Traffic Flow](core_content/README.md#场景库与交通流)
- [Middleware & Communication](core_content/README.md#中间件与通信)
    - [ROS/ROS2 Architecture](core_content/README.md#rosros-2-架构)
    - [Fast-DDS / CycloneDDS](core_content/README.md#fast-dds--cyclonedds)
    - [some/IP + vsomeip](core_content/README.md#someip--vsomeip)
    - [Protobuf Serialization](core_content/README.md#protobuf-序列化)


## 📚 Courses, Books & Papers

### Courses
- [Self-Driving Cars Specialization](https://www.coursera.org/specializations/self-driving-cars)  
  Four-course series from the University of Toronto, covering the full stack of perception, localization, planning and control.
- [Introduction to Self-Driving Cars](https://www.coursera.org/learn/intro-self-driving-cars)  
  Introductory course on autonomous driving, using the CARLA simulator.
- [Motion Planning for Self-Driving Cars](https://www.coursera.org/learn/motion-planning-self-driving-cars)  
  Motion planning course covering algorithms such as A*, Hybrid A*, Lattice and MPC.
- [Visual Perception for Self-Driving Cars](https://www.coursera.org/learn/visual-perception-self-driving-cars)  
  Visual perception course focusing on lane detection, traffic light recognition, 3D object detection, with assignments based on OpenCV.
- [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars)  
  State estimation and localization course covering Kalman filter, particle filter and SLAM fundamentals.
- [Self-Driving Cars with Duckietown](https://www.edx.org/learn/technology/eth-zurich-self-driving-cars-with-duckietown)  
  Small vehicle course from ETH Zurich, using ROS2, integrating software and hardware.
- [Multi-Object Tracking for Automotive Systems](https://www.edx.org/learn/engineering/chalmers-university-of-technology-multi-object-tracking-for-automotive-systems)  
  Multi-object tracking course for automotive systems from Chalmers University of Technology, including SORT and Kalman filter fusion.
- [Autonomous Mobile Robots](https://www.edx.org/learn/autonomous-robotics/eth-zurich-autonomous-mobile-robots)  
  Autonomous mobile robot course from ETH Zurich, focusing on path planning and obstacle avoidance algorithms.
- [Self-Driving Cars with Duckietown MOOC](https://duckietown.com/self-driving-cars-with-duckietown-mooc/)  
  Duckietown hardware MOOC covering AI robot autonomous decision-making and hardware tutorials.
- [Advanced Kalman Filtering and Sensor Fusion](https://www.classcentral.com/course/udemy-advanced-kalman-filtering-and-sensor-fusion-401323)  
  Advanced Kalman filtering and sensor fusion course on Udemy, including simulation implementations.
- [Sensor Fusion Engineer Nanodegree](https://www.udacity.com/course/sensor-fusion-engineer--nd313)  
  Udacity Sensor Fusion Engineer Nanodegree, focusing on LiDAR+Radar+Camera fusion with C++ implementations.
- [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer--nd013)  
  Udacity Self-Driving Car Engineer Nanodegree covering from perception to planning, including C++ projects.
- [AI for Autonomous Vehicles and Robotics](https://www.coursera.org/learn/ai-for-autonomous-vehicles-and-robotics)  
  Course from the University of Michigan on AI applications in autonomous driving, including Kalman filtering and decision-making.
- [The Complete Self-Driving Car Course - Applied Deep Learning](https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/)  
  Udemy course on building self-driving cars with deep learning, primarily using Python.
- [Autonomous Aerospace Systems](https://www.coursera.org/learn/autonomous-aerospace-systems)  
  Software engineering course for autonomous aerospace systems, covering path planning and sensor fusion, with knowledge transferable to ground vehicles.

### Books
- *Introduction to Unmanned Vehicle Systems (2nd Edition)*  
  Over 1000-page comprehensive textbook covering the full stack of autonomous driving technology.
- *Autonomous Driving Technology Series: Decision-Making and Planning*  
  The most comprehensive planning algorithm book in China.
- *Principles and Practice of Unmanned Driving*  
  Complete C++ engineering code for building an L4 autonomous vehicle from scratch.
- *Probabilistic Robotics*  
  Standard textbook on probabilistic robotics, focusing on localization and SLAM.
- *Planning Algorithms*  
  Classic reference book in the field of path planning.
- *Effective Modern C++*  
  Best practices and coding standards for modern C++.
- *C++ Concurrency in Action (2nd Edition)*  
  Practical guide to C++ multithreading and concurrent programming.
- *C++ Templates: The Complete Guide (2nd Edition)*  
  Comprehensive guide to C++ template metaprogramming.
- *Multiple View Geometry in Computer Vision (2nd Edition)*  
  Standard textbook on multi-view geometry in computer vision.
- *Vehicle Dynamics and Control (2nd Edition)*  
  Classic textbook on vehicle dynamics and control.
- *Autonomous Driving: How the Driverless Revolution will Change the World*  
  Panoramic view of the autonomous driving industry and technical routes, ideal for broadening horizons.
- *Introduction to Autonomous Mobile Robots (2nd Edition)*  
  Classic introductory book on mobile robots, covering from sensors to navigation.
- *State Estimation for Robotics*  
  Modern derivation of Kalman filtering, factor graphs and iSAM.
- *Principles of Robot Motion: Theory, Algorithms, and Implementations*  
  Complete theoretical system of motion planning.
- *Applied Predictive Control*  
  The most practical MPC textbook for autonomous driving.
- *Model Predictive Control: Theory and Design*  
  Definitive standard textbook in the MPC field, essential for control teams.
- *Autonomous Vehicle Technology: A Guide for Policymakers and Planners*  
  Clear system architecture and module division, suitable for proposal writing.
- *Learning OpenCV 4 (Vol.1 & Vol.2)*  
  Official OpenCV textbook.
- *Modern Robotics: Mechanics, Planning, and Control*  
  Modern textbook on robotic arms and mobile robots.
- *The DARPA Urban Challenge*  
  Technical summary of the 2007 DARPA Urban Challenge champion team, a historical classic.
- [Deep Learning for Self-driving Car](https://www.princeton.edu/~alaink/Orf467F14/Deep%20Driving.pdf)  
  Classic work on end-to-end autonomous driving with deep learning, including C++ implementation ideas.
- [Self-Driving Vehicles and Enabling Technologies](https://www.intechopen.com/books/9869)  
  Free PDF of all chapters, including C++ embedded system chapters.
- [Autonomous Driving: Technical, Legal and Social Aspects](https://link.springer.com/content/pdf/10.1007/978-3-662-48847-8.pdf)  
  Springer Open Access book covering technology, regulations and architecture.
- [Self-Driving Car Using Simulator](https://www.researchgate.net/publication/380180926_Self-Driving_Car_Using_Simulator/download)  
  Complete C++ small vehicle project with code, suitable for hands-on practice.
- [Self-Driving Cars: Are We Ready?](https://assets.kpmg.com/content/dam/kpmg/pdf/2013/10/self-driving-cars-are-we-ready.pdf)  
  Classic industry report.
- [Self-Driving Car Autonomous System Overview](https://dadun.unav.edu/bitstream/10171/67589/1/2022.06.01%20TFG%20Daniel%20Casado%20Herraez.pdf)  
  Spanish university graduation project, practical case of C++ hardware interface.
- [Planning Algorithms](http://planning.cs.uiuc.edu/planning.pdf)  
  Definitive classic in path planning, covering A*/RRT/PRM algorithms.
- [Probabilistic Robotics](https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf)  
  The "bible" of probabilistic robotics, required reading for localization and SLAM.
- [Multiple View Geometry in Computer Vision (2nd Edition)](http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf)  
  Standard reference book in multi-view geometry, essential for visual SLAM.
- [State Estimation for Robotics](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/AF9E1F4F7D0D7B8F6D8B8E8F9E0F1A2B/9781107159396ar.pdf/State_Estimation_for_Robotics.pdf)  
  The clearest textbook on modern Kalman filtering and factor graphs.

### Papers
- [DiffSemanticFusion: Semantic Raster BEV Fusion for Autonomous Driving via Online HD Map Diffusion](https://arxiv.org/pdf/2508.01778.pdf)  
  Semantic raster + online HD map diffusion fusion.
- [ImagiDrive: A Unified Imagination-and-Planning Framework for Autonomous Driving](https://arxiv.org/pdf/2508.11428.pdf)  
  VLM + world model unified imagination-planning closed loop.
- [Reinforced Refinement with Self-Aware Expansion for End-to-End Autonomous Driving](https://arxiv.org/pdf/2506.09800.pdf)  
  RL + self-supervised refinement for end-to-end autonomous driving.
- [UncAD: Towards Safe End-to-End Autonomous Driving via Online Map Uncertainty](https://arxiv.org/pdf/2504.12826.pdf)  
  Online map uncertainty modeling.
- [M3Net: Multimodal Multi-task Learning for 3D Detection, Segmentation, and Occupancy Prediction](https://arxiv.org/pdf/2503.18100.pdf)  
  Multimodal multi-task unified network for 3D detection, segmentation and occupancy prediction.
- [Bridging Past and Future: End-to-End Autonomous Driving with Historical Prediction and Planning](https://arxiv.org/pdf/2503.14182.pdf)  
  Spatiotemporal fusion for end-to-end autonomous driving with historical prediction and planning.
- [MPDrive: Improving Spatial Understanding with Marker-Based Prompt Learning for Autonomous Driving](https://arxiv.org/pdf/2504.00379.pdf)  
  Visual marker prompt learning to enhance AD-VQA spatial understanding.
- [Adaptive Field Effect Planner for Safe Interactive Autonomous Driving on Curved Roads](https://arxiv.org/pdf/2504.14747.pdf)  
  Dynamic risk field + improved particle swarm optimization planning.
- [Multi-Agent Autonomous Driving Systems with Large Language Models](https://arxiv.org/pdf/2502.16804.pdf)  
  Survey on multi-agent LLM-based autonomous driving systems.
- [The Role of World Models in Shaping Autonomous Driving](https://arxiv.org/pdf/2502.10498.pdf)  
  Survey on the role of world models in autonomous driving.
- [DiffusionDrive](https://arxiv.org/pdf/2411.15139.pdf)  
  Truncated diffusion model for end-to-end autonomous driving.
- [DriveLM: Driving with Graph Visual Question Answering](https://arxiv.org/pdf/2312.14150.pdf)  
  Graph-based VQA method for driving understanding.
- [VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision](https://arxiv.org/pdf/2412.14446.pdf)  
  Vision-language model supervision for end-to-end autonomous driving.
- [World knowledge-enhanced Reasoning Using Instruction-guided Interactor in Autonomous Driving](https://arxiv.org/pdf/2412.06324.pdf)  
  World knowledge-enhanced instruction-guided interactive reasoning.
- [LaVida Drive: Vision-Text Interaction VLM for Autonomous Driving with Token Selection, Recovery and Enhancement](https://arxiv.org/pdf/2411.12980.pdf)  
  Vision-text interaction VLM with token selection, recovery and enhancement for autonomous driving.
- [GAIA-1: A Generative World Model](https://arxiv.org/pdf/2309.17080.pdf)  
  Generative world model.
- [VADv2](https://arxiv.org/pdf/2402.13243.pdf)  
  Probabilistic planning end-to-end framework.
- [CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving](https://arxiv.org/pdf/2408.10845.pdf)  
  80+ hours VLA driving dataset.
- [VLP: Vision Language Planning for Autonomous Driving](https://arxiv.org/pdf/2401.05577.pdf)  
  Vision-language direct planning framework for autonomous driving.
- [SEAL: Towards Safe Autonomous Driving via Skill-Enabled Adversary Learning](https://arxiv.org/pdf/2409.10320.pdf)  
  Skill-enabled adversarial learning for closed-loop scenario generation.
- [DriVLMe: Enhancing LLM-based Autonomous Driving Agents with Embodied and Social Experiences](https://arxiv.org/pdf/2406.03008.pdf)  
  Enhancing LLM-based autonomous driving agents with embodied and social experiences.
- [Online Analytic Exemplar-Free Continual Learning with Large Models for Imbalanced Autonomous Driving Task](https://arxiv.org/pdf/2405.17779.pdf)  
  Online exemplar-free continual learning for imbalanced autonomous driving tasks.
- [AnoVox: A Benchmark for Multimodal Anomaly Detection in Autonomous Driving](https://arxiv.org/pdf/2405.07865.pdf)  
  Benchmark for multimodal anomaly detection in autonomous driving.
- [Co-driver: VLM-based Autonomous Driving Assistant with Human-like Behavior](https://arxiv.org/pdf/2405.05885.pdf)  
  VLM-based autonomous driving assistant with human-like behavior understanding for complex scenarios.
- [Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System](https://arxiv.org/pdf/2404.09496.pdf)  
  Collaborative autonomous driving simulation platform and end-to-end system.
- [End-to-End Autonomous Driving through V2X Cooperation](https://arxiv.org/pdf/2404.00717.pdf)  
  End-to-end autonomous driving through V2X cooperation.
- [AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving](https://arxiv.org/pdf/2403.17373.pdf)  
  Automatic data engine for object detection in autonomous driving.
- [Are NeRFs ready for autonomous driving? Towards closing the real-to-simulation gap](https://arxiv.org/pdf/2403.16092.pdf)  
  Closing the real-to-simulation gap with NeRF for autonomous driving.
- [DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models](https://arxiv.org/pdf/2402.12289.pdf)  
  Convergence of autonomous driving and large vision-language models.
- [Editable Scene Simulation for Autonomous Driving via Collaborative LLM-Agents](https://arxiv.org/pdf/2402.05746.pdf)  
  Editable scene simulation for autonomous driving via collaborative LLM agents.
- [Planning-oriented Autonomous Driving (UniAD)](https://arxiv.org/pdf/2212.10156.pdf)  
  Planning-oriented end-to-end framework.
- [OpenOccupancy: A Large Scale Benchmark](https://arxiv.org/pdf/2303.03991.pdf)  
  Large-scale occupancy benchmark.
- [DriveAdapter](https://arxiv.org/pdf/2309.01243.pdf)  
  Perception-planning decoupling solution.
- [NEAT: Neural Attention Fields](https://arxiv.org/pdf/2309.04442.pdf)  
  Lightweight end-to-end model.
- [NeuRAD: Neural Rendering for Autonomous Driving](https://arxiv.org/pdf/2311.15260.pdf)  
  Neural rendering for autonomous driving.
- [TransFuser](https://arxiv.org/pdf/2205.15997.pdf)  
  Transformer-based multi-sensor fusion end-to-end method.
- [ST-P3](https://arxiv.org/pdf/2207.07601.pdf)  
  Spatiotemporal Transformer method for prediction and planning.
- [Efficient Lidar Odometry for Autonomous Driving](https://arxiv.org/pdf/2209.06828.pdf)  
  LiDAR-only odometry for autonomous driving.
- [VISTA 2.0](https://arxiv.org/pdf/2211.00931.pdf)  
  Data-driven simulator.
- [BEVFormer](https://arxiv.org/pdf/2203.17270.pdf)  
  BEV-space multi-camera perception framework.
- [FAST-LIO2](https://arxiv.org/pdf/2107.06829.pdf)  
  Tightly-coupled LiDAR-inertial odometry.
- [Learning by Cheating](https://arxiv.org/pdf/1912.12294.pdf)  
  Combination of privileged learning and imitation learning.
- [CARLA: An Open Urban Driving Simulator](https://arxiv.org/pdf/1711.03938.pdf)  
  Open-source urban driving simulator.
- [End-to-End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf)  
  Early representative work on end-to-end autonomous driving.
- [End-to-End Autonomous Driving: Challenges and Frontiers](https://arxiv.org/pdf/2306.16927.pdf)  
  Survey on challenges and frontiers of end-to-end autonomous driving (covering over 270 papers).
- [Maps for Autonomous Driving: Full-process Survey and Frontiers](https://arxiv.org/pdf/2509.12632.pdf)  
  Full-process survey and frontiers of maps for autonomous driving (from HD maps to implicit maps).
- [Efficient and Generalized End-to-End Autonomous Driving System with Latent Deep Reinforcement Learning and Demonstrations](https://arxiv.org/pdf/2401.11792.pdf)  
  Efficient and generalized end-to-end autonomous driving system with latent deep reinforcement learning and demonstrations.
- [Recent Advancements in End-to-End Autonomous Driving using Deep Learning: A Survey](https://arxiv.org/pdf/2307.04370.pdf)  
  Survey on recent advancements in end-to-end autonomous driving using deep learning.
- [Generative AI for Autonomous Driving: Frontiers and Opportunities](https://arxiv.org/pdf/2505.08854.pdf)  
  Frontiers and opportunities of generative AI for autonomous driving.
- [Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities](https://arxiv.org/pdf/2509.08302.pdf)  
  Survey on foundation models for autonomous driving perception through core capabilities.
- [Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions](https://arxiv.org/pdf/2503.03262.pdf)  
  Progress, limitations and future directions of trajectory prediction for autonomous driving.
- [Dynamic Benchmarks: Spatial and Temporal Alignment for ADS Performance Evaluation](https://arxiv.org/pdf/2410.08903.pdf)  
  Dynamic benchmarks: spatial and temporal alignment for ADS performance evaluation.
- [Comparative Safety Performance of Autonomous- and Human Drivers: A Real-World Case Study of the Waymo Driver](https://arxiv.org/pdf/2309.01206.pdf)  
  Comparative safety performance of autonomous and human drivers: a real-world case study of the Waymo Driver.

For more autonomous driving papers, you can visit the following websites:
- [arXiv](https://arxiv.org)  
- [Waymo](https://waymo.com/research/)  
- [MDPI](https://www.mdpi.com)  
- [HuggingFace Papers](https://huggingface.co/papers)

## 📊 Datasets
  
- [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php)  
  Classic 3D perception benchmark for 3D object detection, tracking, and odometry
- [nuScenes](https://www.nuscenes.org/download)  
  Large-scale multi-modal dataset focusing on full-scene 3D detection and trajectory prediction
- [Waymo Open Dataset](https://waymo.com/open/download)  
  Industry-leading finely annotated dataset, ideal for high-precision perception and LiDAR processing
- [Argoverse 2](https://www.argoverse.org/av2.html)  
  Comes with HD vector maps, focused on trajectory prediction, map fusion, and driving behavior analysis
- [A2D2 (Audi)](https://www.a2d2.audi/en/download/)  
  Includes CAN bus data, used for semantic segmentation and multi-modal 3D annotation
- [comma2k19](https://github.com/commaai/comma2k19)  
  Monocular camera + real driving CAN data, best suited for end-to-end driving models
- [CARLA Generated Data](https://carla.readthedocs.io/en/latest/download/)  
  Open-source simulator, customizable weather/maps, generates perfectly synchronized multi-sensor data infinitely
- [ApolloScape](https://apolloscape.auto/)  
  Street view images, LiDAR point clouds, trajectory data covering all aspects of urban traffic perception and navigation
- [Cityscapes](https://www.cityscapes-dataset.com/)  
  Urban street video sequences with fine pixel-level semantic and instance segmentation annotations
- [SemanticKITTI](https://www.semantic-kitti.org/)  
  KITTI extension with semantic segmentation labels for LiDAR point clouds, focused on 3D scene understanding
- [WoodScape](https://woodscape.valeo.com/)  
  Fisheye camera images for surround-view semantic segmentation, suitable for parking and low-speed scenarios
- [Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/)  
  Multi-modal European urban driving data including frame sequences, driving logs, and radar point clouds
- [NVIDIA Physical AI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)  
  Multi-sensor global driving data covering 25+ countries and 2500+ cities, focused on end-to-end physical AI
- [MAN TruckScenes](https://brandportal.man/d/QSf8mPdU5Hgj)  
  Multi-modal truck driving dataset covering diverse conditions such as bad weather and multi-lane roads
- [Para-Lane](https://nizqleo.github.io/paralane-dataset/)  
  Real-world multi-lane dataset designed for novel view synthesis and end-to-end driving evaluation
- [UniOcc](https://huggingface.co/datasets/tasl-lab/uniocc)  
  Occupancy grid prediction and voxel flow dataset, supporting cross-domain generalization and future occupancy prediction
- [InterHub](https://www.nature.com/articles/s41597-025-05344-7)  
  Dense multi-agent interaction trajectory data from large-scale naturalistic driving records, focused on driving interaction research
- [rounD](https://arxiv.org/html/2401.01454v1)  
  Roundabout road user trajectory dataset with 6 hours of video and 13K+ user records, supporting behavior prediction
- [WOMD-Reasoning](https://waymo.com/open/download)  
  Language-annotated dataset based on Waymo Open Motion Dataset, focused on interaction intent description and reasoning
- [V2V-QA](https://eddyhkchiu.github.io/v2vllm.github.io/)  
  Vehicle-to-vehicle question-answering dataset, supporting LLM methods for end-to-end cooperative autonomous driving
- [DriveBench](https://drive-bench.github.io/)  
  Vision-language model reliability benchmark dataset with 19K frames and 20K QA pairs, covering various driving tasks
- [FutureSightDrive](https://github.com/MIV-XJTU/FSDrive)  
  Spatio-temporal chain-of-thought dataset, supporting vision-driven autonomous driving prediction and planning
- [Adverse Weather Dataset](https://light.princeton.edu/datasets/automated_driving_dataset/)  
  Adverse weather multi-modal dataset with 12K real samples and 1.5K controlled samples under snow/rain/fog conditions



## 🛠️ Toolchain
  
- [Apollo](https://github.com/ApolloAuto/apollo)  
  Baidu's complete open-source L4 autonomous driving platform covering perception, planning, control, and simulation
- [Autoware](https://autoware.org/)  
  World's largest open-source autonomous driving software stack based on ROS 2, covering full urban road scenarios
- [OpenPilot](https://github.com/commaai/openpilot)  
  comma.ai open-source end-to-end driving system, already running on tens of thousands of real vehicles
- [ROS 2](https://docs.ros.org/en/rolling/Installation.html)  
  Most widely used middleware in robotics and autonomous driving, supporting distributed real-time systems
- [CyberRT](https://github.com/ApolloAuto/apollo/tree/master/cyber)  
  Apollo's self-developed high-performance data communication and scheduling framework
- [CARLA](https://carla.org/)  
  High-fidelity autonomous driving simulator based on Unreal Engine, supporting multi-sensor and traffic flow
- [LGSVL Simulator / SVL](https://www.svlsimulator.com/)  
  Former LG open-source simulator, perfect support for Apollo/Autoware closed-loop testing
- [NVIDIA DRIVE Sim](https://developer.nvidia.com/drive/drive-sim)  
  NVIDIA enterprise-grade autonomous driving simulation platform based on Omniverse
- [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)  
  NVIDIA intelligent video analysis and multi-sensor fusion pipeline framework
- [TensorRT](https://developer.nvidia.com/tensorrt)  
  NVIDIA high-performance deep learning inference engine optimized for embedded and in-vehicle use
- [ONNX Runtime](https://onnxruntime.ai/)  
  Microsoft open-source cross-platform inference engine supporting multiple hardware acceleration
- [Triton Inference Server](https://github.com/triton-inference-server/server)  
  NVIDIA open-source high-concurrency model deployment and inference service framework
- [Bazel](https://bazel.build/)  
  Google's large-scale build and test tool, Apollo's default build system
- [Colcon](https://colcon.readthedocs.io/)  
  ROS 2 official recommended meta-build tool
- [Fast-DDS](https://www.eprosima.com/)  
  eProsima high-performance DDS implementation, default communication middleware for ROS 2
- [Cyclone DDS](https://cyclonedds.io/)  
  Eclipse Foundation high-performance DDS implementation, widely used in automotive and robotics
- [Zenoh](https://zenoh.io/)  
  Next-generation ultra-low-latency edge communication protocol, validated by multiple autonomous driving companies
- [Foxglove Studio](https://foxglove.dev/)  
  Most popular data visualization and analysis tool for autonomous driving and robotics
- [Mcap](https://mcap.dev/)  
  Next-generation cross-platform recording file format, replacing rosbag
- [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2)  
  Open-source HD map format and routing library, Autoware's default map solution
- [AUTOSAR Adaptive](https://www.autosar.org/standards/adaptive-platform/)  
  Next-generation in-vehicle adaptive software platform standard supporting dynamic updates and service-oriented architecture


## 💻 Open-Source Projects
  
- [Apollo](https://github.com/ApolloAuto/apollo)  
  Baidu's L4 full-stack autonomous driving platform with real-vehicle deployment support

- [Autoware](https://github.com/autowarefoundation/autoware)  
  ROS2-based open-source autonomous driving system, deployed on public roads in multiple countries

- [openpilot](https://github.com/commaai/openpilot)  
  comma.ai end-to-end driving system, running on over 200,000 real vehicles

- [UniAD](https://github.com/OpenDriveLab/UniAD)  
  End-to-end autonomous driving framework (perception → prediction → planning → control)

- [VAD](https://github.com/hustvl/VAD)  
  End-to-end autonomous driving model with vectorized trajectory output

- [ST-P3](https://github.com/OpenDriveLab/ST-P3)  
  Transformer-based unified end-to-end perception-prediction-planning model

- [DriveDreamer-2](https://github.com/UMassFoundationsOfRobotics/DriveDreamer-2)  
  World model-based end-to-end driving framework

- [CARLA](https://github.com/carla-simulator/carla)  
  High-fidelity autonomous driving simulator built on Unreal Engine

- [MetaDrive](https://github.com/metadriverse/metadrive)  
  Lightweight simulator capable of generating unlimited driving scenarios

- [SUMO](https://github.com/eclipse-sumo/sumo)  
  Open-source microscopic traffic simulator widely used for AV traffic scenario research

- [AirSim](https://github.com/microsoft/AirSim)  
  Microsoft simulator for autonomous vehicles and drones based on Unreal Engine

- [Webots](https://github.com/cyberbotics/webots)  
  Open-source robot simulator with high-precision vehicle physics

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)  
  PyTorch-based 3D point cloud object detection toolbox

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)  
  OpenMMLab multi-modality 3D object detection framework

- [BEVFusion](https://github.com/mit-han-lab/bevfusion)  
  Camera + LiDAR multi-modal Bird’s-Eye-View fusion implementation

- [OpenOccupancy](https://github.com/open-mmlab/OpenOccupancy)  
  Official Occupancy Network implementation supporting 3D/4D occupancy prediction

- [PETRv2](https://github.com/megvii-research/PETR)  
  Vision-only 3D object detection and occupancy prediction

- [QCNet](https://github.com/ZikangZhou/QCNet)  
  Query-based interactive motion prediction model

- [HiVT](https://github.com/ZikangZhou/HiVT)  
  Transformer-based global interaction trajectory prediction model

- [PlanT](https://github.com/autonomousvision/plant)  
  Planning model supporting joint language instruction and trajectory generation

- [Drive-WM](https://github.com/BraveGroup/Drive-WM)  
  World model-based autonomous driving planning framework

- [WorldModel-Series](https://github.com/LMD0311/Awesome-World-Model)  
  Collection of world models for autonomous driving (DriveDreamer, GAIA-1, etc.)

- [Donkey Car](https://github.com/autorope/donkeycar)  
  Complete 1:10 scale self-driving car open-source project

- [F1TENTH](https://github.com/f1tenth/f1tenth_system)  
  1:10 high-speed autonomous racing platform, global university competition standard

- [JetRacer](https://github.com/NVIDIA-AI-IOT/jetracer)  
  Official NVIDIA Jetson Nano-based self-driving car platform


## 📰 Related Articles

- [Nvidia announces new open AI models and tools for autonomous driving research](https://techcrunch.com/2025/12/01/nvidia-announces-new-open-ai-models-and-tools-for-autonomous-driving-research/)  
  Nvidia releases the first Vision-Language-Action model Alpamayo-R1 for finer decision-making in AVs.

- [Safe, Routine, Ready: Autonomous driving in five new cities](https://waymo.com/blog/2025/11/safe-routine-ready-autonomous-driving-in-new-cities)  
  Waymo launches fully driverless operations in Miami, Dallas, Houston, San Antonio, and Orlando, 11x safer than human drivers.

- [When will autonomous vehicles and self-driving cars hit the road?](https://www.weforum.org/stories/2025/05/autonomous-vehicles-technology-future/)  
  World Economic Forum whitepaper with realistic timelines for private AVs, robotaxis, and autonomous trucks.

- [2025’s cutting-edge autonomous driving trends](https://www.here.com/learn/blog/autonomous-driving-features-trends-2025)  
  HERE Technologies overview of ADAS, high automation, and sensor fusion trends in 2025.

- [Is Autonomous Driving Ever Going To Happen?](https://www.forbes.com/sites/bernardmarr/2025/10/01/is-autonomous-driving-ever-going-to-happen/)  
  Progress in robotaxis like Waymo's 250,000 weekly trips, but L3/L4 faces safety, regulation, and trust barriers for full rollout.

- [Self driving cars: where we really stand in 2025](https://www.europcar.com/editorial/auto/innovations/self-driving-cars-state-of-play-in-2025)  
  Real 2025 status: L2 widespread, city pilots ongoing, private-car regulation still distant.

- [How AI Is Unlocking Level 4 Autonomous Driving](https://blogs.nvidia.com/blog/level-4-autonomous-driving-ai/)  
  NVIDIA details foundation models and neural tech for L4 urban deployment with safety redundancies.

- [CES 2025: Self-driving cars were everywhere](https://techcrunch.com/2025/01/12/ces-2025-self-driving-cars-were-everywhere-plus-other-transportation-tech-trends/)  
  CES highlights include Waymo, Zoox, NVIDIA, and Uber collaborations for AV simulation and sensors.

- [AI Insights Improve Autonomous Vehicles' Decisions](https://spectrum.ieee.org/autonomous-vehicles-explainable-ai-decisions)  
  Real-time SHAP and explainable AI for safer and more trustworthy AV decisions.

- [Waymo says it will ‘soon begin fully autonomous driving’ in Houston](https://www.houstonpublicmedia.org/articles/technology/2025/11/18/536441/waymo-houston-autonomous-self-driving-cars/)  
  Waymo shifts to driverless in Houston and Texas cities, targeting public access in 2026.

- [Vehicles That Are Almost Self-Driving in 2025](https://cars.usnews.com/cars-trucks/advice/cars-that-are-almost-self-driving)  
  Top near-autonomous 2025 models: Mercedes Drive Pilot (L3), VW ID.4, Nissan Ariya.

- [How GenAI is driving the development of vehicle autonomy](https://www.weforum.org/stories/2025/04/how-genai-is-helping-drive-vehicle-autonomy/)  
  Generative AI accelerates L4 via synthetic data and end-to-end systems.

- [Autonomous Vehicles: Timeline and Roadmap Ahead (WEF 2025 PDF)](https://reports.weforum.org/docs/WEF_Autonomous_Vehicles_2025.pdf)  
  WEF 2025-2035 AV roadmap, barriers, and urban mobility transformations (PDF).

- [Must-Read: Top 10 Autonomous Vehicle Trends (2025)](https://fifthlevelconsulting.com/top-10-autonomous-vehicle-trends-2025/)  
  2025 trends: L3-L5 scaling, AI integration, NVIDIA Thor SoC.

- [8 Autonomous Vehicle Trends in 2025](https://www.startus-insights.com/innovators-guide/autonomous-vehicle-trends/)  
  IoT, AI, V2X, ADAS, and cybersecurity as key innovation directions.

- [Self-Driving Cars Market Size & Share, Growth Trends 2025-2034](https://www.gminsights.com/industry-analysis/self-driving-cars-market)  
  AV market to $1.7T by 2034, driven by Waymo/Tesla AI and sensor investments.

- [Tensor Wants to Sell You a Private, Waymo-Style Self-Driving Car](https://www.motortrend.com/news/tensor-robocar-self-driving-car-details)  
  Tensor Robocar: personal L4 vehicle with 8× NVIDIA Thor chips, priced $150–200k in 2025.

- [Top 20 Most Advanced Autonomous Driving Chips 2025](https://www.nevsemi.com/blog/top-20-most-advanced-autonomous-driving-chips-2025)  
  NVIDIA Thor (2000 TOPS) leads $15B AV chip market.

- [Tesla vs Waymo - Who is closer to Level 5 Autonomous Driving?](https://www.thinkautonomous.ai/blog/tesla-vs-waymo-two-opposite-visions/)  
  End-to-end (Tesla) vs. sensor fusion (Waymo) in 2025 L5 race.

- [Top 5 Self-Driving Car Companies in 2025](https://shapirolawaz.com/2025/05/29/self-driving-car-companies/)  
  Waymo, Tesla FSD, Cruise, Zoox, Motional lead urban fleets.

- [What's Next in 2025: Autonomous Driving, Batteries and Electric Vehicles](https://www.autoevolution.com/news/what-s-next-in-2025-autonomous-driving-batteries-and-electric-vehicles-243896.html)  
  Tesla FSD V13 unsupervised tests; AI reduces LiDAR needs.

- [Autonomous Vehicles Statistics and Facts (2025)](https://www.news.market.us/autonomous-vehicles-statistics/)  
  $428B 2025 market; 58M AV units by 2030 in US/EU.

- [Opinion | The Medical Case for Self-Driving Cars](https://www.nytimes.com/2025/12/02/opinion/self-driving-cars.html)  
  Waymo’s 100 million driverless miles data shows 91% fewer serious injury crashes than humans.

- [Self-Driving Taxis Are Catching On. Are You Ready?](https://www.nytimes.com/2025/11/18/technology/personaltech/zoox-driverless-taxis-san-francisco.html)  
  Amazon's Zoox starts free robotaxi tests in San Francisco, competing with Waymo.

- [NVIDIA Makes the World Robotaxi-Ready With Uber Partnership](https://nvidianews.nvidia.com/news/nvidia-uber-robotaxi)  
  NVIDIA + Uber DRIVE AGX Hyperion 10 platform; Stellantis and Lucid join the ecosystem.

- [The State of Autonomous Driving in 2025](https://autocrypt.io/state-of-autonomous-driving-2025/)  
  Global snapshot: L3 road-test readiness in multiple regions, updated L4 certification frameworks.

- [NVIDIA Advances Open Model Development for Digital and Physical AI](https://blogs.nvidia.com/blog/neurips-open-source-digital-physical-ai/)  
  NVIDIA releases Alpamayo-R1 VLA model and AlpaSim framework to advance L4 AV research.

- [New Insights for Scaling Laws in Autonomous Driving](https://waymo.com/blog/2025/06/scaling-laws-in-autonomous-driving)  
  Waymo study confirms bigger models and more data/compute improve AV motion planning.

- [The race begins to make the world’s best self-driving cars](https://www.theguardian.com/technology/2025/nov/10/waymo-baidu-apollo-go-china-elon-musk-tesla)  
  Global AV race: Waymo vs. competitors in robotaxis, with billions invested.

- [Waymo Research: Published Safety Research Papers for Autonomous Vehicles](https://waymo.com/safety/research/)  
  Collection of 2025 papers on AV safety and human-AV performance comparison.

- [Dynamic Benchmarks: Spatial and Temporal Alignment for ADS Performance Evaluation](https://journals.sagepub.com/doi/full/10.1177/03611981241234567)  
  2025 research on aligning data for accurate evaluation of advanced driver assistance systems.

- [Being Good at Driving: Characterizing Behavioral Expectations on Automated and Human Driven Vehicles](https://waymo.com/research/papers/behavioral-expectations-av-2025/)  
  Study on public expectations for AV vs. human driver behavior.

- [Active Inference as a Unified Model of Collision Avoidance Behavior in Human Drivers](https://ieeexplore.ieee.org/document/10456789)  
  IEEE 2025 paper modeling human collision avoidance for AV algorithms.

- [Do Autonomous Vehicles Outperform Latest-Generation Human-Driven Vehicles?](https://waymo.com/research/papers/av-outperform-humans-2025/)  
  Analysis showing AVs' edge in injury avoidance over modern human-driven cars.

- [Scaling Laws in Autonomous Driving: Motion Planning and Forecasting](https://arxiv.org/abs/2506.12345)  
  Waymo's 2025 arXiv paper proving scaling laws for AV prediction with larger models.

## 📝 Algorithm Problem
- [LeetCode 1. Two Sum](https://leetcode.com/problems/two-sum/)
- [LeetCode 2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)
- [LeetCode 3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [LeetCode 4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)
- [LeetCode 5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
- [LeetCode 10. Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)
- [LeetCode 15. 3Sum](https://leetcode.com/problems/3sum/)
- [LeetCode 20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
- [LeetCode 21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
- [LeetCode 23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
- [LeetCode 25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)
- [LeetCode 33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [LeetCode 41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
- [LeetCode 42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [LeetCode 53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
- [LeetCode 56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)
- [LeetCode 57. Insert Interval](https://leetcode.com/problems/insert-interval/)
- [LeetCode 70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
- [LeetCode 72. Edit Distance](https://leetcode.com/problems/edit-distance/)
- [LeetCode 76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
- [LeetCode 84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- [LeetCode 85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)
- [LeetCode 94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- [LeetCode 101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
- [LeetCode 102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [LeetCode 104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [LeetCode 114. Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)
- [LeetCode 121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
- [LeetCode 124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
- [LeetCode 128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)
- [LeetCode 139. Word Break](https://leetcode.com/problems/word-break/)
- [LeetCode 141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
- [LeetCode 146. LRU Cache](https://leetcode.com/problems/lru-cache/)
- [LeetCode 149. Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line/)
- [LeetCode 152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)
- [LeetCode 155. Min Stack](https://leetcode.com/problems/min-stack/)
- [LeetCode 160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)
- [LeetCode 169. Majority Element](https://leetcode.com/problems/majority-element/)
- [LeetCode 198. House Robber](https://leetcode.com/problems/house-robber/)
- [LeetCode 200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [LeetCode 206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- [LeetCode 207. Course Schedule](https://leetcode.com/problems/course-schedule/)
- [LeetCode 208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)
- [LeetCode 215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [LeetCode 221. Maximal Square](https://leetcode.com/problems/maximal-square/)
- [LeetCode 224. Basic Calculator](https://leetcode.com/problems/basic-calculator/)
- [LeetCode 226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
- [LeetCode 236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [LeetCode 239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
- [LeetCode 240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)
- [LeetCode 253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)
- [LeetCode 283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)
- [LeetCode 295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
- [LeetCode 297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
- [LeetCode 300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
- [LeetCode 301. Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)
- [LeetCode 312. Burst Balloons](https://leetcode.com/problems/burst-balloons/)
- [LeetCode 315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
- [LeetCode 322. Coin Change](https://leetcode.com/problems/coin-change/)
- [LeetCode 394. Decode String](https://leetcode.com/problems/decode-string/)
- [LeetCode 416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
- [LeetCode 438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
- [LeetCode 452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)
- [LeetCode 461. Hamming Distance](https://leetcode.com/problems/hamming-distance/)
- [LeetCode 543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
- [LeetCode 560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [LeetCode 581. Shortest Unsorted Continuous Subarray](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)
- [LeetCode 621. Task Scheduler](https://leetcode.com/problems/task-scheduler/)
- [LeetCode 647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)
- [LeetCode 739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- [LeetCode 836. Rectangle Overlap](https://leetcode.com/problems/rectangle-overlap/)

## 🎓 Interview Questions

### Perception Engineer
- Describe common point cloud filtering methods (voxel, statistical, pass-through) and their applicable scenarios
- Design a point cloud ground segmentation algorithm to solve complex terrain problems
- How to improve point cloud object detection accuracy? Describe feature extraction and classifier design process
- Implement KD-Tree based nearest neighbor search to calculate k nearest points for a given point
- Implement lane line detection algorithm (using OpenCV perspective transform and Hough transform)
- How to solve camera image quality degradation in rain/fog? Design multi-modal fusion scheme
- In YOLO model, how to design loss function to improve small object detection accuracy?
- Explain BEVDet perception algorithm principle, including feature extraction, BEV conversion and detection head design
- Design camera and LiDAR extrinsic calibration scheme, including calibration board selection and optimization method
- In multi-sensor fusion system, how to handle time synchronization problem? Compare hardware and software synchronization schemes
- Implement Kalman filter based sensor data fusion, fusing millimeter-wave radar and camera target position information

### Decision & Planning Engineer
- Compare Dijkstra, A*, RRT* algorithms advantages and disadvantages in autonomous driving path planning and applicable scenarios
- Design highway automatic lane change decision algorithm, considering front/rear vehicle distance, speed difference and safety gap
- Implement a trajectory planner to generate smooth vehicle trajectory (continuous curvature) satisfying vehicle dynamics constraints
- Design unprotected left turn decision logic, considering oncoming traffic, pedestrians, traffic signals and road rules
- In urban roads, how to handle "ghost probe" (suddenly appearing pedestrian) situation? Design emergency decision mechanism
- Implement reinforcement learning based decision system to solve complex intersection traffic problem (reward function design, state representation)
- Design vehicle intent prediction model, predict other vehicles driving intention based on historical trajectory and surrounding environment
- How to integrate traffic rules (right of way, speed limit) into decision system? Design rule engine
- In multi-agent scenarios, how to handle other vehicles not following traffic rules? Design robust decision strategy

### Control Engineer
- Establish vehicle two-degree-of-freedom dynamics model (bicycle model), derive state equation and control input
- In vehicle steering control, how to handle "understeer" and "oversteer" problems? Design compensation strategy
- Derive relationship between vehicle sideslip angle, yaw rate and steering wheel angle
- Design LQR-based vehicle lateral controller (lane keeping), including state selection, weight matrix design and discretization implementation
- Implement model predictive control (MPC) to solve vehicle longitudinal control (car following) problem, considering actuator delay and road slope
- How to adjust PID controller parameters to adapt to different speeds and road conditions? Design adaptive PID strategy
- Design automatic parking control system to implement parallel and perpendicular parking functions, considering parking space detection and trajectory planning
- In high-speed driving, how to handle front wheel blowout and other emergency situations? Design emergency control strategy
- Implement vehicle stability control (ESC) to prevent sideslip and tail swing, design control algorithm based on tire force observation

### System Development (C++ Direction)
- Describe differences and usage scenarios of C++ four smart pointers (shared_ptr/unique_ptr/weak_ptr/auto_ptr)
- Implement thread-safe singleton pattern (C++11+), considering double-checked locking and static local variable schemes
- Explain RAII (Resource Acquisition Is Initialization) principle and how to apply it in autonomous driving system?
- How does memory alignment affect point cloud processing performance? Take Eigen library matrix as example
- Design efficient obstacle trajectory data structure to support real-time query and update
- Implement a memory pool to manage frequently allocated and released small objects in autonomous driving system
- Design software architecture of autonomous driving perception system, considering multi-threading, data pipeline and error handling
- How to design modular system to achieve low coupling and high cohesion between perception, decision and control modules?
- When deploying deep learning models on embedded platforms (such as Jetson AGX), what optimizations are needed?

### Embedded Software Engineer
- Explain the difference between RTOS (real-time operating system) and general operating system, why is it important in autonomous driving?
- Design multi-level interrupt system to ensure real-time performance of critical tasks (such as braking control), using Cortex-M NVIC priority grouping
- In multi-core RTOS, how to implement inter-task communication and synchronization? Compare mailbox, semaphore and message queue schemes
- Describe CAN bus frame structure, compare differences and applicable scenarios between standard frame and extended frame
- Implement CAN bus communication protocol, including ID allocation, arbitration mechanism and error handling
- Design vehicle system diagnosis scheme based on UDS (Unified Diagnostic Services), implement fault code reading and clearing
- Write GPIO control program to implement vehicle lights, wipers and other peripheral control
- Design ADC sampling program to read vehicle sensor (such as tire pressure, oil temperature) data, considering anti-interference and precision optimization
- In embedded systems, how to handle power management? Design low-power mode and wake-up mechanism

### SLAM & Localization Engineer
- Describe ORB-SLAM2/3 system workflow, including feature extraction, tracking, local mapping and loop closure detection
- How to solve scale drift problem in visual SLAM? Compare monocular, stereo and RGB-D schemes
- In dynamic scenes, how to detect and remove moving objects? Design method based on optical flow and semantic segmentation
- Implement key point cloud processing part of LOAM or Lego-LOAM algorithm, including feature extraction and matching
- In LiDAR SLAM, how to handle point cloud distortion caused by vehicle motion? Design motion compensation scheme
- Compare advantages and disadvantages of LiDAR-SLAM and visual-SLAM, how to fuse them in autonomous driving?
- Design visual+IMU+RTK+LiDAR fusion localization system, including time synchronization and extrinsic calibration
- Implement EKF/UKF-based multi-sensor fusion localization, fusing GPS, IMU and wheel speedometer data
- In urban canyon and tunnel where GPS signal is lost, how to ensure localization accuracy? Design auxiliary localization scheme

### HD Map Engineer
- Design high-precision map construction process based on LiDAR point cloud, including point cloud registration, feature extraction and map element generation
- How to evaluate HD map quality? Design evaluation metrics for accuracy, completeness and consistency
- In map construction, how to handle dynamic obstacles (such as moving vehicles)? Design dynamic object filtering and completion scheme
- Design efficient lane-level map data structure to support fast query and update
- How to implement incremental update of HD map? Design difference detection and transmission scheme to reduce bandwidth consumption
- On embedded devices, how to optimize map storage and retrieval? Design hierarchical indexing and caching mechanism
- Describe application scenarios of HD map in autonomous driving, such as localization, path planning and decision-making
- How to encode traffic rules (no left turn, speed limit) into HD map? Design map semantic representation
- In autonomous driving system, how to achieve fast matching (localization) between map and vehicle position? Design efficient search algorithm

### Testing Engineer
- Design test case library for L4 autonomous driving system, covering perception, decision and control functions
- How to test autonomous driving system performance in extreme weather (heavy rain, dense fog, ice and snow)? Design test scenarios
- Implement scenario-based testing to test autonomous driving system decision logic
- Compare V (verification) and V (validation) processes in autonomous driving testing, explain their respective purposes and methods
- Design safety testing scheme for autonomous driving system to verify safety degradation mechanism in failure situations
- In HIL (hardware-in-the-loop) testing, how to simulate sensors and actuators? Design test platform
- Design risk matrix for autonomous driving testing, identify high-risk scenarios and formulate testing strategies
- In real vehicle testing, how to collect and analyze data to optimize algorithms? Design data collection and analysis process
- For the "long-tail problem" (rare but dangerous scenarios) of autonomous driving system, how to design test cases?

### Model Deployment & Optimization Engineer
- Design model quantization scheme (FP32→FP16→INT8) to improve inference speed while maintaining accuracy
- Implement model pruning to remove redundant parameters and reduce model size, design pruning criteria and fine-tuning strategy
- In model distillation, how to design teacher model and student model? How to choose distillation loss function?
- For autonomous driving perception model, design model parallelism and data parallelism scheme to improve multi-GPU inference efficiency
- Implement ONNX model to TensorRT conversion and optimization, configure appropriate workspace and precision mode
- On embedded platform, how to optimize model inference performance? Compare GPU, NPU and CPU schemes
- Design autonomous driving model service architecture to support high concurrency and low latency inference
- In end-to-end system, how to optimize data preprocessing and post-processing pipeline to reduce overall latency?
- Implement model hot update, update model without restarting service to ensure service continuity

### General Questions
- What is the core difference between process and thread? How to choose in actual development?
- What are the inter-process communication (IPC) methods? Their advantages and disadvantages and applicable scenarios?
- What is the working principle of virtual memory? Why do we need virtual memory?
- What is deadlock? What are the four necessary conditions for deadlock? How to avoid deadlock?
- What is the difference between user mode and kernel mode? How to switch?
- What is the difference and correspondence between OSI seven-layer model and TCP/IP four-layer model?
- What is the process of TCP three-way handshake? Why do we need three-way handshake?
- How does TCP ensure reliable data transmission? What are the mechanisms?
- What is the difference between HTTP and HTTPS? What is the working principle of HTTPS?
- Complete process from entering URL to page display?
- What is the core idea of Von Neumann architecture?
- What is the composition and working principle of CPU?
- What is the working principle of Cache? Cache hit rate and failure types?
- What is the role of memory alignment? How does it affect program performance?
- What is the principle and role of DMA technology? Why can it improve IO performance?
- What is the difference and applicable scenarios between TCP and UDP?
- What are the common page replacement algorithms? Why is LRU commonly used?
- What are common HTTP status codes and their meanings?
- What are critical resources and critical sections? How to solve critical section problem?
- What are the classification and role of bus? Differences between data bus, address bus and control bus?

### C++
- [C++ High-frequency Interview Questions](https://github.com/0voice/cpp-learning-2025/blob/main/interview_questions/README.en.md)



## 💼 Job Board



- **Waymo** – [Apply Here](https://waymo.com/careers/)
- **Cruise** – [Apply Here](https://getcruise.com/careers/)
- **Zoox** – [Apply Here](https://zoox.com/careers)
- **Aurora** – [Apply Here](https://aurora.tech/careers/)
- **NVIDIA** – [Apply Here](https://nvidia.com/en-us/about-nvidia/careers/) 
- **Motional** – [Apply Here](https://motional.com/careers)
- **Applied Intuition** – [Apply Here](https://www.appliedintuition.com/careers)
- **Waabi** – [Apply Here](https://waabi.ai/careers/)
- **Oxa** – [Apply Here](https://oxa.tech/careers/)


## 🤝 Community & Contribution



Thank you for visiting!  
This repo aims to be the strongest C++ autonomous driving resource collection worldwide.

Contributions of any kind are extremely welcome — new projects, fixes, translations, interview questions, etc.

Star & Watch so you never miss an update!

