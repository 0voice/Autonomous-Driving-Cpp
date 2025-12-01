#  C++ 自动驾驶资源库
> **核心定位：** 这是一个专注于 **高性能 C++** 实现、**工程化落地** 和 **职业发展** 的自动驾驶资源精选集 🚗

## 📖 目录
- [🗺️ 学习路线图](#%EF%B8%8F-学习路线图)
- [✨ 核心算法讲解](#-核心算法讲解)
- [📚 学习资源（书籍/论文/课程）](#-学习资源书籍论文课程)
- [🛠️ 工具链](#-工具链)
- [🎓 面试八股文](#-面试八股文)
- [💼 招聘信息（2025最新）](#-招聘信息2025最新)
- [🤝 社区与贡献](#-社区与贡献)

## 🗺️ 学习路线图
<details>
<summary>点击展开</summary>
    
![Roadmap](./roadmap/roadmap.svg)

</details>

## ✨ 核心算法讲解
<details>
<summary>点击展开</summary>
    
- [数学与几何基础](core_algorithms/数学与几何基础)
  - [Eigen](core_algorithms/数学与几何基础#eigen)
  - [SO(3)、SE(3)、李代数](core_algorithms/数学与几何基础#so3se3李代数)
  - [四元数与旋转表示](core_algorithms/数学与几何基础#四元数与旋转表示)
  - [滤波器（KF/EKF/UKF/ESKF）](core_algorithms/数学与几何基础#滤波器kfekfukfesef)

- [感知](core_algorithms/感知)
  - [PointPillars](core_algorithms/感知#pointpillars)
  - [CenterPoint Voxel-to-BEV + CenterHead](core_algorithms/感知#centerpoint-voxel-to-bev--centerhead)
  - [多模态融合（激光雷达+相机）](core_algorithms/感知#多模态融合激光雷达相机)
  - [TensorRT 自定义插件开发](core_algorithms/感知#tensorrt-自定义插件开发)

- [定位](core_algorithms/定位)
  - [NDT 配准](core_algorithms/定位#ndt-配准)
  - [FAST-LIO 紧耦合](core_algorithms/定位#fast-lio-紧耦合)
  - [ESKF 误差状态卡尔曼](core_algorithms/定位#eskf-误差状态卡尔曼)
  - [GPS/IMU 紧耦合](core_algorithms/定位#gpsimu-紧耦合)

- [建图](core_algorithms/建图)
  - [离线建图](core_algorithms/建图#离线建图)
  - [在线回环检测](core_algorithms/建图#在线回环检测)
  - [高精地图与矢量地图](core_algorithms/建图#高精地图与矢量地图)

- [预测](core_algorithms/预测)
  - [多目标跟踪](core_algorithms/预测#多目标跟踪)
  - [意图预测](core_algorithms/预测#意图预测)
  - [轨迹预测](core_algorithms/预测#轨迹预测)

- [规划](core_algorithms/规划)
  - [Hybrid A* + Reeds-Shepp](core_algorithms/规划#hybrid-a--reeds-shepp)
  - [Lattice Planner](core_algorithms/规划#lattice-planner)
  - [EM Planner](core_algorithms/规划#em-planner)
  - [行为决策与状态机](core_algorithms/规划#行为决策与状态机)

- [控制](core_algorithms/控制)
  - [MPC 横纵向解耦](core_algorithms/控制#mpc-横纵向解耦)
  - [LQR 与最优控制](core_algorithms/控制#lqr-与最优控制)
  - [Stanley / Pure Pursuit](core_algorithms/控制#stanley--pure-pursuit)
  - [车辆动力学模型](core_algorithms/控制#车辆动力学模型)

- [端到端](core_algorithms/端到端)
  - [模仿学习](core_algorithms/端到端#模仿学习)
  - [端到端模型 C++ 部署](core_algorithms/端到端#端到端模型-c-部署)

- [仿真](core_algorithms/仿真)
  - [CARLA C++ Client](core_algorithms/仿真#carla-c-client)
  - [传感器仿真与同步](core_algorithms/仿真#传感器仿真与同步)
  - [场景库与交通流](core_algorithms/仿真#场景库与交通流)

- [中间件与通信](core_algorithms/中间件与通信)
  - [Fast-DDS / CycloneDDS](core_algorithms/中间件与通信#fast-dds--cyclonedds)
  - [some/IP + vsomeip](core_algorithms/中间件与通信#someip--vsomeip)
  - [Protobuf 序列化](core_algorithms/中间件与通信#protobuf-序列化)
 
</details>
