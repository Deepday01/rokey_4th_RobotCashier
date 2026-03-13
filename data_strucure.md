PackingPlanList
└── planList: List[PackingPlan]
    └── PackingPlan
        ├── task_index: int
        ├── item: ItemState
        │   ├── item_id: str
        │   ├── name: str
        │   ├── pose: Pose3D
        │   │   ├── x: float
        │   │   ├── y: float
        │   │   ├── z: float
        │   │   ├── roll: float
        │   │   ├── pitch: float
        │   │   └── yaw: float
        │   ├── size: Size3D
        │   │   ├── width: float
        │   │   ├── depth: float
        │   │   └── height: float
        │   └── durability: int
        │
        ├── placement: PlacementState
        │   ├── object_index: int
        │   └── pose: Pose3D
        │       ├── x: float
        │       ├── y: float
        │       ├── z: float
        │       ├── roll: float
        │       ├── pitch: float
        │       └── yaw: float
        │
        ├── stage_plan: StagePlan
        │   ├── pick_approach_pose: Pose3D
        │   ├── pick_pose: Pose3D
        │   ├── pick_retreat_pose: Pose3D
        │   ├── station_approach_pose: Pose3D
        │   ├── station_place_pose: Pose3D
        │   └── station_retreat_pose: Pose3D
        │
        ├── align_plan: AlignPlan
        │   ├── required: bool
        │   └── steps: List[AlignStep]
        │       └── AlignStep
        │           ├── rx_deg: float
        │           ├── ry_deg: float
        │           └── rz_deg: float
        │
        ├── box_plan: BoxPlan
        │   ├── station_pick_approach_pose: Pose3D
        │   ├── station_pick_pose: Pose3D
        │   ├── station_pick_retreat_pose: Pose3D
        │   ├── box_approach_pose: Pose3D
        │   ├── box_place_pose: Pose3D
        │   └── box_retreat_pose: Pose3D