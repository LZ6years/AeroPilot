# 07 鍥剧ず姹囨€?
鏈枃浠堕泦涓繚瀛樿蹇嗙郴缁熺殑鍏抽敭鍥撅紝鏂逛究蹇€熻瘎瀹°€?
## 1. 瀹屾暣涓婁笅鏂囦富閾捐矾

```mermaid
flowchart TD
    U["UserInstruction<br/>鐢ㄦ埛杈撳叆"] --> TP["TaskParser<br/>鎸囦护瑙ｆ瀽"]
    TP --> SP["SemanticPlan<br/>璇箟闃舵璁″垝"]
    SP --> CP["CurrentPhaseSelector<br/>褰撳墠 phase / 灏忛樁娈?]

    CP --> R["ContextModuleRegistry"]
    FM["FlightMemory<br/>鍘嗗彶鎸囦护/濮挎€?鍥惧儚/鍔ㄤ綔"] --> R
    MM["MapMemory<br/>鍦板浘绾跨储"] --> R
    RM["RunMemory<br/>杩愯缁忛獙"] --> R

    R --> I["InstructionContextModule"]
    R --> IMG["ImageContextModule"]
    R --> P["PoseContextModule"]
    R --> ACT["ActionContextModule"]
    R --> MAP["MapContextModule"]

    I --> FR["ContextFragments"]
    IMG --> FR
    P --> FR
    ACT --> FR
    MAP --> FR

    FR --> C["ContextComposer"]
    FLY["FLYAGENT.md"] --> C
    USER["USER.md"] --> C
    SAFE["SafetyConstraints"] --> C

    C --> PC["PlannerContext<br/>contexts/latest_context.json"]
    PC --> PL["ModelActionPlanner"]
    PL --> SF["SafetySupervisor"]
    SF --> TD["ToolDispatcher"]
```

## 2. 鎬讳綋鍒嗗眰

```mermaid
flowchart TD
    A["FLYAGENT.md<br/>绯荤粺瑙勫垯/瀹夊叏杈圭晫"]
    B["USER.md<br/>鐢ㄦ埛鍋忓ソ/浜や簰涔犳儻"]
    C["FlightMemory<br/>杩炵画椋炶鐘舵€?]
    D["MapMemory<br/>椋炶繃鍖哄煙/鍦版爣绾跨储"]
    E["RunMemory<br/>杩愯璁板綍/澶嶇洏缁忛獙"]
    R["ContextModuleRegistry<br/>妯″潡娉ㄥ唽/鍔犺浇"]
    F["ContextComposer<br/>鍥哄畾澶栧３/鍚堝苟鍣?]

    C --> R
    D --> R
    E --> R
    R --> F
    A --> F
    B --> F

    F --> G["PlannerContext"]
    G --> H["ModelActionPlanner"]
    H --> I["SafetySupervisor"]
    I --> J["ToolDispatcher"]
```

## 3. 杩炵画鎸囦护

```mermaid
sequenceDiagram
    participant U as User
    participant TP as TaskParser
    participant PH as CurrentPhase
    participant FM as FlightMemory
    participant R as ContextModuleRegistry
    participant CC as ContextComposer
    participant PL as Planner

    U->>TP: instruction #1
    TP->>PH: SemanticPlan + current phase
    PH->>FM: update active_instruction
    FM->>R: history data + current phase data
    R->>CC: context fragments
    CC->>PL: PlannerContext
    PL->>FM: action/observation result appended

    U->>TP: instruction #2
    TP->>PH: new SemanticPlan + current phase
    FM->>FM: update active_instruction only
    FM->>R: keep flight history + new phase data
    R->>CC: context fragments
    CC->>PL: PlannerContext from current UAV state
```

## 4. 鍘嗗彶鍘嬬缉

```mermaid
flowchart TD
    A["memory/*.jsonl + artifacts<br/>鍘熷鍘嗗彶鍜屽ぇ鏂囦欢寮曠敤"] --> B["Module Selected Window<br/>妯″潡鎸?refs 璇诲彇鐭獥鍙?]
    A --> C["Rolling Summary<br/>鎸囦护/濮挎€?鍔ㄤ綔鏁堟灉/椋庨櫓鎽樿"]
    A --> D["Long-term Memories<br/>MapMemory / RunMemory"]

    B --> E["Context Modules"]
    C --> E
    D --> F["Recall Selection"]
    F --> E
    E --> G["PlannerContext"]
```

## 5. MapMemory 鍙洖

```mermaid
flowchart TD
    A["Current map_id / scene_id"] --> D["MapMemory Recall"]
    B["active_instruction"] --> D
    C["current pose / phase"] --> D
    E["MapMemory manifest"] --> D
    D --> F["selected_map_memory<br/>鏈€澶?3-5 鏉?]
    F --> G["ContextComposer"]
```

## 6. Reset 璁板繂杈圭晫

```mermaid
flowchart LR
    A{"Event"}
    A -->|鏅€氭柊鎸囦护| B["淇濈暀 FlightMemory<br/>鏇存柊 active_instruction"]
    A -->|reset pose| C["鍒锋柊椋炶杩炵画鎽樿<br/>reset_epoch += 1"]
    A -->|restart scene| D["鏂板缓 FlightMemory"]
    A -->|switch map| E["鏂板缓 FlightMemory<br/>鍙彫鍥炴柊 map memory"]

    C --> F["MapMemory 淇濈暀<br/>requires_reconfirmation=true"]
    D --> G["RunMemory 淇濈暀"]
    E --> G
```

## 7. 鎻掍欢鏋舵瀯鎬诲浘

```mermaid
flowchart TD
    A["StageContextInput<br/>SemanticPlan / current phase"] --> R["ContextModuleRegistry"]
    B["Memory Sources<br/>FlightMemory / MapMemory / RunMemory"] --> R

    R --> I["InstructionContextModule"]
    R --> IMG["ImageContextModule"]
    R --> P["PoseContextModule"]
    R --> ACT["ActionContextModule"]
    R --> M["MapContextModule"]

    I --> C["ContextComposer"]
    IMG --> C
    P --> C
    ACT --> C
    M --> C

    F["FLYAGENT.md"] --> C
    U["USER.md"] --> C
    S["Safety Constraints"] --> C

    C --> O["PlannerContext"]
    O --> PL["ModelActionPlanner"]
    PL --> SF["SafetySupervisor"]
```

## 8. 鐢ㄦ埛鏇挎崲 ImageContextModule

```mermaid
flowchart LR
    A["src/memory/context/custom_modules/image_memory/paper_x.py"] --> B["PaperXImageMemoryModule"]
    B --> C["configs/context_modules.json"]
    C --> D["ContextModuleRegistry"]
    D --> E["ImageContextModule = paper_x"]
    D --> F["Instruction/Pose/Action/Map = default"]
    E --> G["ContextComposer"]
    F --> G
    G --> H["PlannerContext"]
```

## 9. 鎺ㄨ崘鐩綍缁撴瀯

```mermaid
flowchart TD
    A["src/memory<br/>瀹樻柟璁板繂绯荤粺瀹炵幇"]
    B["src/memory/context/modules<br/>榛樿浜旂被 ContextModule"]
    C["src/memory/stores<br/>MapMemory / RunMemory store"]
    D["configs/context_modules.json<br/>妯″潡娉ㄥ唽閰嶇疆"]
    E["memory/FLYAGENT.md<br/>绯荤粺瑙勫垯"]
    F["memory/USER.md<br/>鐢ㄦ埛鍋忓ソ"]
    G["src/memory/context/custom_modules<br/>鐢ㄦ埛鑷畾涔夋ā鍧?]
    H["src/memory/context/custom_modules/image_memory/paper_x.py<br/>绀轰緥鍥剧墖鍘嗗彶妯″潡"]

    A --> B
    A --> C
    D --> B
    D --> G
    E --> A
    F --> A
    G --> H
```
## 10. Manifest + JSONL Store

```mermaid
flowchart TD
    RT["TaskRuntime"] --> FM["flight_memory.json<br/>manifest / recent refs"]

    RT --> INS["memory/instructions.jsonl"]
    RT --> ACT["memory/actions.jsonl"]
    RT --> OBS["memory/observations.jsonl"]
    RT --> POSE["memory/poses.jsonl"]
    RT --> FRAME["memory/frames.jsonl"]
    FRAME --> RGB["memory/artifacts/rgb"]
    FRAME --> DEPTH["memory/artifacts/depth"]

    FM --> R["ContextModuleRegistry"]
    INS --> I["InstructionContextModule"]
    ACT --> A["ActionContextModule"]
    OBS --> IMG["ImageContextModule"]
    OBS --> P["PoseContextModule"]
    POSE --> P
    FRAME --> IMG
    RGB --> IMG
    DEPTH --> IMG

    R --> I
    R --> A
    R --> IMG
    R --> P
    I --> C["ContextComposer"]
    A --> C
    IMG --> C
    P --> C
    C --> PC["PlannerContext"]
```

## 11. Observation 瀵归綈鍏崇郴

```mermaid
flowchart LR
    OBS["obs_0007<br/>observations.jsonl"] --> P["pose_0007<br/>poses.jsonl"]
    OBS --> RGB["frame_rgb_0007<br/>frames.jsonl"]
    OBS --> DEP["frame_depth_0007<br/>frames.jsonl"]
    OBS --> ACT["act_0006<br/>actions.jsonl"]
    OBS --> PH["phase_0003<br/>current phase"]
    RGB --> RGBFILE["artifacts/rgb/obs_0007.png"]
    DEP --> DEPFILE["artifacts/depth/obs_0007.npy"]
```
