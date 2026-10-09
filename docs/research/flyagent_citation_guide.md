# FlyAgent Introduction and Related Work Citation Guide

This guide maps the 41 entries in `flyagent_references.bib` to the current FlyAgent manuscript. The split between **core** and **candidate** controls citation density; it does not rank paper quality.

## 1. Recommended citation combinations

Use these compact groups in the main text. Avoid citing every paper in the corresponding literature pool.

| Manuscript claim | Recommended LaTeX citation | Note |
|---|---|---|
| Indoor embodied-navigation tasks are distinguished by the information that specifies the goal or route. | `~\cite{batra2020objectnav,krantz2023imagenav,anderson2018vision}` | Gives object category, goal image, and route-language examples without introducing PointNav in the prose. |
| Existing indoor VLN literature organizes tasks by instruction content and interaction form. | `~\cite{gu2022survey,wu2024taxonomy}` | Best support for the classification rationale in Sec. 2.1. |
| Existing aerial work covers several task and interaction settings, but no single benchmark-wide taxonomy is universally adopted. | `~\cite{xia2026aerialvlnsurvey,yao2025aeroverse_review}` | Phrase this as our synthesis of the surveys, not as a claim that the surveys explicitly prove absence. |
| Motion-instructed control. | `~\cite{wang2025uavflow}` | UAV-Flow defines the short-horizon Flying-on-a-Word setting. |
| Goal-described aerial navigation. | `~\cite{wang2025traveluav,lee2025citynav}` | TravelUAV/OpenUAV and CityNav both specify destinations without full route instructions. |
| Route-instructed aerial navigation. | `~\cite{liu2023aerialvln,gao2026openfly}` | Both pair long routes with landmark- and direction-rich instructions. |
| Dialog-based aerial navigation. | `~\cite{fan2023avdn,chen2026aerialvla}` | Use TravelUAV instead of AerialVLA when discussing assistant guidance rather than online questioning. |
| Top-down or geographic observation settings. | `~\cite{fan2023avdn,lee2025citynav}` | AVDN uses satellite/top-down scenes; CityNav adds geographic information. |
| Flight-data-trained navigation policies. | `~\cite{liu2023aerialvln,gao2026openfly,wang2025uavflow}` | These papers also contain the CMA, OpenFly-Agent, OpenVLA-UAV, and Pi-0-UAV baselines discussed in the draft. |
| Explicit maps, spatial representations, or hierarchical planning. | `~\cite{gao2024stmr,zhang2025citynavagent,xu2026geonav}` | Use in Sec. 2.2 after the second method family. |
| Tool-based embodied-navigation agents. | `~\cite{li2026agenticnav,chen2026harnessvln}` | Use for the final paragraph of Sec. 2.2. |
| OpenVLA-UAV or Pi-0-UAV inherits a general VLA backbone. | `~\cite{kim2025openvla,black2025pi0,wang2025uavflow}` | The UAV variants are baselines in UAV-Flow, not independent papers. |

### Suggested citation-ready Sec. 2.1 structure

```latex
Existing aerial VLN surveys distinguish target- and route-oriented instructions as well as single-instruction and dialog-based interaction, but they organize the growing benchmark landscape from different perspectives~\cite{xia2026aerialvlnsurvey,yao2025aeroverse_review}. Indoor embodied navigation commonly defines task settings by the information provided to the agent, such as an object category, a goal image, or a natural-language route instruction~\cite{batra2020objectnav,krantz2023imagenav,anderson2018vision}. Following this task-centered convention, we summarize representative aerial studies from the perspectives of task requirement, interaction form, and observation setting. These settings are not necessarily mutually exclusive.

Motion-instructed control requires a UAV to execute local flight commands such as turning, translation, or orbiting~\cite{wang2025uavflow}. Goal-described navigation specifies a destination through its appearance, direction, or environmental relations and requires the UAV to search for and approach it~\cite{wang2025traveluav,lee2025citynav}. Route-instructed navigation describes a long-range flight path using intermediate landmarks and directional changes~\cite{liu2023aerialvln,gao2026openfly}. Dialog-based navigation provides additional information through multi-turn communication during execution~\cite{fan2023avdn,chen2026aerialvla}, while top-down navigation uses satellite imagery, maps, or geographic observations for localization and navigation~\cite{fan2023avdn,lee2025citynav}.
```

## 2. Per-paper mapping

### Indoor and ground navigation

| Cite key | Tier | Type and setting | Best placement |
|---|---|---|---|
| `anderson2018evaluation` | Candidate | Navigation task definitions and evaluation; includes PointGoal-style formulations | Introduction or Sec. 2.1 only when discussing general task-definition conventions or metrics. |
| `anderson2018vision` | Core | R2R; natural-language route instruction | Sec. 2.1 after the indoor route-instruction example. |
| `jain2019stay` | Candidate | R4R; longer routes and instruction fidelity | Sec. 2.1 or experiments when route fidelity and path-sensitive metrics are discussed. |
| `ku2020room` | Candidate | RxR; multilingual, densely grounded route instructions | Sec. 2.1 when broadening route following beyond R2R. |
| `krantz2020beyond` | Core | VLN-CE; continuous environment and CMA-family baseline context | Sec. 2.2 when contrasting continuous control with discrete navigation graphs. |
| `batra2020objectnav` | Core | ObjectNav; object category as the goal specification | Sec. 2.1 in the indoor classification rationale. |
| `krantz2023imagenav` | Core | InstanceImageNav; image-specified target instance | Sec. 2.1 in the indoor classification rationale. |
| `qi2020reverie` | Candidate | Goal-oriented language navigation plus remote object grounding | Sec. 2.1 after goal-described navigation if an indoor analogue is useful. |
| `zhu2021soon` | Candidate | Scenario-oriented, description-based object navigation | Sec. 2.1 as stronger evidence for goal descriptions rather than route instructions. |
| `thomason2020visiondialog` | Core | CVDN; navigation from dialog history | Sec. 2.1 after the indoor dialog-navigation example. |
| `krantz2023iterative` | Candidate | Persistent, cross-episode navigation with retained memory | Introduction or method motivation when discussing persistent histories. |
| `khanna2024goat` | Candidate | Object-, language-, and image-goal lifelong navigation | Introduction when motivating heterogeneous task inputs and reusable models. |
| `zheng2024generalist` | Core | One model across R2R, CVDN, SOON, REVERIE, and related tasks | Introduction or Sec. 2.2 when motivating cross-task reuse. |
| `gu2022survey` | Core | Broad VLN task and method survey | First paragraph of Sec. 2.1. |
| `wu2024taxonomy` | Core | Taxonomy by instruction frequency and goal/route/interaction form | First paragraph of Sec. 2.1; strongest direct support for the adopted classification principle. |

### Aerial task settings and benchmarks

| Cite key | Tier | Type and setting | Best placement |
|---|---|---|---|
| `misra2018mapping` | Candidate | LANI navigation and CHAI instruction execution in simulated 3D environments | Sec. 2.1 only when giving historical context for language-guided aerial/simulated agents. |
| `fan2023avdn` | Core | AVDN benchmark and method; dialog interaction over top-down aerial imagery | Sec. 2.1 for both dialog-based and top-down navigation. |
| `liu2023aerialvln` | Core | AerialVLN benchmark plus CMA/LAG-style learned baselines; route following | Sec. 2.1 route setting and Sec. 2.2 learned-policy family. |
| `wang2025traveluav` | Core | OpenUAV platform, target-oriented TravelUAV data, and UAV-Need-Help assistance setting | Sec. 2.1 goal description/dialog guidance and Sec. 2.2 learned or hierarchical methods. |
| `lee2025citynav` | Core | CityNav benchmark; language goal with geographic information | Sec. 2.1 goal-described and top-down/geographic settings. |
| `wang2025uavflow` | Core | UAV-Flow benchmark; short-range language-conditioned flight control | Sec. 2.1 motion control and Sec. 2.2 VLA adaptation. |
| `gao2026openfly` | Core | OpenFly platform and benchmark plus OpenFly-Agent; long-range route following | Sec. 2.1 route setting and Sec. 2.2 learned-policy family. |
| `yao2026aeroverse` | Candidate | Broad UAV foundation-model benchmark suite spanning perception, reasoning, navigation, planning, and action | Introduction as a broad aerial embodied benchmark; do not present it as a pure closed-loop VLN benchmark. |
| `liu2026indooruav` | Candidate | Indoor UAV benchmark with long-horizon VLN and short-horizon VLA subsets | Sec. 2.1 when showing that route and motion settings can coexist in one benchmark. |
| `chen2026aerialvla` | Core | AerialVLA method plus UNOD online-dialog benchmark | Sec. 2.1 dialog setting or Sec. 2.2 learned policy; label the benchmark as UNOD. |
| `zhang2026apex` | Candidate | Aerial ObjectGoal navigation method evaluated on UAV-ON | Sec. 2.1 as a recent aerial object-search setting; it is primarily a method paper. |
| `zheng2026flight` | Candidate | FLIGHT benchmark combining fine-grained control and long-horizon instructions | Sec. 2.1 as evidence that task settings can overlap; currently an arXiv preprint. |
| `cai2026airnav` | Candidate | Large-scale real-world UAV VLN dataset with diverse natural instructions | Sec. 2.1 when discussing realism and instruction diversity; currently an arXiv preprint. |
| `wu2025aeroduo` | Candidate | DuAl-VLN task, HaL-13k data, and a dual-altitude method | Sec. 2.1 for multi-UAV target-oriented navigation; it is both a task and method paper. |
| `wang2025airstar` | Candidate | Voice/gesture-operated aerial assistant platform and system demonstration | Introduction for interaction diversity; do not describe it as a standard navigation benchmark. |

### Aerial surveys

| Cite key | Tier | Type and setting | Best placement |
|---|---|---|---|
| `xia2026aerialvlnsurvey` | Core | Aerial VLN survey emphasizing single-instruction versus dialog interaction and method families | First paragraph of Sec. 2.1. |
| `yao2025aeroverse_review` | Core | Peer-reviewed survey of aerial embodied vision-and-language navigation | Introduction or first paragraph of Sec. 2.1. |
| `chen2026uavvlnroadmap` | Candidate | UAV-VLN survey and research roadmap | Background or future-work discussion; currently an arXiv preprint. |
| `tian2025uavsmeetllms` | Candidate | Broader survey of LLM-enabled and agentic UAV systems | Introduction when motivating tool use, memory, and agentic UAVs; broader than VLN. |

### Methods, tool agents, and VLA foundations

| Cite key | Tier | Type and setting | Best placement |
|---|---|---|---|
| `gao2024stmr` | Core | Training-free LLM navigation using a semantic-topo-metric spatial representation | Sec. 2.2 map/representation/planning family. The latest arXiv version uses the updated title stored in the BibTeX file. |
| `zhang2025citynavagent` | Core | Hierarchical semantic planning and global topological memory | Sec. 2.2 map/representation/planning family; cite the ACL 2025 version. |
| `xu2026geonav` | Core | Dual-scale geospatial memory and coarse-to-fine agentic planning | Sec. 2.2 map/representation/planning family; cite the Pattern Recognition version. |
| `li2026agenticnav` | Core | Tool interface for action, depth queries, and observation recall | Final paragraph of Sec. 2.2. |
| `chen2026harnessvln` | Core | Shared navigation harness coordinating tools, memory, validation, execution feedback, recovery, and termination | Final paragraph of Sec. 2.2; currently an arXiv preprint. |
| `kim2025openvla` | Candidate | General OpenVLA backbone | Sec. 2.2 only when explaining the origin of OpenVLA-UAV. |
| `black2025pi0` | Candidate | General VLA flow-model backbone | Sec. 2.2 only when explaining the origin of Pi-0-UAV. |

## 3. Naming and attribution rules

- **LAG** is a baseline/training design introduced with AerialVLN, so cite `liu2023aerialvln`; do not create a separate LAG entry.
- **OpenFly-Agent** is introduced in the OpenFly paper, so cite `gao2026openfly`.
- **OpenVLA-UAV** and **Pi-0-UAV** are UAV-Flow adaptations/baselines, so cite `wang2025uavflow`; add `kim2025openvla` or `black2025pi0` only when the backbone itself matters.
- **TravelUAV**, **OpenUAV**, and **UAV-Need-Help** belong to the same ICLR 2025 paper, cited as `wang2025traveluav`.
- **AerialVLA** is overloaded in the literature. `chen2026aerialvla` denotes the AAAI 2026 online-dialog paper and its UNOD benchmark, not the separate minimalist-control preprint.
- **CityNav** (`lee2025citynav`) is a dataset/benchmark; **CityNavAgent** (`zhang2025citynavagent`) and **GeoNav** (`xu2026geonav`) are methods evaluated on related urban aerial settings.
- **AeroVerse** (`yao2026aeroverse`) is the benchmark suite; **AeroVerse-Review** (`yao2025aeroverse_review`) is the survey.
- The five task settings in Sec. 2.1 are organizational views, not mutually exclusive classes. CityNav spans goal description and geographic observation; AVDN spans dialog interaction and top-down observation; IndoorUAV and FLIGHT span long- and short-horizon behavior.

## 4. Additional watch-list papers not included in the 41-entry core pool

These are relevant but currently less necessary for the drafted Introduction and Related Work. Add them only if the corresponding topic enters the prose or experiments.

| Paper | Why it is not in the current `.bib` pool |
|---|---|
| HUGE-Bench, arXiv:2603.19822 | High-level UAV VLA benchmark; not needed for the present five-setting explanation. |
| SpatialSky-Bench, arXiv:2511.13269 | Evaluates spatial intelligence rather than complete closed-loop navigation. |
| MM-UAVBench, arXiv:2512.23219 | Evaluates UAV-oriented perception, cognition, and planning rather than one navigation task. |
| AirHunt, arXiv:2601.12742 | Primarily a recent open-set aerial object-navigation method/system. |
| OpenVLN, arXiv:2511.06182 | Primarily a method evaluated on existing aerial navigation settings. |
| TouchDown, CVPR 2019 | Useful outdoor-ground bridge, but the current argument only needs indoor task-definition examples. |
| VLN-Video, AAAI 2024 | Outdoor VLN method, not necessary for the aerial task taxonomy. |

## 5. Minimal LaTeX integration

Copy `flyagent_references.bib` beside the paper's main `.tex` file or add its path to the bibliography command:

```latex
\bibliographystyle{ieeenat_fullname}
\bibliography{flyagent_references}
```

The official CVPR template determines the rendered reference style. The `.bib` file therefore keeps standard BibTeX entry types and protects acronyms such as `{UAV}`, `{VLN}`, `{LLM}`, and benchmark names from automatic lowercasing.
