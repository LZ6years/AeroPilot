# UAV Agent MVP 璁″垝

## 1. MVP 鑼冨洿

MVP 鍙仛鍗曟満鍗?UAV 鐨?UE / AirSim 浠跨湡闂幆锛屽厛浠?CLI 璺戦€氫换鍔★紝涓嶅仛瀹屾暣 Electron 鍓嶇锛屼笉鍋氬鏈哄崗鍚岋紝涓嶅仛鐙珛鎵嬫満绔兘鍔涖€?
浼樺厛鏀寔浠诲姟锛?
```text
鍓嶉潰鍙宠浆浼氭湁涓€涓粦鑹茬殑璺伅锛屽彸杞潰鍚戣矾鐏悗鍋滀笅
```

杩欎釜浠诲姟涓嶅簲涓€娆℃€х敓鎴愬緢闀跨殑鍥哄畾鍔ㄤ綔搴忓垪锛岃€屽簲鎷嗘垚锛?
- 鏁翠綋璇箟鐩爣锛氬彸杞悗瀵绘壘榛戣壊璺伅锛屽苟璁?UAV 鏈濆悜瀹冩偓鍋溿€?- 鐭湡鎵ц绐楀彛锛氭瘡杞彧杈撳嚭灏戦噺鍙獙璇佺殑鍘熷瓙鍔ㄤ綔銆?- 瑙傛祴鏇存柊锛氭瘡娆″姩浣滃悗閲嶆柊璇诲彇 RGB/depth/pose锛屽啀鍐冲畾涓嬩竴姝ャ€?- VLM 鍒ゆ柇锛氱敤澶氭ā鎬佸ぇ妯″瀷鐞嗚В褰撳墠鍥惧儚鍜屼换鍔′笂涓嬫枃銆?- 鍥版儜鎻愰棶锛欰gent 涓嶇‘瀹氫笖缁х画椋炶鏈夐闄╂椂锛屽厛 hover锛屽啀鍚戠敤鎴锋彁闂€?
## 2. 闃舵鍒掑垎

C/S 杈圭晫浠?[C/S 鏋舵瀯鑱岃矗鍒掑垎](C:/Users/HP/Desktop/UAV-Agent/docs/cs_architecture.md) 涓哄噯銆?
```text
Agent Runtime
  -> LocalEnvClient
    -> LocalSimulationServer
      -> One AirVLN-style Scene
      -> One UAV
```

### M1锛歎AV 鍘熷瓙椋炴帶灏佽

鐩爣锛?
- 浠?AirNavigation 涓娊鍙?UAV 鍩虹鎿嶄綔銆?- 鍙皝瑁呭師瀛愰鎺у拰鍘熷瑙傛祴锛屼笉缁ф壙瀹冪殑鏁翠綋鏋舵瀯鍜屽湴鍥炬灦鏋勩€?
浜や粯锛?
- `src/adapters/airnavigation_uav.py`
- `src/adapters/uav_primitives.py`
- `src/tools/registry.py`
- `src/tools/dispatcher.py`

鍘熷瓙鍔ㄤ綔锛?
- `uav.connect`
- `uav.takeoff`
- `uav.hover`
- `uav.land`
- `uav.move_body`
- `uav.turn_relative`
- `uav.ascend`
- `uav.descend`
- `uav.get_pose`
- `uav.get_frames`

楠屾敹锛?
- 鍙互杩炴帴 UAV銆?- 鍙互璧烽銆佹偓鍋溿€佽浆鍚戙€佷綆閫熺Щ鍔ㄣ€侀檷钀姐€?- 鍙互鑾峰彇 pose銆丷GB銆乨epth銆?
### M2锛欰irVLN 椋庢牸鍦板浘璧勬簮涓庡満鏅鐞?
鐩爣锛?
- 鍙傝€?AirVLN 鐨?scene 绠＄悊鏂瑰紡绠＄悊鍦板浘璧勬簮銆?- 浣跨敤 `MapPackage` 鎻忚堪闈欐€佸湴鍥捐祫婧愩€?- 浣跨敤 `SceneInstance` 鎻忚堪涓€娆¤繍琛屼腑鐨勫湴鍥惧疄渚嬨€?- 鐢?`SimulationServer` 璐熻矗鏌ユ壘 scene銆佺敓鎴?AirSim settings銆佸垎閰嶇鍙ｃ€佸惎鍔ㄨ繘绋嬨€?- UnrealCV / gym_unrealcv 鏀惧埌鍚庣画 backend锛屼笉浣滀负绗竴闃舵涓绘ā鍨嬨€?
浜や粯锛?
- `src/env_client/base.py`
- `src/env_client/local_client.py`
- `src/env_client/schemas.py`
- `src/sim_server/server.py`
- `src/sim_server/scene_manager.py`
- `src/sim_server/process_manager.py`
- `src/sim_server/port_manager.py`
- `src/sim_server/backends/airvln_backend.py`
- `src/sim_server/backends/local_ue_backend.py`
- `configs/maps/airvln_maps.json`

楠屾敹锛?
- 鍙互浠庨厤缃姞杞?AirVLN 椋庢牸 `MapPackage`銆?- 鍙互鏍规嵁 `scene_id` 鏌ユ壘骞跺惎鍔?scene銆?- 鍙互鑷姩鐢熸垚 AirSim `settings.json`銆?- 鍙互鍒嗛厤 AirSim port 骞惰褰曞埌 `SceneInstance`銆?- Agent Runtime 鍙€氳繃 EnvClient 鑾峰彇 RGB/depth/pose銆?- Agent Runtime 鍙€氳繃 EnvClient 璇锋眰 UAV 鍘熷瓙鍔ㄤ綔銆?- 鍚庣画鍙互鎶?UnrealCV/gym_unrealcv 鎺ユ垚 SimulationServer backend銆?
### M3锛氫换鍔¤В鏋愪笌璇箟瑙勫垝

鐩爣锛?
- 灏嗙敤鎴疯嚜鐒惰瑷€瑙ｆ瀽鎴愮ǔ瀹氱殑璇箟浠诲姟銆?- 璇箟浠诲姟鍙弿杩版剰鍥俱€佺害鏉熷拰瀹屾垚鏉′欢锛屼笉鐩存帴灞曞紑涓哄畬鏁村姩浣滃垪琛ㄣ€?
浜や粯锛?
- `src/agent/semantic/task_parser.py`
- `src/agent/semantic/validator.py`
- `src/agent/semantic/schemas.py`
- `src/agent/planning/schemas.py`

楠屾敹锛?
- 杈撳叆涓枃浠诲姟鍚庤兘鐢熸垚缁撴瀯鍖栬涔夎鍒掋€?- schema 鏍￠獙澶辫触鏃舵嫆缁濇墽琛屻€?- 璁″垝涓嶄細鎶娾€滈潬杩?闈㈠悜鈥濊鍐欐垚鍘熷瓙鍔ㄤ綔銆?
### M4锛氱煭鏈熸粴鍔ㄦ墽琛屽櫒

鐩爣锛?
- 姣忚疆鏍规嵁褰撳墠鐘舵€佸拰璇箟璁″垝锛屽彧杈撳嚭鏈潵鍑犳鍘熷瓙鍔ㄤ綔銆?- 姣忔鎵ц鍚庨噸鏂拌娴嬪苟鏇存柊鐘舵€併€?- `connect`銆乣takeoff`銆佽捣椋炲悗绋冲畾鎮仠鍜屽垵濮嬭娴嬪睘浜?Runtime preflight锛屼笉灞炰簬 phase planner 鎴栨ā鍨嬪姩浣滆鍒掋€?
浜や粯锛?
- `src/agent/planning/model_action_planner.py`
- `src/agent/mission_state.py`
- `src/agent/safety.py`
- `src/runtime/main.py`

楠屾敹锛?
- phase 鐭湡绐楀彛涓嶅寘鍚?`uav.connect` / `uav.takeoff` / `uav.land` 绛夌敓鍛藉懆鏈熷姩浣溿€?- run record 鑳藉尯鍒?`control_connected`銆乣preflight_takeoff_completed`銆乣initial_observation_updated` 鍜屽悗缁?phase 浜嬩欢銆?- 涓嶄竴娆℃€ф墽琛岄暱璺濈 `move_forward`銆?- 鍔ㄤ綔鏈夐€熷害銆佽搴︺€佹椂闂淬€佽窛绂讳笂闄愩€?- 姣忚疆鎵ц鍚庨噸鏂拌鍙栬娴嬨€?- 寮傚父銆佽秴鏃躲€佷綆缃俊搴︽椂杩涘叆 hover銆?
### M5锛氬師濮嬭娴嬩笌鐘舵€佺鐞?
鐩爣锛?
- 鍩虹鎰熺煡灞傚彧璐熻矗鍘熷鏁版嵁锛屼笉鍋氶珮绾ц涔夈€?- 灏?EnvClient 杩斿洖鐨?observation 瑙勮寖鍖栦负 `RawObservation`锛屽苟鏋勫缓鍚庣画妯″瀷鍔ㄤ綔瑙勫垝鍙敤鐨?context銆?- 瑙嗚璇箟鍒ゆ柇鏆備笉鎷嗘垚鐙珛妯″潡锛岀涓€闃舵缁熶竴浜ょ粰 VLM 澶勭悊銆?
浜や粯锛?
- `src/perception/raw_observation.py`
- `src/perception/context_builder.py`
- `src/agent/mission_state.py`
- `src/runtime/record_writer.py`

楠屾敹锛?
- 鑳戒繚瀛樺師濮嬪浘鍍忋€佹繁搴︺€佷綅濮裤€佸姩浣滄棩蹇椼€?- 姣忚疆鍐欏叆 `observations/latest_observation.json` 鍜?`contexts/latest_context.json`銆?- context 鍖呭惈褰撳墠 phase銆佺洰鏍囥€佹渶杩戣娴嬨€佹渶杩戝姩浣滃拰瀹夊叏绾︽潫銆?- 鐩爣妗嗐€佺洰鏍囪窛绂荤瓑瑙嗚璇箟缁撴灉涓嶄綔涓哄繀濉姸鎬佸瓧娈点€?- VLM 鏈‘璁ょ洰鏍囨椂绯荤粺涓嶄細鍋囪宸茬粡鐪嬭鐩爣銆?
### M6锛歏LM 涓存椂鍐崇瓥灞?
鐩爣锛?
- 绗竴闃舵浣跨敤澶氭ā鎬佸ぇ妯″瀷澶勭悊鍥惧儚鍜屼笂涓嬫枃銆?- VLM 璐熻矗鈥滅湅鍥剧悊瑙?+ 涓嬩竴姝ュ缓璁€濓紝浣嗕笉鐩存帴鎵ц鎺у埗銆?- 绗竴鐗堝厛瀹炵幇 `ModelActionPlanner`锛屽熀浜?M5 context JSON 鐢熸垚淇濆畧鐭湡鍔ㄤ綔寤鸿锛岀湡瀹炲浘鍍忓妯℃€佽緭鍏ュ悗缁ˉ榻愩€?
浜や粯锛?
- `src/cognition/vlm_client.py`
- `src/cognition/context_builder.py`
- `src/cognition/schemas.py`
- `src/agent/planning/model_action_planner.py`
- `src/agent/planning/prompts/action_planner_system.txt`
- `configs/models/vlm.json`

楠屾敹锛?
- 鑳芥妸褰撳墠瑙傛祴鍜屼换鍔′笂涓嬫枃鍙戦€佺粰 VLM銆?- `run-task` 鍥哄畾浣跨敤 `ModelActionPlanner`锛汣LI 涓嶅啀鎻愪緵瑙勫垯/妯″瀷 planner 鍒囨崲鍙傛暟銆?- VLM 杈撳嚭蹇呴』閫氳繃 schema 鏍￠獙銆?- VLM 寤鸿鍔ㄤ綔蹇呴』缁忚繃 Safety Supervisor銆?- 浣庣疆淇″害鏃跺彧鍏佽瑙傚療銆佹偓鍋溿€佸皬瑙掑害鎵弿鎴栨彁闂€?- 鍚庣画绛夊畬鏁撮棴鐜窇閫氬悗锛屽啀鑰冭檻鏄惁鍋氳瑙夎涔夎兘鍔涙ā鍧楀寲銆?
### M6 鍚庣画澧炲己锛歅lan-Execute-Repair 閿欒鑷慨澶嶉棴鐜?
鐩爣锛?
- 鍩虹 pipeline 鍜岀湡瀹炲紩鎿庨摼璺窇绋冲悗锛屽紩鍏ョ粺涓€鐨勯敊璇弽棣堜笌妯″瀷淇鏈哄埗銆?- 褰?action plan銆乼ool 鍙傛暟銆丼afety銆佹墽琛屾垨瑙傛祴鍑虹幇閿欒鏃讹紝灏嗛敊璇粨鏋勫寲涓?`repair_context` 杩斿洖缁欐ā鍨嬶紝璁╂ā鍨嬮噸鏂拌鍒掓垨淇锛岃€屼笉鏄寔缁墿灞曡鍒欒ˉ涓併€?- 褰撳墠 `_normalize_move_body_args()` 杩欑被瑙勫垯褰掍竴鍖栧彧浣滀负鐭湡鍏煎灞傦紝鐢ㄤ簬绋冲畾鍩虹鍔ㄤ綔娴嬭瘯锛涘畠涓嶅簲缁х画鎵╁睍鎴愬ぇ鍨嬪弬鏁版槧灏勮〃銆?
鏈潵閾捐矾锛?
```text
ModelActionPlanner
  -> ToolContractValidator
    -> if invalid: ModelRepairPlanner
    -> SafetySupervisor
    -> ToolDispatcher
    -> if execution/observation failed: ModelRepairPlanner or safe hover
```

`repair_context` 搴旀槸缁撴瀯鍖栨暟鎹紝鑰屼笉鏄彧浼犺嚜鐒惰瑷€閿欒锛?
```json
{
  "error_type": "tool_argument_schema_error",
  "phase": "reverse_translation",
  "invalid_plan": {},
  "validation_errors": [],
  "allowed_tools": [],
  "latest_context": {}
}
```

杈圭晫锛?
- 褰撳墠闃舵涓嶅疄鐜板畬鏁?repair loop锛屽彧璁板綍璁捐鏂瑰悜銆?- 杩戞湡鍙厛鍙 action plan schema 閿欒鍋氫竴娆?repair銆?- 杩滄湡鍐嶇粺涓€澶勭悊鎵ц澶辫触銆佽娴嬪紓甯搞€佺洰鏍囨湭纭銆丼afety 鎷︽埅绛夐敊璇€?- 妯″瀷鏈€澶?repair 1-2 娆★紱淇鍚庝粛蹇呴』閫氳繃 schema validation 鍜?SafetySupervisor銆?- Safety 鐨勭‖鎬у畨鍏ㄧ瓥鐣ヤ笉鍏佽妯″瀷缁曡繃锛涚湡瀹炲紩鎿庢墽琛屽け璐ユ椂榛樿鍏?hover锛屽啀鍐冲畾鏄惁 repair銆?
鏈潵楠屾敹锛?
- `rear`銆乣duration_sec`銆乣x_m` 绛夊弬鏁伴敊璇彲鐢辨ā鍨嬩慨澶嶄负鍚堟硶 schema銆?- 鏈煡 tool 鍙敱妯″瀷淇涓哄厑璁稿伐鍏锋垨 `hover` / `observe`銆?- Safety 鎷︽埅浼氳褰曞師鍥狅紝浣嗘ā鍨嬩笉鑳借鐩?Safety 鍐崇瓥銆?- repair 澶氭澶辫触鏃朵换鍔¤繘鍏ュ畨鍏ㄧ瓑寰呮垨澶辫触鐘舵€併€?
### M7锛氬洶鎯戞彁闂笌瀹夊叏涓柇

鐩爣锛?
- 褰?Agent 涓嶇‘瀹氱户缁鍔ㄦ槸鍚︽纭垨瀹夊叏鏃讹紝涓诲姩鍋滀笅鏉ラ棶鐢ㄦ埛銆?
浜や粯锛?
- `src/agent/confusion_detector.py`
- `src/agent/clarification.py`
- `src/runtime/user_reply_handler.py`

楠屾敹锛?
- 瑙﹀彂鍥版儜鏃剁珛鍗?hover銆?- 姣忔鍙棶涓€涓棶棰樸€?- 鐢ㄦ埛鍥炲鍙互鏇存柊鐩爣銆佹柟鍚戙€佸€欓€夐€夋嫨鎴栧仠姝㈡寚浠ゃ€?- 鐢ㄦ埛瓒呮椂涓嶅洖澶嶆椂淇濇寔 hover 鎴栨寜閰嶇疆瀹夊叏闄嶈惤銆?
### M8锛氳繍琛岃褰曚笌 CLI

鐩爣锛?
- 鍙傝€?QiClaw 鐨?run store / records銆?- CLI 鏄涓€闃舵涓诲叆鍙ｃ€?
浜や粯锛?
- `src/runtime/run_store.py`
- `src/runtime/record_writer.py`
- `src/runtime/main.py`
- `runs/<run_id>/`

楠屾敹锛?
- 姣忔浠诲姟鏈?run id銆?- 姣忚疆淇濆瓨杈撳叆銆佽涔夎鍒掋€佺煭鏈熻鍒掋€佸姩浣溿€佽娴嬨€乂LM 鍒ゆ柇銆佸洶鎯戞彁闂€佺敤鎴峰洖澶嶃€侀敊璇€?- 淇濆瓨 `MapPackage`銆乣SceneInstance`銆丄irSim port銆乻ettings 璺緞鍜?scene process 淇℃伅銆?- 澶辫触鏃惰兘鐪嬪埌澶辫触闃舵鍜屽師鍥犮€?
### M9锛欵lectron 涓庨涔﹂鐣?
鐩爣锛?
- Electron 绛?UE CLI 瀹屾暣璺戦€氬悗鍐嶅仛銆?- 鎵嬫満绔厛涓嶅仛鐙珛 API锛屾湭鏉ュ浐瀹氭帴鍏ラ涔︺€?
浜や粯锛?
- `ui/electron/README.md`
- `src/integrations/feishu_gateway.py`
- runtime event schema

楠屾敹锛?
- 鏂囨。鏄庣‘ UI 鍙兘璋冪敤 runtime/gateway銆?- 椋炰功鍙綔涓烘湭鏉ユ秷鎭叆鍙ｅ拰閫氱煡鍏ュ彛銆?- 绗竴闃舵涓嶄緷璧栨墜鏈轰紶鎰熷櫒 API銆?
## 3. 浼樺厛绾?
鏈€楂樹紭鍏堢骇锛?
- UAV 鍘熷瓙椋炴帶灏佽銆?- AirVLN 椋庢牸 scene 璧勬簮閰嶇疆鍜屽惎鍔ㄣ€?- EnvClient / SimulationServer 鏈湴浼?C/S銆?- RGB/depth/pose 鍘熷瑙傛祴銆?- 鐭湡婊氬姩鎵ц銆?- VLM 涓存椂鍐崇瓥銆?- hover/timeout/confusion guard銆?
涓紭鍏堢骇锛?
- 鑷劧璇█鍒拌涔夎鍒掋€?- run store 鍜岃建杩硅褰曘€?- 涓诲姩鎻愰棶銆?- AirVLN 椋庢牸 scene 鍚姩 demo 楠岃瘉銆?
浣庝紭鍏堢骇锛?
- 瑙嗚璇箟鑳藉姏妯″潡鍖栥€?- UnrealCV / gym_unrealcv 瀵硅薄绾ф搷浣溿€?- 澶?UAV銆?- AirVLN C/S 瀹屾暣杩佺Щ銆?- Electron 鍓嶇銆?- 椋炰功鎺ュ叆銆?
## 4. 鎺ㄨ崘绗竴鏉?Demo

鏂囦欢锛?
```text
src/runtime/lamp_right_turn_demo.py
```

娴佺▼锛?
1. 璇诲彇 AirVLN 椋庢牸 `MapPackage`銆?2. Agent Runtime 閫氳繃 LocalEnvClient 璇锋眰 LocalSimulationServer 鍚姩 scene銆?3. LocalSimulationServer 鐢熸垚 AirSim `settings.json`銆佸垎閰?port銆佸惎鍔?scene process銆?4. EnvClient 鑾峰緱 `SceneInstance`銆?5. 杩炴帴 UAV銆?6. 璧烽鎴栫‘璁ゅ凡璧烽銆?7. 瑙ｆ瀽鐢ㄦ埛鎸囦护涓鸿涔夎鍒掋€?8. 鎵ц鐭湡绐楀彛锛氬厛灏忚搴﹀彸杞垨鍙宠浆鍒扮害 90 搴︺€?9. 鑾峰彇 RGB/depth/pose銆?10. 璋冪敤 VLM 鍒ゆ柇鏄惁鏈夐粦鑹茶矾鐏€欓€夈€?11. 鑻?VLM 缃俊搴﹁冻澶燂紝鍒欑敤灏忚搴?yaw 璋冩暣璁╃洰鏍囧眳涓€?12. 鑻?VLM 鍥版儜鎴栧彂鐜板涓€欓€夛紝鍒?hover 骞跺悜鐢ㄦ埛鎻愰棶銆?13. 婊¤冻瀹屾垚鏉′欢鍚?hover銆?14. 杈撳嚭 run 璁板綍銆?
## 5. 椋庨櫓

- AirNavigation 鏁翠綋鏋舵瀯涓嶉€傚悎缁ф壙锛屽彧鑳芥娊鍙?UAV 鍩虹鎿嶄綔銆?- AirVLN 鍦烘櫙鍖呰矾寰勩€丄irSim settings銆佺鍙ｅ垎閰嶅拰杩涚▼ ready 妫€娴嬮渶瑕佸厛楠岃瘉銆?- Windows 鏈湴鍚姩鏂瑰紡鍙兘鍜?AirVLN Linux packaged scene 涓嶅悓锛岄渶瑕?`local_ue_backend` 鍏煎銆?- LLM 鐩存帴鐢熸垚鍔ㄤ綔搴忓垪涓嶇ǔ瀹氾紝蹇呴』缁忚繃 schema 鏍￠獙鍜岀煭鏈熸墽琛屽櫒銆?- VLM 瑙嗚鍒ゆ柇鍙兘骞昏锛屽繀椤昏 Safety Supervisor 绾︽潫銆?- VLM 瑙嗚鍒ゆ柇涓嶇ǔ瀹氭椂锛岀郴缁熶笉搴旀墽琛屸€滈潬杩戠洰鏍団€濈瓑鍗遍櫓鍔ㄤ綔銆?
## 6. Phase 璇箟涓?M4/M6 鐩爣褰㈡€佽ˉ鍏?
M3 鐢熸垚鐨?phase 涓嶅啀琚涓哄浐瀹氭灇涓俱€俻hase name 鏄ā鍨嬪褰撳墠闃舵鐨勮涔夊懡鍚嶏紝涓昏鐢ㄤ簬鍚庣画妯″瀷鍔ㄤ綔瑙勫垝鍜屼汉宸ュ鐩橈紝鑰屼笉鏄瑙勫垯浠ｇ爜鐩存帴 `if/elif` 鍒嗗彂鍔ㄤ綔銆?
phase 缁撴瀯搴斿寘鍚樁娈典綅缃細

```json
{
  "index": 1,
  "total": 4,
  "name": "turn_to_search_possible_lamp_area",
  "meaning": "鏍规嵁鐢ㄦ埛鐨勫彸杞嚎绱紝璋冩暣瑙嗚鍘诲鎵惧彲鑳藉嚭鐜伴粦鑹茶矾鐏殑鍖哄煙"
}
```

褰撳墠宸蹭笉淇濈暀瑙勫垯鐗堢煭鏈?planner銆俽untime 鐨勭煭鏈熷姩浣滆鍒掑浐瀹氫娇鐢?`ModelActionPlanner`锛岀敱 `PlannerContext` 鎻愪緵鍔ㄤ綔濂戠害銆佽蹇嗘憳瑕佸拰瀹夊叏绾︽潫锛涚绾挎祴璇曢渶瑕佹棤妯″瀷琛屼负鏃讹紝搴旀敞鍏ユ祴璇?planner stub銆?
鏈€缁堢洰鏍囨槸寮曞叆 `ModelActionPlanner`锛?
```text
褰撳墠 phase锛堝惈 index/total/name/meaning锛?  + MissionState
  + 褰撳墠 RGB/depth/pose
  + 鏈€杩戝姩浣滃巻鍙?  + 鍙敤 tools/skills
  + Safety 绾︽潫
  -> 妯″瀷鐢熸垚 1-3 涓煭鏈熷姩浣滃缓璁?  -> SafetySupervisor 鏍￠獙/瑁佸壀
  -> ToolDispatcher 鎵ц
```

鍥犳 M4 褰撳墠闃舵浠モ€滃彲杩愯銆佸彲澶嶇洏銆佸畨鍏ㄥ厹搴曗€濅负涓伙紱鐪熸澶嶆潅鍦烘櫙涓嬬殑涓嬩竴姝ュ姩浣滈€夋嫨搴斾氦缁?M6/ModelActionPlanner锛岃€屼笉鏄緷璧栧浐瀹?phase 鍚嶃€?
