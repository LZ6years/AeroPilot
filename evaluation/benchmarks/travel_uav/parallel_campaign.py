"""TravelUAV prepare/validate/run/watch entry over the common campaign engine.

Use the explicit flyagent interpreter. Pass repeatable --config arguments for
test_seen, test_unseen_map and test_unseen_object when preparing a campaign.
"""
from pathlib import Path
from evaluation.benchmarks.core.campaign.parallel import main


if __name__ == "__main__":
    main(default_dataset="travel_uav", default_output=Path("G:/FlyAgent-Eval/travel_uav"))
