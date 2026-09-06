# Person Detector Output

The first mock implementation returns a stable structure:

- `detections`: a list of detected target candidates
- `mock`: whether the result came from a fake handler
- `observation_id`: exact ID from the image label; the demo detector rejects stale IDs
- `image_source`: optional image source, currently `latest_rgb`

Future real handlers should keep the same high-level shape so downstream semantic state updates can consume the result.
