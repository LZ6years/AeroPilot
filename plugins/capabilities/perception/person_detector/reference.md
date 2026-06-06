# Person Detector Output

The first mock implementation returns a stable structure:

- `detections`: a list of detected target candidates
- `mock`: whether the result came from a fake handler
- `image_source`: the requested image source

Future real handlers should keep the same high-level shape so downstream semantic state updates can consume the result.
