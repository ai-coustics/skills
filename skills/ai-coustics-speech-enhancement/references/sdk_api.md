# AIC SDK Python API cheat-sheet

Written against the `aic-sdk` 3.x Python binding. It deliberately avoids naming model
versions: read those from the SDK and the manifest at run time (see "Model resolution").
Authoritative reference: https://docs.ai-coustics.com/reference/sdk/models and the
Python migration guides under https://docs.ai-coustics.com/reference/deprecated/.

## Versions the SDK exposes
```python
import aic_sdk as aic

aic.get_sdk_version()               # native core version, e.g. "0.24.0"
aic.get_compatible_model_version()  # artifact version this SDK loads, e.g. 7
```
Each SDK release loads exactly one artifact version. A model id is usable only if the
manifest lists it under that version. This is the only compatibility rule you need.

## Model resolution (what `_aic_models.py` does)
1. `GET https://artifacts.ai-coustics.io/manifest.json` →
   `{"models": {"<id>": {"versions": {"v7": {file, checksum, …}}}}}`.
2. Keep ids that have `f"v{aic.get_compatible_model_version()}"`.
3. Parse ids as `<family>[-<variant>][-<major.minor>][-<size>]-<rate>khz`, e.g.
   `quail-vf-2.2-l-16khz`. Family specs leave the version out; resolve to the newest,
   then the largest size.
4. Model type is inferred from the family: `vad-*` → `aic.Vad`, `tyto-*` →
   `aic.FileAnalyzer` / `aic.Analyzer`, everything else → `aic.Processor`. The SDK
   enforces this at object creation with `ModelTypeUnsupportedError`.

## Model loading
```python
model = aic.Model.from_file("/path/to/model.aicmodel")

# Download by id (checksum-verified, cached in download_dir; re-checks the manifest each call)
path = aic.Model.download("<artifact-id>", "~/.cache/aic-models")
model = aic.Model.from_file(path)

model.get_id()                    # internal build id, not the artifact id
model.get_optimal_sample_rate()   # Hz
model.get_optimal_block_size()    # samples at the optimal rate
```
`ModelDownloadError` means the id is not published for this SDK's artifact version.

## Enhancement: async processor
```python
processor = aic.ProcessorAsync(model, license_key)      # ModelTypeUnsupportedError if not an enhancement model

config = aic.ProcessorConfig.optimal(model, sample_rate=fs)   # non-native rates are resampled internally
await processor.initialize_async(config)

ctx = processor.get_context()
ctx.reset()                                              # clear state between files/streams

try:
    ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 1.0)   # 0.0–1.0
except aic.ParameterFixedError:
    pass                                                 # model has a fixed level
```
A synchronous `aic.Processor` has the same surface without `_async`.

## Latency compensation
`get_audio_delay()` is how many samples the processed audio leaves `process()` *behind*
its input. Pad the **tail** so the last samples are flushed out, then drop the **head**:
```python
latency = ctx.get_audio_delay()  # samples
padded = np.concatenate([audio, np.zeros(latency, dtype=np.float32)])
# ... process padded in blocks ...
output = output[latency:]        # aligned with the input, same length
```
Padding the head instead keeps the length but leaves the output delayed by `latency`.

## Block processing
```python
block_size = config.block_size
for start in range(0, padded.shape[0], block_size):
    chunk = padded[start:start + block_size]
    buf = np.zeros(block_size, dtype=np.float32)
    buf[:chunk.shape[0]] = chunk
    out_chunk = await processor.process_async(buf)
    output[start:start + chunk.shape[0]] = out_chunk[:chunk.shape[0]]
```

## Audio shape convention
- **Mono 1-D float32.** Downmix yourself: `audio.mean(axis=1)` for `(frames, channels)`.
- Read: `audio, fs = sf.read(path, dtype="float32")` · Write: `sf.write(path, output, fs)`

## VAD (dedicated class; see the ai-coustics-voice-activity-detection skill)
```python
config = aic.ProcessorConfig.optimal(vad_model, sample_rate=fs)
vad = aic.Vad(vad_model, license_key, config)
ctx = vad.get_context()          # is_speech_detected(), raw_vad_probability(),
                                 # get_prediction_delay() in samples, get/set_parameter
vad.process(block)               # one block of config.block_size samples; audio untouched
```
Feed the VAD the **original** input, not the enhanced output. `VadParameter.Sensitivity`
is a 0–1 probability threshold: higher = fewer detections. Defaults for Sensitivity,
SpeechHoldDuration and MinimumSpeechDuration come from the model file.

## Analysis / Tyto (see the ai-coustics-audio-insight skill)
```python
analyzer = aic.FileAnalyzer(tyto_model, license_key)
results = analyzer.analyze(samples, sample_rate, step_samples)   # one AnalysisResult per window
dims = [n for n in dir(aic.AnalysisResult) if not n.startswith("_")]   # discover dimensions
```
For live streams use `aic.analyzer_pair(model, license_key)` (Collector + Analyzer).

## License key
```python
license_key = os.environ["AIC_SDK_LICENSE_KEY"]   # never print it
```

## Errors worth catching
`ModelDownloadError`, `ModelTypeUnsupportedError`, `ModelVersionUnsupportedError`,
`ParameterFixedError`, `ParameterOutOfRangeError`, `LicenseExpiredError`,
`LicenseFormatInvalidError`, `AudioConfigUnsupportedError`.
