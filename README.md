# BronchoscopicDepthMapEstimation_Holoscan

Real-time depth map estimation from bronchoscope video, running the cGAN depth
estimation model from
[Visually_Navigated_Bronchoscopy](https://github.com/SNRLab/Visually_Navigated_Bronchoscopy)
on the [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk) platform.
Each camera frame is converted to grayscale and passed through the model, and
the predicted depth map is shown in 3D with the video frame as its color. The
video frame and depth map are also sent over OpenIGTLink, for example to 3D Slicer.

## Contents

- `DepthMapEstimation/`: the Holoscan application. It captures video from a
  camera (`/dev/video0`), runs the model as ONNX with TensorRT, displays the
  result with Holoviz along with frame rate and inference time, and sends
  `RGB_Image` and `Depth_Image` messages over OpenIGTLink (port 18939). Put the
  ONNX model in `DepthMapEstimation/model/` as `depthmap_exVivo_folded.onnx`.
  Settings are in `Holoscan_DepthMapEstimation.yaml`.
- `Torch_Standalone/`: the same pipeline in plain PyTorch, without Holoscan.
  It expects the trained generator at `model/6Level-ex-vivo-G_AB.pth`.
- `ImageDisplayServer.py`: a small OpenIGTLink server that displays the
  received RGB and depth images, for testing.

## License

Distributed under the BSD 3-Clause License. See [LICENSE](LICENSE).
`DepthMapEstimation/Holoscan_DepthMapEstimation.yaml` is based on NVIDIA
Holoscan sample code and keeps its Apache 2.0 license header.
