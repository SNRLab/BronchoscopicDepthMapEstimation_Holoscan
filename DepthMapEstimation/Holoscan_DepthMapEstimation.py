import os, sys
import cv2
import cupy as cp
import time
import math
import pyigtl
from argparse import ArgumentParser

# Import the Holoscan SDK modules
import holoscan
from holoscan.gxf import Entity
from holoscan.core import Application, Operator, OperatorSpec, arg_to_py_object
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator

class DepthPostProcessingOp(Operator):
  def __init__(self, *args, **kwargs):
    self.frameCount = 0
    self.lastTime = time.time()    
    self.Client = pyigtl.OpenIGTLinkClient("127.0.0.1", 18939)
    self.Client.start()
    super().__init__(*args, **kwargs)

  def setup(self, spec: OperatorSpec):
    spec.input("depth_in")
    spec.input("video_in")
    spec.input("timestamp_in")
    spec.output("out_DepthRGB")
    spec.output("out_specs")
    spec.param("in_tensor_name", "inference_output_tensor")

  def compute(self, op_input, op_output, context):
    messageDepth = op_input.receive("depth_in")

    # Latency calcuiation
    timestamp_in = op_input.receive("timestamp_in")
    currentTimestamp = time.time()

    inputDepth = messageDepth.get(self.in_tensor_name)

    # Depth processing
    cpDepth = cp.asarray(inputDepth)
    cpDepth = cp.moveaxis(cpDepth, [0,1], [-1,-2])
    cpDepth = (cpDepth[:,:,:,0] * 255.0).astype(cp.uint8)

    # Combine frames for display
    messageVideo = op_input.receive("video_in")
    inputVideo = messageVideo.get("source_video")
    cpVideo = cp.asarray(inputVideo).astype('uint8')
    
    # RGB output
    imageRGB = cp.asnumpy(cp.fliplr(cp.flipud(cpVideo)))
    imageRGBMessage = pyigtl.ImageMessage(imageRGB, device_name="RGB_Image")
    self.Client.send_message(imageRGBMessage, wait=False)

    # Depth Output
    imageDepth = cp.asnumpy(cp.fliplr(cp.flipud(cpDepth)))
    imageDepthMessage = pyigtl.ImageMessage(imageDepth, device_name="Depth_Image")
    self.Client.send_message(imageDepthMessage, wait=False)
    #cpDepth_pseudoRGB = cp.dstack((cpDepth,cpDepth,cpDepth))
    
    # Depth with RGB output
    textCoord = cp.zeros([1, 1, 3], dtype=cp.float32)
    textCoord[0][0][2] = 0.06
    out_message_DepthRGB = Entity(context)
    #cpDepth = 255 - cpDepth
    out_message_DepthRGB.add(holoscan.as_tensor(cpVideo[:,:,0:4]), "color_data_overlay")
    out_message_DepthRGB.add(holoscan.as_tensor(cpDepth), "depth_data_overlay")
    out_message_DepthRGB.add(holoscan.as_tensor(textCoord), "dynamic_text")

    # Framerate calculation; should match camera
    self.frameCount += 1
    deltaTime = time.time() - self.lastTime
    frameRate = cp.around(self.frameCount / deltaTime, 1)
    if self.frameCount > 20:
      self.frameCount = 0
      self.lastTime = time.time()

    # Latency calculation ; ms
    deltaLatencyTime = math.ceil((currentTimestamp - timestamp_in)*1000)
    statsString = f'FPS: {frameRate} ; Inference Time: {deltaLatencyTime} ms'

    specs = []
    spec = HolovizOp.InputSpec("dynamic_text", "text")
    spec.text = [statsString]
    specs.append(spec)

    op_output.emit(out_message_DepthRGB, "out_DepthRGB")
    op_output.emit(specs, "out_specs")



class GrayscaleOp(Operator):
  def __init__(self, fragment, *args, **kwargs):
    super().__init__(fragment, *args, **kwargs)

  def setup(self, spec: OperatorSpec):
    spec.input("in")
    spec.output("out")
    spec.output("timestamp_out")
    
  def compute(self, op_input, op_output, context):
    frameTensor = op_input.receive("in").get("source_video")

    timestamp = time.time()

    cpFrame = cp.asarray(frameTensor).astype('float32')
    grayFrame = self.rgbToGray(cpFrame)
    grayFrame = cp.ascontiguousarray(grayFrame)

    outMessage = Entity(context)
    outMessage.add(holoscan.as_tensor(grayFrame), "source_video")
    op_output.emit(outMessage, "out")
    op_output.emit(timestamp, "timestamp_out")

  def rgbToGray(self, rgbArray):
    red = rgbArray[:,:,0]
    green = rgbArray[:,:,1]
    blue = rgbArray[:,:,2]

    # Calculate the grayscale values
    grayArray = red * 0.3 + green * 0.59 + blue * 0.11

    # Reshape the grayscale values to match the original shape
    grayArray = grayArray.reshape(1, 200, 200)

    return grayArray


# Define the main application that runs the onnx model on the input stream and visualizes the results
class DepthMapEstimationApp(Application):
  def __init__(self):
    super().__init__()
    self.name = "Bronchoscopy Depth Map Estimation App"
    self.model_path = os.getcwd() + "/model"
    self.model_path_map = {"model": os.path.join(self.model_path, "depthmap_exVivo_folded.onnx")}
    self.video_dir = os.getcwd() + "/video"
    if not os.path.exists(self.video_dir):
      raise ValueError(f"Could not find video data: {self.video_dir=}")

  def compose(self):
    host_allocator = UnboundedAllocator(self, name="host_allocator")

    #source = VideoStreamReplayerOp(self, name="replayer", directory=self.video_dir, **self.kwargs("replayer"))
    source = V4L2VideoCaptureOp(self, name="camera", allocator=host_allocator, **self.kwargs("camera"))
    grayscaleProcessor = GrayscaleOp(self, name="grayscaleProcessor", pool=host_allocator)
    preprocessor = FormatConverterOp(self, name="preprocessor", pool=host_allocator, **self.kwargs("preprocessor"))
    inference = InferenceOp(self, name="inference", allocator=host_allocator, model_path_map=self.model_path_map, **self.kwargs("inference"))
    postprocessor = DepthPostProcessingOp(self, name="postprocessor", pool=host_allocator, *self.kwargs("postprocessor"))
    viz_DepthRGB = HolovizOp(self, name="holoviz_DepthRGB", allocator=host_allocator, **self.kwargs("holoviz_DepthRGB"))

    # Workflow definition
    self.add_flow(source, preprocessor, {("signal", "source_video")})
    self.add_flow(preprocessor, grayscaleProcessor, {("tensor", "in")})
    self.add_flow(grayscaleProcessor, inference, {("out", "receivers")})

    self.add_flow(preprocessor, postprocessor, {("tensor", "video_in")})
    self.add_flow(inference, postprocessor, {("transmitter", "depth_in")})
    self.add_flow(grayscaleProcessor, postprocessor, {("timestamp_out", "timestamp_in")})

    self.add_flow(postprocessor, viz_DepthRGB, {("out_DepthRGB", "receivers")})
    self.add_flow(postprocessor, viz_DepthRGB, {("out_specs", "input_specs")})


# Run the application
if __name__ == "__main__":
  config_file = os.path.join(os.path.dirname(__file__), "Holoscan_DepthMapEstimation.yaml")

  if len(sys.argv) >= 2:
    config_file = sys.argv[1]

  app = DepthMapEstimationApp()
  app.config(config_file)
  app.run()

