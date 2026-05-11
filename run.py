import app.main as model
import app.camera.camera_runner as camera
import app.utils.load as load
import app.utils.calibrate_background as calibrate

calibrate.run_calibration()
# model.build_model("v2")
# model.test_model("v2")
# camera.run_camera_with_model("v2")