"""Read one frame + the arm pose (no motion), validate the camera extrinsic against the table
plane, detect objects, and save an annotated image. Usage: .venv/bin/python scripts/probe_camera.py [out.jpg]"""
import sys, json, time
import cv2, numpy as np
from robojev.perception.camclient import CamClient
from robojev.perception.geometry import Intrinsics, cam_to_base
from robojev.perception.detect import Detector

out = sys.argv[1] if len(sys.argv) > 1 else "probe.jpg"
import trossen_arm as t
d = t.TrossenArmDriver()
d.configure(t.Model.wxai_v0, t.StandardEndEffector.wxai_v0_follower, "192.168.1.5", False)
pose = list(d.get_cartesian_positions()); d.cleanup()
print("EE pose", [round(x, 4) for x in pose])
cam = CamClient(); intr = Intrinsics(cam.info)
f = cam.frame()
R, tt = cam_to_base(pose)
print("camera position in base", np.round(tt, 3), "optical z axis in base", np.round(R[:, 2], 3))
det = Detector(intr)
t0 = time.time(); dets, info = det.run(f.color, f.depth_m, pose); dt = time.time() - t0
print("detect ms", round(dt * 1000), json.dumps(info))
img = f.color.copy()
for i, o in enumerate(dets):
    print(f"  obj{i}: base_xyz={np.round(o.base_xyz,3).tolist()} h={o.height:.3f} w={o.width:.3f} color={o.color_name} n={o.n_points} px={o.pixel}")
    cv2.circle(img, o.pixel, 8, (0, 0, 255), 2)
    cv2.putText(img, f"{o.color_name} x{o.base_xyz[0]:.2f} y{o.base_xyz[1]:.2f} h{o.height:.2f}", (o.pixel[0] + 10, o.pixel[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.imwrite(out, img); print("wrote", out)
