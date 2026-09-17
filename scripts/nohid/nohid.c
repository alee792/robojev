/* librealsense 2.56 (pyrealsense2-macosx) segfaults on macOS 26 while opening the D455's IMU
   over HID (hidapi -> IOHIDDeviceCreate returns NULL -> IOHIDDeviceOpen(NULL)).  We do not use the
   IMU, so hide every HID device from hidapi: hid_enumerate() then returns nothing and the D455 is
   created as depth+colour only.  Load with DYLD_INSERT_LIBRARIES (see scripts/camserver.sh). */
#include <stdio.h>
#include <stdlib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/hid/IOHIDManager.h>

static CFSetRef no_devices(IOHIDManagerRef manager) {
    (void)manager;  /* hidapi calls CFSetGetCount on the result, so hand back an empty set, not NULL */
    return CFSetCreate(kCFAllocatorDefault, NULL, 0, &kCFTypeSetCallBacks);
}

__attribute__((used)) static const struct { const void *replacement; const void *original; }
interposers[] __attribute__((section("__DATA,__interpose"))) = {
    { (const void *)no_devices, (const void *)IOHIDManagerCopyDevices },
};

__attribute__((constructor)) static void nohid_loaded(void) {
    if (getenv("NOHID_VERBOSE")) fprintf(stderr, "[nohid] loaded: HID devices hidden from librealsense\n");
}
