# Duplicate X-dispatch failure and correction

The first hardware launch failed to resolve `cmd_vel_mux`, but left a Python
button process running (PID 9146). The subsequent launch created another
dispatcher (PID 10689). Both processed X. The roll controller activated; the
duplicate STRICT switch then failed, and the handler sent emergency stop before
roll START. See `startup_69f50016_retry.log` around timestamps 1789309219–220.
The earlier check for a running hardware manager alone missed the orphan handler.

The Pi subsequently rebooted (new boot 13f88ae2-7cbe-42c9-9f08-1f94c968d3c0).
No controller processes remained when checked. The cause of that reboot is not
established by this log.

The dispatcher now takes a nonblocking process lock before constructing any ROS
publishers. A duplicate exits without publishing stop or switching controllers.
The prepared startup wrapper also rejects an orphan dispatcher before launching
hardware. The command mux was built in `install-combined`, and startup resolves
all required command packages before launch.

Seven native Python tests passed, including two real subprocesses competing for
the dispatcher lock, followed by successful lock acquisition after release.
Installed/source dispatcher SHA256:
`0cb13750e1589068a1618e1ecf61b98c0cb675bf51747f362ac07aa1c59a1bc6`.
No motor restart was performed while installing this fix. Fresh physical startup
confirmation and calibration are required after the reboot.
