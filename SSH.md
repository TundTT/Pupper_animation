# SSH access to the robot

Practical notes from actually doing this repeatedly across several hardware sessions —
not a generic SSH tutorial. `pupper.local` (mDNS) is unreliable in practice; the steps
below are the path that has actually worked.

## 1. Find the robot's current IP

The robot's IP **changes** across reboots/reconnects — don't assume a previously-used
IP still works, and don't rely on `pupper.local`:

```sh
ping -c 2 pupper.local        # often fails to resolve even when the robot is up
```

If that fails (it usually will), read the IP directly off the robot:

- Its screen shows a network icon / connection info panel — open **Connection
  Information** and read the **IPv4 → IP Address** field.
- Double-check the digit count. `10.140.55.163` and `10.14.55.163` look almost
  identical at a glance and are on **different subnets** — the second one will time
  out with no useful error, and it's easy to misread off a small/angled screen photo.

Once you have the IP, confirm it's actually reachable before trying SSH:

```sh
ping -c 3 -W 2 <ip>
```

- **Ping fails entirely** ("Destination Host Unreachable" or 100% loss with no
  response): your machine and the robot are likely on genuinely different networks
  (e.g., your machine on a lab/office LAN, the robot on a phone hotspot or different
  Wi-Fi AP). Get both on the same network, or use a machine that's already on the
  robot's network.
- **Ping works fine**: proceed to SSH.

## 2. Connect

```sh
ssh pi@<ip>
```

- **The password changes** — ask for the current one rather than reusing an old one
  from notes or memory. Never hardcode it in a script or commit it anywhere.
- First connection to a given IP will prompt to accept the host key
  (`-o StrictHostKeyChecking=accept-new` if scripting this non-interactively).

### If SSH times out but ping just worked

This happened repeatedly across sessions and is **not** the same failure as "wrong
network": ping succeeds, then `ssh ... port 22: Connection timed out` right after,
usually caused by transient Wi-Fi jitter on the robot's side (often correlated with the
robot doing something current-hungry, like actuator homing). **Just retry** — a second
attempt moments later typically succeeds without needing a reboot or IP re-check first.
Re-confirm with `ping` again if the retry also fails.

### If the robot was just power-cycled or rebooted

Wait for it to fully boot (the on-robot display coming up is a good sign), then
**re-check the IP** — it very often changes across a full power cycle even if nothing
else about the network changed.

## 3. Before touching anything on the robot: check what's already there

**Do not assume `~` only has what you expect.** This robot (and likely others) can have
multiple, unrelated checkouts in the home directory — including forks belonging to
other people/projects that must not be clobbered. Before running anything destructive
or doing a fresh clone, look:

```sh
ls -la ~
find ~ -maxdepth 2 -iname '*pupper*' -o -iname '*robot-code*'
```

For any checkout you find, confirm whose it is and what branch it's on before touching
it:

```sh
cd <dir> && git remote -v && git branch --show-current
```

If it points at a different GitHub org/user than `TundTT/Pupper_animation`, or a branch
name you don't recognize, **leave it alone** — it's someone else's work, not this
project's checkout.

## 4. Running commands non-interactively (scripting)

For automation (not for a human typing at a terminal), `sshpass` avoids an interactive
password prompt:

```sh
sshpass -p '<password>' ssh -o ConnectTimeout=15 pi@<ip> "<command>"
```

Use a generous `-o ConnectTimeout` (10–15s) — the robot's Wi-Fi can be slow to respond
even when it's about to succeed. Never write the password into a file that gets
committed; pass it inline per-invocation and treat it as ephemeral.

See [LEG_LIFT_TESTING.md](LEG_LIFT_TESTING.md) and [WHEEL_TESTING.md](WHEEL_TESTING.md)
for what to actually do once you're connected.
